"""An engineering-team assistant with graph memory, running on GLM 5.3 Flash via OpenRouter.

The point of this example is the part a vector store cannot do. The model is asked a
question whose answer is never stated in any single stored sentence and never mentioned
in the question. anatid finds it by walking two hops through the entity graph.

It also shows the three things that separate a memory from a log:

  supersede    a belief is replaced, the old one is closed rather than deleted
  as_of        any read can be replayed as the database saw the world at an earlier instant
  provenance   a fact can be traced back to the raw episode it came from, and to who wrote it

Writes go through an approval gate, so the model proposes a change to its memory and a
person decides whether it lands. Every write goes through the gate. Reads run
straight through.

Setup:

    pip install "anatid" openai
    export OPEN_ROUTER_KEY=sk-or-...        # or put open_router_key= in a .env beside this file

    python examples/glm_openrouter_agent.py

The script writes a database to ./glm_demo.anatid and deletes it on the next run.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

from anatid import Anatid, utcnow

MODEL = "z-ai/glm-5.3-flash"
DB_PATH = pathlib.Path(__file__).with_name("glm_demo.anatid")
AUTO_APPROVE = "--auto-approve" in sys.argv or not sys.stdin.isatty()


def load_key() -> str:
    for var in ("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    for parent in (pathlib.Path(__file__).parent, *pathlib.Path(__file__).parents):
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                key, _, value = line.partition("=")
                if key.strip().lower() in ("open_router_key", "openrouter_api_key"):
                    return value.strip().strip("\"'")
    sys.exit("Set OPEN_ROUTER_KEY, or put open_router_key= in a .env file.")


# --------------------------------------------------------------------------------------
# The tools the model can call. Reads run immediately; writes need a person to say yes.
# --------------------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search the team's memory. Combines BM25 text search with a two-hop walk "
                "of the entity graph, so it returns facts connected to the subject even "
                "when they do not contain the search words. Pass seed_entity when the "
                "question is about a specific person, project or service."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "words to search for"},
                    "seed_entity": {
                        "type": "string",
                        "description": "an entity name to walk the graph from, e.g. 'Ada'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a new durable fact. Requires human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "the people, projects or services this fact is about",
                    },
                },
                "required": ["content", "entities"],
            },
        },
    },
]

WRITE_TOOLS = {"remember"}


def approve(name: str, args: dict) -> bool:
    detail = args.get("content") or json.dumps(args)
    print(f"\n    [approval needed] {name}: {detail}")
    if AUTO_APPROVE:
        print("    [auto-approved]")
        return True
    return input("    approve? [y/N] ").strip().lower().startswith("y")


def run_tool(db: Anatid, name: str, args: dict) -> str:
    if name in WRITE_TOOLS and not approve(name, args):
        return "The person declined this write. The memory was not changed."

    if name == "recall":
        hits = db.recall(
            args["query"], seed_entity=args.get("seed_entity"), k=5, on_stale_fts="ignore"
        )
        if not hits:
            return "No memories matched."
        return json.dumps(
            [
                {
                    "content": h.memory.content,
                    "about": list(h.about),
                    "found_by": [
                        arm
                        for arm, rank in (
                            ("text", h.text_rank),
                            ("graph", h.graph_rank),
                            ("vector", h.vector_rank),
                        )
                        if rank is not None
                    ],
                }
                for h in hits
            ]
        )

    if name == "remember":
        m = db.remember(args["content"], entities=args["entities"], writer="glm-5.3-flash")
        db.rebuild_fts_index()
        return f"Stored as memory {m.memory_id}."

    return f"Unknown tool {name}."


def ask(client, db: Anatid, question: str, history: list) -> str:
    """One turn. Loops until the model stops calling tools.

    reasoning_details from each assistant message is passed back unmodified on the next
    request, so the model continues its chain of thought instead of restarting it.
    """
    history.append({"role": "user", "content": question})
    print(f"\n  user: {question}")

    for _ in range(6):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            extra_body={"reasoning": {"enabled": True}},
        )
        message = response.choices[0].message

        entry = {"role": "assistant", "content": message.content}
        if getattr(message, "reasoning_details", None):
            entry["reasoning_details"] = message.reasoning_details
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {"name": t.function.name, "arguments": t.function.arguments},
                }
                for t in message.tool_calls
            ]
        history.append(entry)

        if not message.tool_calls:
            print(f"  agent: {message.content}")
            return message.content or ""

        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"    -> {call.function.name}({json.dumps(args)[:90]})")
            result = run_tool(db, call.function.name, args)
            print(f"    <- {result[:160]}")
            history.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    return "(stopped after six tool rounds)"


def current_relations(db: Anatid) -> list[tuple[str, str, str]]:
    """Every relates edge the database still believes, as (source, target, rel_kind).

    Current means both valid_to and tx_to are null: unrelate closes an edge by stamping
    tx_to on the version it corrects and inserting a successor whose valid_to is the
    closing instant, so a query that tests only valid_to keeps returning the corrected one.
    """
    return db.execute(
        """
        select s.name, d.name, coalesce(r.rel_kind, 'relates_to')
        from edges_relates r
        join entities s on s.entity_id = r.src and s.valid_to is null and s.tx_to is null
        join entities d on d.entity_id = r.dst and d.valid_to is null and d.tx_to is null
        where r.valid_to is null and r.tx_to is null
        order by s.name, d.name
        """
    ).fetchall()


def main() -> None:
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())

    for suffix in ("", ".wal", ".shadow", ".tmp"):
        pathlib.Path(str(DB_PATH) + suffix).unlink(missing_ok=True)

    with Anatid.open(DB_PATH, tenant=1, embedding_dim=64) as db:
        # ---------------------------------------------------------------------------
        # Day one. Someone briefs the assistant. The graph is the shape of the team.
        # ---------------------------------------------------------------------------
        print("=" * 78)
        print("SETUP: what the team told the assistant on 2026-03-01")
        print("=" * 78)

        db.relate("Ada", "Project Kestrel", rel_kind="leads")
        db.relate("Project Kestrel", "ingest-service", rel_kind="owns")
        db.relate("Bo", "ingest-service", rel_kind="maintains")
        db.relate("Project Kestrel", "postgres-primary", rel_kind="depends_on")

        briefing = [
            ("Ada leads Project Kestrel.", ["Ada", "Project Kestrel"]),
            ("Bo maintains the ingest-service and is the person to page for it.",
             ["Bo", "ingest-service"]),
            ("The ingest-service is owned by Project Kestrel.",
             ["ingest-service", "Project Kestrel"]),
            ("postgres-primary has a nightly vacuum window at 02:00 UTC.",
             ["postgres-primary"]),
            ("Project Kestrel ships on Fridays.", ["Project Kestrel"]),
        ]
        for content, entities in briefing:
            db.remember(content, entities=entities, writer="onboarding",
                        episode="Team onboarding doc, 2026-03-01")
            print(f"  stored: {content}")
        db.rebuild_fts_index()

        day_one = utcnow()
        time.sleep(0.01)

        # ---------------------------------------------------------------------------
        # The question no single stored sentence answers.
        # ---------------------------------------------------------------------------
        print()
        print("=" * 78)
        print("1. THE TWO-HOP QUESTION")
        print("=" * 78)
        print("  No stored sentence contains both 'Ada' and 'page'. Bo is never mentioned")
        print("  in the question. The answer needs Ada -> Kestrel -> ingest-service -> Bo.")

        history = [
            {
                "role": "system",
                "content": (
                    "You are an engineering-team assistant with a graph memory. "
                    "Always call recall before answering a question about the team, "
                    "passing seed_entity when the question names a person or project. "
                    "Answer only from what recall returns. Be brief."
                ),
            }
        ]
        ask(client, db, "Ada's project is paging. Who should I wake up, and why?", history)

        # ---------------------------------------------------------------------------
        # A belief changes. The old one is closed, not deleted.
        # ---------------------------------------------------------------------------
        print()
        print("=" * 78)
        print("2. THE WORLD CHANGES (supersede)")
        print("=" * 78)

        bo_fact = next(
            m for m in db.recall_2hop("ingest-service") if m.content.startswith("Bo maintains")
        )
        # The sentence and the graph change together. Bo's maintains edge is closed in the
        # same transaction that opens Cy's, so no read can see two maintainers at once and a
        # failure part-way leaves the handover unapplied rather than half applied.
        with db.transaction():
            replacement = db.supersede(
                bo_fact.memory_id,
                "Cy maintains the ingest-service. Bo moved to Project Harrier on 2026-04-15.",
                entities=["Cy", "ingest-service", "Bo"],
                writer="handover-notes",
                episode="Handover notes, 2026-04-15: Bo -> Harrier, Cy takes ingest-service.",
            )
            closed = db.unrelate("Bo", "ingest-service", rel_kind="maintains")
            db.relate("Cy", "ingest-service", rel_kind="maintains")
        db.rebuild_fts_index()
        print(f"  superseded memory {bo_fact.memory_id} -> {replacement.memory_id}")
        print(f"  old fact still stored, is_current={db.get(bo_fact.memory_id).is_current}")
        print(f"  closed {closed} maintains edge from Bo, opened one from Cy")
        standing = [
            f"{src} {kind} {dst}" for src, dst, kind in current_relations(db) if kind == "maintains"
        ]
        print(f"  every maintains edge the database still believes: {', '.join(standing)}")

        ask(client, db, "Same question again. Who do I page for Ada's project?", history)

        # ---------------------------------------------------------------------------
        # A write the model proposes, gated on a human.
        # ---------------------------------------------------------------------------
        print()
        print("=" * 78)
        print("3. A WRITE THE MODEL PROPOSES (human in the loop)")
        print("=" * 78)

        ask(
            client,
            db,
            "Remember this: the ingest-service must not be deployed during the "
            "postgres-primary vacuum window.",
            history,
        )

        # ---------------------------------------------------------------------------
        # The parts that need no model at all.
        # ---------------------------------------------------------------------------
        print()
        print("=" * 78)
        print("4. TIME TRAVEL AND PROVENANCE (no model involved)")
        print("=" * 78)

        def maintainer(memories):
            for m in memories:
                if "maintains" in m.content:
                    return m.content
            return "(nothing on record)"

        print("  who maintained ingest-service, as the database saw it on day one:")
        print(f"    {maintainer(db.as_of(day_one).recall_2hop('ingest-service', limit=20))}")
        print("  and as it stands now:")
        print(f"    {maintainer(db.recall_2hop('ingest-service', limit=20))}")
        print("  the day-one answer is still retrievable because supersede closed the old")
        print("  row instead of deleting it.")

        p = db.provenance(replacement.memory_id)
        print(f"\n  why do we believe memory {replacement.memory_id}?")
        print(f"    supersession chain: {len(p.chain)} versions, writers {sorted(p.writers)}")
        for episode in p.episodes:
            print(f"    source: {episode.content}")

        stats = db.stats()
        print(
            f"\n  one file, {DB_PATH.stat().st_size / 1024:.0f} KB: "
            f"{stats['memories']} memories ({stats['current_memories']} current), "
            f"{stats['entities']} entities"
        )


if __name__ == "__main__":
    main()
