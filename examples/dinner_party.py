"""A household assistant with graph memory, running on GLM 5.3 Flash via OpenRouter.

Someone is cooking on Friday and asks whether there is a problem with the menu. The
question names two dishes and a day. It names no guest and no ingredient. No stored
sentence contains both 'pesto' and 'Priya'. Word search finds the recipe cards and
stops, because the answer lives in the edges between sentences, and one guest carries
an adrenaline pen.

The database holds twenty-six facts by the time the question is asked. Six of them carry
the question. The rest is household noise: the oven, the boiler, the car, who walks home,
which bread is better. recall returns eight. The point of the demonstration is which
eight.

The script also shows the three things that separate a memory from a log:

  supersede    a belief is replaced, the old one is closed rather than deleted
  as_of        any read can be replayed as the database saw the world at an earlier instant
  provenance   a fact can be traced back to the raw episode it came from, and to who wrote it

Writes go through an approval gate, so the model proposes a change to its memory and a
person decides whether it lands. Every write goes through the gate. Reads run straight
through.

Setup:

    pip install "anatid" openai
    export OPEN_ROUTER_KEY=sk-or-...        # or put open_router_key= in a .env file

    python examples/dinner_party.py
    python examples/dinner_party.py --scenario oncall
    python examples/dinner_party.py --auto-approve

The script writes a database to ./dinner_demo.anatid and deletes it on the next run.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import textwrap
from collections import deque
from itertools import pairwise

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import scenarios
from anatid import Anatid

MODEL = "z-ai/glm-5.3-flash"
DB_PATH = pathlib.Path(__file__).with_name("dinner_demo.anatid")

# recall walks two hops out from the seed, so a memory it picks up at the frontier can be
# about one entity further out again. Paths longer than this were not part of the walk.
RECALL_HOPS = 2

# The retrieval budget the model's recall tool spends, against a corpus several times
# larger. Every rank printed below is a rank out of this many returned rows.
RECALL_K = 8
WIDTH = 86


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
# The tools the model can call. Reads run immediately, writes need a person to say yes.
# --------------------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search memory. Combines BM25 text search with a two-hop walk of the "
                "entity graph, so it returns facts about things connected to the seed "
                "even when those facts contain none of the search words. That is the "
                "point: a guest's allergy never mentions the dish, and a dish's recipe "
                "never mentions the guest. Always pass seed_entity, naming the event, "
                "person or thing the question is about, and read every returned fact "
                "before answering, including the ones that look unrelated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "words to search for"},
                    "seed_entity": {
                        "type": "string",
                        "description": (
                            "an entity name to walk the graph out from, for example "
                            "'Friday dinner' or 'Ada'"
                        ),
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
                        "description": "the people, events or things this fact is about",
                    },
                },
                "required": ["content", "entities"],
            },
        },
    },
]

WRITE_TOOLS = {"remember"}


def approve(name: str, args: dict, auto: bool) -> bool:
    detail = args.get("content") or json.dumps(args)
    print(f"\n    [approval needed] {name}: {detail}")
    if auto:
        print("    [auto-approved]")
        return True
    return input("    approve? [y/N] ").strip().lower().startswith("y")


def run_tool(db: Anatid, name: str, args: dict, state: dict) -> str:
    if name in WRITE_TOOLS and not approve(name, args, state["auto_approve"]):
        return "The person declined this write. The memory was not changed."

    if name == "recall":
        seed = args.get("seed_entity")
        hits = db.recall(args["query"], seed_entity=seed, k=RECALL_K, on_stale_fts="ignore")
        # Kept so the explanation afterwards can quote the call the model actually made
        # and the rank the answer actually came back at, rather than a fresh read.
        state.setdefault("recalls", []).append({"query": args["query"], "seed": seed, "hits": hits})
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


def ask(client, db: Anatid, question: str, history: list, state: dict) -> str:
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
            print(f"\n  agent: {message.content}")
            return message.content or ""

        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"    -> {call.function.name}({clip(json.dumps(args), 110)})")
            result = run_tool(db, call.function.name, args, state)
            print(f"    <- {clip(result, 190)}")
            history.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "(stopped after six tool rounds)"


# --------------------------------------------------------------------------------------
# Reading the graph back out of the database, and walking it.
# --------------------------------------------------------------------------------------


def read_edges(db: Anatid) -> list[tuple[str, str, str]]:
    """Every current relates edge, as (source name, target name, rel_kind).

    Current means both columns null. unrelate closes an edge by stamping tx_to on the
    version it corrects and inserting a successor whose valid_to is the closing instant, so
    a query that tests only valid_to keeps returning the corrected version for ever.
    """
    rows = db.execute(
        """
        select s.name, d.name, coalesce(r.rel_kind, 'relates_to')
        from edges_relates r
        join entities s on s.entity_id = r.src and s.valid_to is null and s.tx_to is null
        join entities d on d.entity_id = r.dst and d.valid_to is null and d.tx_to is null
        where r.valid_to is null and r.tx_to is null
        """
    ).fetchall()
    seen, edges = set(), []
    for src, dst, rel_kind in rows:
        if (src, dst, rel_kind) not in seen:
            seen.add((src, dst, rel_kind))
            edges.append((src, dst, rel_kind))
    return edges


def about_map(db: Anatid) -> dict[int, set[str]]:
    """Every current memory, as memory_id -> the names of the entities it is about."""
    rows = db.execute(
        """
        select a.src, e.name
        from edges_about a
        join entities e on e.entity_id = a.dst and e.valid_to is null and e.tx_to is null
        where a.valid_to is null and a.tx_to is null
        """
    ).fetchall()
    out: dict[int, set[str]] = {}
    for memory_id, name in rows:
        out.setdefault(int(memory_id), set()).add(name)
    return out


def shortest_paths(edges, seed: str, target: str, limit: int = 3) -> list[list[str]]:
    """Every shortest undirected path from seed to target, up to `limit` of them."""
    if seed == target:
        return [[seed]]
    adjacency: dict[str, set[str]] = {}
    for src, dst, _ in edges:
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    parents: dict[str, list[str]] = {}
    depth = {seed: 0}
    queue = deque([seed])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour not in depth:
                depth[neighbour] = depth[node] + 1
                parents[neighbour] = [node]
                queue.append(neighbour)
            elif depth[neighbour] == depth[node] + 1:
                parents[neighbour].append(node)
    if target not in depth:
        return []

    paths: list[list[str]] = []

    def walk(node: str, tail: list[str]) -> None:
        if len(paths) >= limit:
            return
        if node == seed:
            paths.append([seed] + tail)
            return
        for parent in parents.get(node, ()):
            walk(parent, [node] + tail)

    walk(target, [])
    return paths


def distances(edges, seed: str) -> dict[str, int]:
    adjacency: dict[str, set[str]] = {}
    for src, dst, _ in edges:
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
    depth = {seed: 0}
    queue = deque([seed])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour not in depth:
                depth[neighbour] = depth[node] + 1
                queue.append(neighbour)
    return depth


def edge_between(edges, a: str, b: str) -> str:
    for src, dst, rel_kind in edges:
        if (src, dst) == (a, b):
            return f"{src} {rel_kind} {dst}"
        if (src, dst) == (b, a):
            return f"{src} {rel_kind} {dst}"
    return f"{a} -- {b}"


def walked_paths(db: Anatid, hits, seed: str, question: str) -> list[list[str]]:
    """The longest paths the two-hop walk actually covered, computed from the edges.

    Candidates are the entities the returned facts are about, plus any entity the
    question names, kept to what a two-hop recall can reach. The longest of those is
    where the walk had to go to connect the seed to the answer.
    """
    edges = read_edges(db)
    depth = distances(edges, seed)
    names = {src for src, _, _ in edges} | {dst for _, dst, _ in edges}

    candidates = {name for hit in hits for name in hit.about}
    candidates |= {name for name in names if name.lower() in question.lower()}
    reach = {
        name: depth[name]
        for name in candidates
        if depth.get(name, 999) <= RECALL_HOPS + 1 and depth.get(name, 0) > 0
    }
    if not reach:
        return []

    furthest = max(reach.values())
    endpoints = sorted(name for name, d in reach.items() if d == furthest)
    paths = []
    for endpoint in endpoints:
        paths.extend(shortest_paths(edges, seed, endpoint, limit=2))
    return paths[:4]


STOPWORDS = {
    "the",
    "and",
    "for",
    "any",
    "are",
    "was",
    "with",
    "that",
    "this",
    "have",
    "has",
    "you",
    "your",
    "our",
    "she",
    "her",
    "his",
    "him",
    "they",
    "them",
    "who",
    "why",
    "what",
    "when",
    "how",
    "should",
    "would",
    "could",
    "can",
    "will",
    "not",
    "but",
    "from",
    "into",
    "about",
    "there",
    "here",
    "then",
    "than",
    "some",
    "just",
    "get",
    "got",
    "one",
    "two",
    "all",
    "out",
    "off",
    "now",
    "its",
    "it's",
    "i'm",
    "does",
    "did",
    "make",
    "made",
    "same",
    "again",
    "still",
    "which",
    "were",
    "been",
    "being",
}


def content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def key_fact(hits, paths: list[list[str]], answer: str = ""):
    """The fact on the walked path that word search could not have returned.

    Candidates are facts the graph arm returned whose entities all lie on the path. More
    than one usually qualifies, so the one picked is the one the model's answer actually
    used, measured by how many of its words appear in the answer. With no answer text to
    compare against, the best-ranked candidate wins.
    """
    nodes = {name for path in paths for name in path}
    endpoints = {path[-1] for path in paths if path}
    answer_words = content_words(answer)

    def used(hit) -> int:
        return len(content_words(hit.memory.content) & answer_words)

    candidates = [h for h in hits if h.text_rank is None and h.about and set(h.about) <= nodes]
    if candidates:
        return max(candidates, key=used) if answer_words else candidates[0], True
    for hit in hits:
        if set(hit.about) & endpoints:
            return hit, hit.text_rank is None
    for hit in hits:
        if set(hit.about) & nodes:
            return hit, hit.text_rank is None
    return (hits[0], hits[0].text_rank is None) if hits else (None, False)


def path_kinds(edges, paths: list[list[str]]) -> set[str]:
    """The words in the rel_kinds the walk crossed, for example {invites, reacts, contains}."""
    kinds: set[str] = set()
    for path in paths:
        for a, b in pairwise(path):
            for src, dst, rel_kind in edges:
                if {src, dst} == {a, b}:
                    kinds |= {w for w in rel_kind.lower().split("_") if len(w) > 2}
    return kinds


def graph_only_fact(db: Anatid, seed: str, question: str, answer: str, paths):
    """The fact on the walked path that the question's own words cannot reach.

    Found by walking, not by asking a search engine: every memory the two-hop walk from
    the seed reaches, kept to the ones whose entities all lie on the path and that share
    no word with the question. Several usually qualify. The one that wins is the one that
    states an edge the walk crossed, which is what the walk went out there to collect;
    then the one the model's answer used most; then the older belief. Returns (memory,
    position, reach), the position being where the graph arm put it among the facts the
    walk reached.
    """
    nodes = {name for path in paths for name in path}
    question_words = content_words(question)
    answer_words = content_words(answer)
    kinds = path_kinds(read_edges(db), paths)
    about = about_map(db)
    reach = db.recall_2hop(seed, limit=100)

    candidates = [
        (i, m)
        for i, m in enumerate(reach, start=1)
        if about.get(m.memory_id)
        and about[m.memory_id] <= nodes
        and not (content_words(m.content) & question_words)
    ]
    if not candidates:
        return None, None, len(reach)
    position, memory = max(
        candidates,
        key=lambda pair: (
            len(content_words(pair[1].content) & kinds),
            len(content_words(pair[1].content) & answer_words),
            -pair[1].created_at.timestamp(),
        ),
    )
    return memory, position, len(reach)


def rank_in(hits, memory_id: int):
    """(rank, text_rank) for one memory inside a recall result, or (None, None)."""
    for hit in hits:
        if hit.memory.memory_id == memory_id:
            return hit.rank, hit.text_rank
    return None, None


def best_recall(recalls: list[dict]) -> dict | None:
    """The recall call whose result carried a fact the text arm never saw.

    The model makes several calls. The one worth quoting is the one that returned a fact
    with a graph rank and no text rank, because that is the row word search could not
    have produced. Falling back, the call that returned the most.
    """
    if not recalls:
        return None
    graph_only = [
        call
        for call in recalls
        if any(h.text_rank is None and h.graph_rank is not None for h in call["hits"])
    ]
    pool = graph_only or recalls
    return max(pool, key=lambda call: len(call["hits"]))


def clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def rule(title: str = "") -> None:
    print()
    print("=" * WIDTH)
    if title:
        print(title)
        print("=" * WIDTH)


def wrap(text: str, indent: str = "  ") -> None:
    for line in textwrap.wrap(text, width=WIDTH - len(indent)):
        print(indent + line)


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", default=scenarios.DEFAULT, choices=sorted(scenarios.SCENARIOS)
    )
    parser.add_argument("--auto-approve", action="store_true")
    args = parser.parse_args()

    scenario = scenarios.SCENARIOS[args.scenario]
    state = {"auto_approve": args.auto_approve or not sys.stdin.isatty()}

    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())

    for suffix in ("", ".wal", ".shadow", ".tmp"):
        pathlib.Path(str(DB_PATH) + suffix).unlink(missing_ok=True)

    with Anatid.open(DB_PATH, tenant=1, embedding_dim=64) as db:
        # -----------------------------------------------------------------------------
        # What the database was told, and when.
        # -----------------------------------------------------------------------------
        rule(scenario.title)
        wrap(scenario.one_liner)
        print()
        wrap(scenario.explain["setup"])

        scenarios.build(db, scenario)

        print()
        print(f"  all {len(scenario.facts)} facts, in the order they were told:")
        for day, fact in scenarios.timeline(scenario):
            print(f"    {day:>18}  {fact.writer:<14}  {fact.content}")
        print()
        print("  and the shape of the world, told at the same time:")
        for src, dst, rel_kind in scenario.relations:
            print(f"    {src} {rel_kind} {dst}")

        # -----------------------------------------------------------------------------
        # The question no single stored sentence answers.
        # -----------------------------------------------------------------------------
        rule("1. THE QUESTION")
        wrap(scenario.explain["two_hop"])
        print()
        wrap(scenario.explain["stakes"])

        corpus = db.stats()["memories"]
        print()
        print(
            f"  what word search alone returns from those {corpus} facts, with no seed "
            f"and no graph:"
        )
        for hit in db.recall(scenario.question, k=5, on_stale_fts="ignore"):
            print(f"    {hit.memory.content}")

        history = [{"role": "system", "content": scenario.system_prompt}]
        state["recalls"] = []
        answer = ask(client, db, scenario.question, history, state)

        # Everything below is read back out of the database. The path is computed from the
        # edges, and the ranks are the ranks the model's own recall calls returned.
        seed = scenario.seed_entity
        edges = read_edges(db)
        call = best_recall(state["recalls"])
        if call is None:
            call = {
                "query": scenario.question,
                "seed": seed,
                "hits": db.recall(
                    scenario.question, seed_entity=seed, k=RECALL_K, on_stale_fts="ignore"
                ),
            }
        hits = call["hits"]
        paths = walked_paths(db, hits, seed, scenario.question)
        winner, position, reach = graph_only_fact(db, seed, scenario.question, answer, paths)
        text_rank = None

        print()
        print(f"  what the graph added, walking out from {seed}:")
        if winner is not None:
            shared = sorted(content_words(scenario.question) & content_words(winner.content))
            # Every line below is measured against this run's database, not asserted.
            wide = db.recall(scenario.question, seed_entity=seed, k=corpus, on_stale_fts="ignore")
            fused_rank, text_rank = rank_in(wide, winner.memory_id)
            call_rank, _ = rank_in(hits, winner.memory_id)
            print(f"    the fact that carried the answer: {winner.content}")
            print(f"    words it shares with the question: {', '.join(shared) or 'none'}")
            print(
                "    the text arm, given the question's own words: "
                + ("never returns it" if text_rank is None else f"rank {text_rank}")
            )
            print(
                f"    the graph arm, walking out from {seed}: returns it, "
                f"{position} of the {reach} facts the walk reaches"
            )
            above = sum(
                1 for h in wide if fused_rank and h.rank < fused_rank and h.text_rank is not None
            )
            print(
                f"    the two arms fused, with the question as the query: rank "
                f"{fused_rank or 'not returned'} of {len(wide)}, with {above} of the rows "
                f"above it put there by shared words"
            )
            print(
                f"    the model's own call, recall(query={call['query']!r}, "
                f"seed_entity={call['seed']!r}):"
            )
            print(
                f"      returned {len(hits)} of the {corpus} stored facts, and "
                + (
                    f"this one came back at rank {call_rank}"
                    if call_rank
                    else "did not return this one"
                )
            )
        for path in paths:
            print(f"    the path the graph walked: {' -> '.join(path)}")
            for a, b in pairwise(path):
                print(f"      {edge_between(edges, a, b)}")
        print()
        if text_rank is None:
            wrap(
                "The question as typed is a weak query and a strong seed. Its words pull "
                "the text arm toward recipe cards, which is why the fused rank for the raw "
                "sentence sits outside the budget. The seed pulls the graph arm toward the "
                "guests, and the assistant's own narrower question brings the fact back "
                "inside it. No query built from the question's words reaches this row, "
                "because they are not in it."
            )
        else:
            wrap(
                "Word search does reach this row here, because the question and the fact "
                "happen to share a word. The dinner story is the sharper demonstration, "
                "because there the two share none."
            )

        # -----------------------------------------------------------------------------
        # A belief changes. The old one is closed, not deleted.
        # -----------------------------------------------------------------------------
        rule("2. THE WORLD CHANGES (supersede)")
        wrap(scenario.explain["supersede"])
        print()
        print(f"  new information, from {scenario.supersede.writer}:")
        wrap(scenario.supersede.episode, indent="    ")

        correction = scenarios.apply_supersede(db, scenario)
        old, replacement = correction.old, correction.new
        print()
        print(f"  closed:  {old.content}")
        print(f"  wrote:   {replacement.content}")
        print(f"           stored as memory {replacement.memory_id}")
        print(
            f"  the closed row is still there, and reports is_current="
            f"{db.get(old.memory_id).is_current}"
        )

        # The sentence is half the correction. The other half is the edges, moved in the
        # same transaction, and read back out of the database here rather than asserted.
        print()
        print("  the edges the same transaction moved:")
        for src, dst, rel_kind in correction.closed_relations:
            print(f"    closed:  {src} {rel_kind} {dst}")
        for src, dst, rel_kind in correction.opened_relations:
            print(f"    opened:  {src} {rel_kind} {dst}")
        moved_kinds = {rel_kind for _, _, rel_kind in correction.closed_relations}
        standing = [edge for edge in read_edges(db) if edge[2] in moved_kinds]
        if moved_kinds:
            print(
                f"  every {', '.join(sorted(moved_kinds))} edge the database still believes, "
                f"read back:"
            )
            for src, dst, rel_kind in standing:
                print(f"    {src} {rel_kind} {dst}")
            if not standing:
                print("    (none)")

        state["recalls"] = []
        ask(client, db, scenario.followup_question, history, state)

        call2 = best_recall(state["recalls"])
        hits2 = (
            call2["hits"]
            if call2
            else db.recall(
                scenario.followup_question, seed_entity=seed, k=RECALL_K, on_stale_fts="ignore"
            )
        )
        print()
        print("  the paths the graph walked this time:")
        for path in walked_paths(db, hits2, seed, scenario.followup_question):
            print(f"    {' -> '.join(path)}")
        print()
        wrap(
            "The memory and the edges moved in one transaction, so nothing can read a "
            "database where the sentence has changed and the graph has not. The closed edge "
            "is not deleted. A read as_of an instant before the change still walks it, which "
            "is what step 4 does."
        )

        # -----------------------------------------------------------------------------
        # A write the model proposes, gated on a human.
        # -----------------------------------------------------------------------------
        rule("3. A WRITE THE MODEL PROPOSES (human in the loop)")
        wrap(scenario.explain["approval"])
        state["recalls"] = []
        ask(client, db, scenario.write_request, history, state)

        # -----------------------------------------------------------------------------
        # The parts that need no model at all.
        # -----------------------------------------------------------------------------
        rule("4. TIME TRAVEL AND PROVENANCE (no model involved)")
        wrap(scenario.explain["as_of"])

        def matching(memories):
            return {m.content: m for m in memories if scenario.asof_match in m.content}

        past_at = scenarios.parse_time(scenario.asof_time)
        then = matching(db.as_of(past_at).recall_2hop(scenario.seed_entity, limit=40))
        now = matching(db.recall_2hop(scenario.seed_entity, limit=40))

        print()
        print(f"  {scenario.asof_label_past}:")
        for line in sorted(then) or ["(nothing on record)"]:
            print(f"    {line}")
        print(f"  {scenario.asof_label_now}, every line marked:")
        out_of_reach = False
        for line in sorted(set(then) | set(now)) or ["(nothing on record)"]:
            # The mark is read off the row, not off whether today's walk returned it. A
            # correction that closes an edge can put a line out of the seed's two hops
            # while leaving the belief standing, and calling that closed would be a lie.
            memory = then.get(line) or now[line]
            live = db.get(memory.memory_id)
            if live is not None and not live.is_current:
                mark = "closed"
            elif line not in then:
                mark = "written since"
            elif line not in now:
                mark = "still true"
                out_of_reach = True
            else:
                mark = "unchanged"
            print(f"    {mark:<13}  {line}")
        if out_of_reach:
            print()
            wrap(
                f"A line marked still true is one the database still believes and the walk "
                f"from {scenario.seed_entity} no longer reaches in two hops, because the edge "
                f"that led to it is the one the correction closed."
            )
        print()
        wrap(
            "The March note is unchanged and the belief it was written under is closed. "
            "Both are still readable, so the note can be explained rather than second "
            "guessed. Nothing was overwritten."
        )

        print()
        wrap(scenario.explain["provenance"])
        trail = db.provenance(replacement.memory_id)
        print()
        print(f"  the versions behind memory {replacement.memory_id}, the belief that stands now:")
        print(
            f"    supersession chain: {len(trail.chain)} versions, "
            f"writers {', '.join(sorted(trail.writers))}"
        )
        for episode in sorted(trail.episodes, key=lambda e: e.created_at):
            print(f"    {episode.writer}, {scenarios.format_day(episode.created_at)}:")
            wrap(episode.content, indent="      ")

        stats = db.stats()
        print()
        print(
            f"  one file, {DB_PATH.stat().st_size / 1024:.0f} KB: "
            f"{stats['memories']} memories ({stats['current_memories']} current), "
            f"{stats['entities']} entities"
        )
        if winner is not None:
            reached_by = (
                "It shares no word with the question, so no amount of word search reaches it."
                if text_rank is None
                else "The question happens to share a word with it, so word search reaches it too."
            )
            wrap(
                f"The fact that carried the first answer was one of {corpus} stored when "
                f"the question was asked. {reached_by} The walk from {seed} reached it "
                f"along the path printed in step 1."
            )
            # That path is a statement about the graph as it was then. Step 2 may have
            # closed one of the edges it crosses, and saying otherwise would be a claim
            # this run has already disproved.
            crossed = {pair for path in paths for pair in pairwise(path)}
            closed_on_path = [
                f"{src} {rel_kind} {dst}"
                for src, dst, rel_kind in correction.closed_relations
                if (src, dst) in crossed or (dst, src) in crossed
            ]
            if closed_on_path:
                wrap(
                    f"The correction in step 2 closed {', '.join(closed_on_path)}, which that "
                    f"path crosses, so the walk does not run that way any more. The edge is "
                    f"closed rather than deleted, so a read as of an earlier instant still "
                    f"crosses it."
                )
        print()


if __name__ == "__main__":
    main()
