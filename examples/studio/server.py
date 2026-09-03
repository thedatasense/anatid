"""anatid studio: a local web UI that runs examples/glm_openrouter_agent.py one step at a time.

The scenario is the one in the sibling example. A team briefs an assistant on day one, the
assistant answers a question that needs a two-hop walk of the entity graph, a handover
supersedes one fact, the model proposes a write that a person approves or declines, and then
as_of and provenance replay history without any model at all.

Run it:

    pip install -r examples/studio/requirements.txt
    export OPEN_ROUTER_KEY=sk-or-...      # optional; only the "ask" steps need it
    python examples/studio/server.py      # serves http://127.0.0.1:8765

The database lives beside this file as studio.anatid. POST /api/reset deletes it and rebuilds
the day-one briefing. Every endpoint except /api/ask and /api/approve works without a key.

The server binds 127.0.0.1 only, never logs the key, and never logs memory content.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any

try:
    import anatid
    from anatid import Anatid, NotFoundError, utcnow
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
except ImportError as exc:  # a plain, actionable message instead of a traceback
    import sys

    sys.exit(
        f"anatid studio needs its dependencies ({exc.name} is missing).\n"
        "From the repository root, run:\n"
        "    uv venv .venv-studio && uv pip install --python .venv-studio/bin/python -r examples/studio/requirements.txt\n"
        "    .venv-studio/bin/python examples/studio/server.py\n"
        "or, with plain pip:\n"
        "    python -m venv .venv-studio && .venv-studio/bin/pip install -r examples/studio/requirements.txt"
    )

HERE = pathlib.Path(__file__).resolve().parent
DB_PATH = HERE / "studio.anatid"
INDEX_HTML = HERE / "index.html"
MODEL = "z-ai/glm-5.3-flash"
TENANT = 1
EMBEDDING_DIM = 64
MAX_TOOL_ROUNDS = 6
WRITER_MODEL = "glm-5.3-flash"

# The same briefing as the example. Edges give the graph its shape; facts hang off entities.
BRIEFING_EDGES = [
    ("Ada", "Project Kestrel", "leads"),
    ("Project Kestrel", "ingest-service", "owns"),
    ("Bo", "ingest-service", "maintains"),
    ("Project Kestrel", "postgres-primary", "depends_on"),
]
BRIEFING_FACTS = [
    ("Ada leads Project Kestrel.", ["Ada", "Project Kestrel"]),
    ("Bo maintains the ingest-service and is the person to page for it.", ["Bo", "ingest-service"]),
    ("The ingest-service is owned by Project Kestrel.", ["ingest-service", "Project Kestrel"]),
    ("postgres-primary has a nightly vacuum window at 02:00 UTC.", ["postgres-primary"]),
    ("Project Kestrel ships on Fridays.", ["Project Kestrel"]),
]
BRIEFING_EPISODE = "Team onboarding doc, 2026-03-01"
BRIEFING_WRITER = "onboarding"

HANDOVER_CONTENT = "Cy maintains the ingest-service. Bo moved to Project Harrier on 2026-04-15."
HANDOVER_ENTITIES = ["Cy", "ingest-service", "Bo"]
HANDOVER_EPISODE = "Handover notes, 2026-04-15: Bo -> Harrier, Cy takes ingest-service."
HANDOVER_WRITER = "handover-notes"

SYSTEM_PROMPT = (
    "You are an engineering-team assistant with a graph memory. "
    "Always call recall before answering a question about the team, "
    "passing seed_entity when the question names a person or project. "
    "Answer only from what recall returns. Be brief."
)

NO_KEY_MESSAGE = (
    "No OpenRouter key is configured, so the model steps are unavailable. "
    "Set OPEN_ROUTER_KEY or add open_router_key= to a .env file in the repository root, "
    "then restart the server. Steps 1, 3, 5 and 6 work without a model."
)

# The tools the model can call. Reads run immediately; the write waits for a person.
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


# --------------------------------------------------------------------------------------
# Key handling. Same lookup order as the example. The key is held in memory only.
# --------------------------------------------------------------------------------------


def load_key() -> str | None:
    if os.environ.get("STUDIO_NO_MODEL"):
        # Run without a model on purpose, for example to demonstrate the no-key path.
        return None
    for var in ("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    for parent in (HERE, *HERE.parents):
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                key, _, value = line.partition("=")
                if key.strip().lower() in ("open_router_key", "openrouter_api_key"):
                    value = value.strip().strip("\"'")
                    if value:
                        return value
    return None


# --------------------------------------------------------------------------------------
# Process state. One database handle, one conversation, and the writes waiting on a person.
# --------------------------------------------------------------------------------------


class State:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.db: Anatid | None = None
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # pending id -> what the loop needs to carry on after the person decides
        self.pending: dict[str, dict] = {}
        self.api_key: str | None = None
        self.client: Any = None


STATE = State()


def get_db() -> Anatid:
    if STATE.db is None:
        raise HTTPException(503, "The database is not open. Call POST /api/reset.")
    return STATE.db


def delete_db_files() -> None:
    for suffix in ("", ".wal", ".shadow", ".tmp"):
        pathlib.Path(str(DB_PATH) + suffix).unlink(missing_ok=True)


def open_db() -> Anatid:
    return Anatid.open(DB_PATH, tenant=TENANT, embedding_dim=EMBEDDING_DIM)


def brief(db: Anatid) -> None:
    """Write the day-one briefing exactly as the example does."""
    for src, dst, kind in BRIEFING_EDGES:
        db.relate(src, dst, rel_kind=kind)
    for content, entities in BRIEFING_FACTS:
        db.remember(content, entities=entities, writer=BRIEFING_WRITER, episode=BRIEFING_EPISODE)
    db.rebuild_fts_index()


def reset_db() -> None:
    with STATE.lock:
        if STATE.db is not None:
            STATE.db.close()
            STATE.db = None
        delete_db_files()
        db = open_db()
        brief(db)
        STATE.db = db
        STATE.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        STATE.pending.clear()


# --------------------------------------------------------------------------------------
# Serialisation. Ids are 64-bit and larger than JavaScript's safe integer, so they travel as
# strings. Timestamps are naive UTC in anatid; they are sent as ISO-8601 with a Z suffix.
# --------------------------------------------------------------------------------------


def sid(value: int | None) -> str | None:
    return None if value is None else str(value)


def pid(value: str | int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(400, "Ids are decimal strings.")


def ts(value: _dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return value.isoformat(timespec="microseconds") + "Z"


def parse_ts(value: str) -> _dt.datetime:
    try:
        parsed = _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, "t must be an ISO-8601 timestamp, for example 2026-03-01T12:00:00Z.")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return parsed


def memory_dict(m: anatid.Memory, about: list[str] | None = None) -> dict:
    return {
        "id": sid(m.memory_id),
        "content": m.content,
        "kind": m.kind,
        "writer": m.writer,
        "episode_id": sid(m.episode_id),
        "created_at": ts(m.created_at),
        "valid_from": ts(m.valid_from),
        "valid_to": ts(m.valid_to),
        "tx_from": ts(m.tx_from),
        "tx_to": ts(m.tx_to),
        "is_current": m.is_current,
        "about": about if about is not None else [],
    }


def episode_dict(e: anatid.Episode) -> dict:
    return {
        "id": sid(e.episode_id),
        "content": e.content,
        "source": e.source,
        "writer": e.writer,
        "created_at": ts(e.created_at),
    }


# --------------------------------------------------------------------------------------
# Read-only SQL helpers over db.connection. Everything is scoped to the studio tenant.
# --------------------------------------------------------------------------------------


def about_names(db: Anatid) -> dict[int, list[str]]:
    rows = db.connection.execute(
        "SELECT a.src, e.name FROM edges_about a JOIN entities e ON e.entity_id = a.dst "
        "WHERE a.tenant_id = ? ORDER BY a.edge_id",
        [TENANT],
    ).fetchall()
    out: dict[int, list[str]] = {}
    for memory_id, name in rows:
        out.setdefault(memory_id, []).append(name)
    return out


def scalar_ts(db: Anatid, sql: str, params: list) -> _dt.datetime | None:
    row = db.connection.execute(sql, params).fetchone()
    return row[0] if row else None


def day_one_ts(db: Anatid) -> _dt.datetime | None:
    """Day one is the instant just after the briefing landed, derived from the rows themselves."""
    latest = scalar_ts(
        db,
        "SELECT max(created_at) FROM memories WHERE tenant_id = ? AND writer = ?",
        [TENANT, BRIEFING_WRITER],
    )
    return None if latest is None else latest + _dt.timedelta(milliseconds=1)


def handover_ts(db: Anatid) -> _dt.datetime | None:
    first = scalar_ts(
        db,
        "SELECT min(created_at) FROM memories WHERE tenant_id = ? AND writer = ?",
        [TENANT, HANDOVER_WRITER],
    )
    return None if first is None else first + _dt.timedelta(milliseconds=1)


def entity_row(db: Anatid, name: str) -> tuple[int, str] | None:
    return db.connection.execute(
        "SELECT entity_id, name FROM entities WHERE tenant_id = ? "
        "AND entity_key = trim(regexp_replace(lower(?), '\\s+', ' ', 'g')) LIMIT 1",
        [TENANT, name],
    ).fetchone()


def frontier(db: Anatid, seed_name: str, hops: int = 2) -> dict | None:
    """Breadth-first walk over currently valid RELATES_TO edges, undirected, like recall_2hop.

    Returns the entities reached at each hop and the edges used, so the UI can draw the path.
    """
    seed = entity_row(db, seed_name)
    if seed is None:
        return None
    con = db.connection
    levels: list[list[dict]] = [[{"id": sid(seed[0]), "name": seed[1]}]]
    seen = {seed[0]}
    edges: list[dict] = []
    current = [seed[0]]
    for hop in range(1, hops + 1):
        marks = ",".join("?" for _ in current)
        rows = con.execute(
            "SELECT r.edge_id, r.src, r.dst, r.rel_kind, s.name, d.name FROM edges_relates r "
            "JOIN entities s ON s.entity_id = r.src JOIN entities d ON d.entity_id = r.dst "
            f"WHERE r.tenant_id = ? AND r.valid_to IS NULL AND r.tx_to IS NULL "
            f"AND (r.src IN ({marks}) OR r.dst IN ({marks})) ORDER BY r.edge_id",
            [TENANT, *current, *current],
        ).fetchall()
        next_level: list[dict] = []
        next_ids: list[int] = []
        for edge_id, src, dst, kind, src_name, dst_name in rows:
            other_id, other_name = (dst, dst_name) if src in seen and dst not in seen else (src, src_name)
            if src in seen and dst in seen:
                continue
            edges.append({"id": sid(edge_id), "src": sid(src), "dst": sid(dst), "kind": kind, "hop": hop})
            if other_id not in seen:
                seen.add(other_id)
                next_level.append({"id": sid(other_id), "name": other_name})
                next_ids.append(other_id)
        levels.append(next_level)
        current = next_ids
        if not current:
            break
    return {"seed": {"id": sid(seed[0]), "name": seed[1]}, "hops": levels, "edges": edges}


def snapshot(db: Anatid) -> dict:
    con = db.connection
    entities = [
        {"id": sid(r[0]), "name": r[1], "kind": r[2], "valid_to": ts(r[3])}
        for r in con.execute(
            "SELECT entity_id, name, kind, valid_to FROM entities WHERE tenant_id = ? ORDER BY entity_id",
            [TENANT],
        ).fetchall()
    ]
    relates = [
        {"id": sid(r[0]), "src": sid(r[1]), "dst": sid(r[2]), "kind": r[3],
         "valid_from": ts(r[4]), "valid_to": ts(r[5]), "writer": r[6]}
        for r in con.execute(
            "SELECT edge_id, src, dst, rel_kind, valid_from, valid_to, writer FROM edges_relates "
            "WHERE tenant_id = ? ORDER BY edge_id",
            [TENANT],
        ).fetchall()
    ]
    supersedes = [
        {"new_id": sid(r[0]), "old_id": sid(r[1]), "writer": r[2], "tx_from": ts(r[3])}
        for r in con.execute(
            "SELECT src, dst, writer, tx_from FROM edges_supersedes WHERE tenant_id = ? ORDER BY edge_id",
            [TENANT],
        ).fetchall()
    ]
    names = about_names(db)
    memories = []
    for r in con.execute(
        "SELECT memory_id FROM memories WHERE tenant_id = ? ORDER BY created_at, memory_id", [TENANT]
    ).fetchall():
        m = db.get(r[0], with_embedding=False)
        if m is not None:
            memories.append(memory_dict(m, names.get(m.memory_id, [])))
    stats = db.stats()
    try:
        fts = db.fts_status()
        fts_info = {"available": fts.available, "stale": fts.stale, "indexed_rows": fts.indexed_rows}
    except Exception:
        fts_info = {"available": None, "stale": None, "indexed_rows": None}
    return {
        "entities": entities,
        "relates": relates,
        "supersedes": supersedes,
        "memories": memories,
        "stats": {
            "memories": stats.get("memories"),
            "current_memories": stats.get("current_memories"),
            "entities": stats.get("entities"),
            "episodes": stats.get("episodes"),
            "edges_relates": stats.get("edges_relates"),
            "edges_about": stats.get("edges_about"),
            "edges_supersedes": stats.get("edges_supersedes"),
            "expand_path": stats.get("expand_path"),
            "anatid_version": anatid.__version__,
            "schema_version": anatid.SCHEMA_VERSION,
            "embedding_dim": EMBEDDING_DIM,
            "tenant": TENANT,
            "fts": fts_info,
            "arms": [
                {"name": "text", "active": True, "reason": "BM25 over the fts index"},
                {"name": "graph", "active": True, "reason": "2-hop walk over RELATES_TO edges"},
                {"name": "vector", "active": False, "reason": "no embeddings in this demo"},
            ],
        },
        "day_one_ts": ts(day_one_ts(db)),
        "handover_ts": ts(handover_ts(db)),
        "now_ts": ts(utcnow()),
        "has_model_key": STATE.client is not None,
        "model": MODEL,
        "pending_write": next(iter(pending_public(p) for p in STATE.pending.values()), None),
        "db_path": str(DB_PATH),
        "db_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else 0,
    }


# --------------------------------------------------------------------------------------
# The model loop. Same shape as the example, but the write tool parks instead of prompting.
# --------------------------------------------------------------------------------------


def reasoning_text(message: Any) -> str:
    """Readable text from reasoning_details, if the provider sent any."""
    parts: list[str] = []
    details = getattr(message, "reasoning_details", None)
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                text = item.get("text") or item.get("summary")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    if not parts:
        plain = getattr(message, "reasoning", None)
        if isinstance(plain, str) and plain.strip():
            parts.append(plain.strip())
    return "\n\n".join(parts)


def run_recall_tool(db: Anatid, args: dict) -> str:
    hits = db.recall(args["query"], seed_entity=args.get("seed_entity"), k=5, on_stale_fts="ignore")
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


def pending_public(p: dict) -> dict:
    return {"id": p["id"], "content": p["args"].get("content", ""), "entities": list(p["args"].get("entities", []))}


def call_model(history: list[dict]) -> Any:
    try:
        response = STATE.client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            extra_body={"reasoning": {"enabled": True}},
        )
    except Exception as exc:  # the key is never part of the message we return
        status = getattr(exc, "status_code", None)
        detail = f"The model call failed ({type(exc).__name__}"
        if status:
            detail += f", HTTP {status}"
        detail += "). Check the OpenRouter key and network, then try again."
        raise HTTPException(502, detail)
    return response.choices[0].message


def process_calls(db: Anatid, steps: list[dict], calls: list[dict], rounds_left: int) -> dict | None:
    """Run the tool calls of one assistant message in order.

    Reads run now. The first write is parked as a pending approval and the function returns
    it; the loop resumes from /api/approve with the calls that were still queued.
    """
    history = STATE.history
    while calls:
        call = calls.pop(0)
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        steps.append({"type": "tool_call", "id": call["id"], "name": name, "arguments": args})

        if name in WRITE_TOOLS:
            pending_id = uuid.uuid4().hex
            record = {
                "id": pending_id,
                "name": name,
                "args": args,
                "tool_call_id": call["id"],
                "remaining_calls": calls,
                "rounds_left": rounds_left,
            }
            STATE.pending[pending_id] = record
            return record

        if name == "recall":
            with STATE.lock:
                result = run_recall_tool(db, args)
        else:
            result = f"Unknown tool {name}."
        history.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        steps.append({"type": "tool_result", "tool_call_id": call["id"], "name": name, "content": result})
    return None


def run_loop(db: Anatid, steps: list[dict], rounds_left: int) -> dict:
    """Call the model until it answers or a write needs a person. Returns the API payload."""
    history = STATE.history
    while rounds_left > 0:
        message = call_model(history)
        entry: dict = {"role": "assistant", "content": message.content}
        # reasoning_details goes back exactly as received so the model keeps its chain of thought
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

        thought = reasoning_text(message)
        if thought:
            steps.append({"type": "reasoning", "text": thought})

        if not message.tool_calls:
            answer = message.content or ""
            steps.append({"type": "answer", "text": answer})
            return {"steps": steps, "answer": answer, "pending_write": None}

        rounds_left -= 1
        parked = process_calls(db, steps, list(entry["tool_calls"]), rounds_left)
        if parked is not None:
            return {"steps": steps, "answer": None, "pending_write": pending_public(parked)}

    answer = "(stopped after six tool rounds)"
    steps.append({"type": "answer", "text": answer})
    return {"steps": steps, "answer": answer, "pending_write": None}


# --------------------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.api_key = load_key()
    if STATE.api_key:
        from openai import OpenAI

        STATE.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=STATE.api_key)
    if DB_PATH.exists():
        try:
            STATE.db = open_db()
        except Exception:
            reset_db()
    else:
        reset_db()
    try:
        yield
    finally:
        with STATE.lock:
            if STATE.db is not None:
                STATE.db.close()
                STATE.db = None


app = FastAPI(title="anatid studio", lifespan=lifespan, docs_url=None, redoc_url=None)


class AskBody(BaseModel):
    question: str


class ApproveBody(BaseModel):
    id: str
    approved: bool


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "anatid_version": anatid.__version__,
        "schema_version": anatid.SCHEMA_VERSION,
        "has_model_key": STATE.client is not None,
        "model": MODEL,
        "db_open": STATE.db is not None,
    }


@app.post("/api/reset")
def reset() -> dict:
    reset_db()
    with STATE.lock:
        return snapshot(get_db())


@app.get("/api/state")
def state() -> dict:
    with STATE.lock:
        return snapshot(get_db())


@app.post("/api/ask")
def ask(body: AskBody) -> dict:
    if STATE.client is None:
        raise HTTPException(409, NO_KEY_MESSAGE)
    if STATE.pending:
        raise HTTPException(409, "A write is waiting for a decision. Approve or decline it first.")
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "The question is empty.")
    db = get_db()
    mark = len(STATE.history)
    STATE.history.append({"role": "user", "content": question})
    steps: list[dict] = [{"type": "user", "text": question}]
    try:
        return run_loop(db, steps, MAX_TOOL_ROUNDS)
    except HTTPException:
        # Roll the conversation back so a failed call leaves no half turn behind.
        del STATE.history[mark:]
        raise


@app.post("/api/approve")
def approve(body: ApproveBody) -> dict:
    record = STATE.pending.pop(body.id, None)
    if record is None:
        raise HTTPException(404, "No write with that id is waiting.")
    db = get_db()
    steps: list[dict] = []
    stored_id: int | None = None
    if body.approved:
        args = record["args"]
        with STATE.lock:
            m = db.remember(args["content"], entities=list(args.get("entities", [])), writer=WRITER_MODEL)
            db.rebuild_fts_index()
        stored_id = m.memory_id
        result = f"Stored as memory {m.memory_id}."
    else:
        result = "The person declined this write. The memory was not changed."
    STATE.history.append({"role": "tool", "tool_call_id": record["tool_call_id"], "content": result})
    steps.append(
        {
            "type": "tool_result",
            "tool_call_id": record["tool_call_id"],
            "name": record["name"],
            "content": result,
            "approved": body.approved,
        }
    )
    # Any tool calls queued behind the parked one in the same assistant message run now.
    parked = process_calls(db, steps, record["remaining_calls"], record["rounds_left"])
    if parked is not None:
        payload = {"steps": steps, "answer": None, "pending_write": pending_public(parked)}
    else:
        payload = run_loop(db, steps, record["rounds_left"])
    payload["stored_memory_id"] = sid(stored_id)
    with STATE.lock:
        payload["state"] = snapshot(db)
    return payload


@app.post("/api/supersede")
def supersede() -> dict:
    db = get_db()
    with STATE.lock:
        bo_fact = next(
            (m for m in db.recall_2hop("ingest-service", limit=50)
             if m.content.startswith("Bo maintains") and m.is_current),
            None,
        )
        if bo_fact is None:
            raise HTTPException(409, "The handover has already been applied. Reset to run it again.")
        replacement = db.supersede(
            bo_fact.memory_id,
            HANDOVER_CONTENT,
            entities=HANDOVER_ENTITIES,
            writer=HANDOVER_WRITER,
            episode=HANDOVER_EPISODE,
        )
        db.relate("Cy", "ingest-service", rel_kind="maintains")
        db.rebuild_fts_index()
        old = db.get(bo_fact.memory_id, with_embedding=False)
        return {
            "old_id": sid(bo_fact.memory_id),
            "new_id": sid(replacement.memory_id),
            "old_is_current": old.is_current if old else None,
            "old_valid_to": ts(old.valid_to) if old else None,
            "state": snapshot(db),
        }


@app.get("/api/recall")
def recall(seed: str = "", q: str = "", k: int = 10) -> dict:
    seed = seed.strip()
    q = q.strip()
    if not seed and not q:
        raise HTTPException(400, "Pass q (words for the text arm), seed (an entity for the graph arm), or both.")
    db = get_db()
    with STATE.lock:
        try:
            hits = db.recall(q or None, seed_entity=seed or None, k=k, on_stale_fts="ignore")
            walk = frontier(db, seed) if seed else None
            reached = [sid(m) for m, _ in db.recall_2hop_ids(seed, limit=50)] if seed else []
        except NotFoundError:
            raise HTTPException(404, f"No entity named {seed!r}.")
        if seed and walk is None:
            raise HTTPException(404, f"No entity named {seed!r}.")
        return {
            "query": q or None,
            "seed": seed or None,
            "arms": list(hits.arms),
            "bm25_stale": hits.bm25_stale,
            "hits": [
                {
                    **memory_dict(h.memory, list(h.about)),
                    "rank": h.rank,
                    "score": h.score,
                    "text_rank": h.text_rank,
                    "graph_rank": h.graph_rank,
                    "vector_rank": h.vector_rank,
                    "found_by": [
                        arm
                        for arm, rank in (("text", h.text_rank), ("graph", h.graph_rank), ("vector", h.vector_rank))
                        if rank is not None
                    ],
                }
                for h in hits
            ],
            "frontier": walk,
            "graph_memories": reached,
        }


def maintainer_line(memories: list[anatid.Memory]) -> str:
    for m in memories:
        if "maintains" in m.content:
            return m.content
    return "(nothing on record)"


@app.get("/api/asof")
def asof(t: str) -> dict:
    when = parse_ts(t)
    db = get_db()
    with STATE.lock:
        names = about_names(db)
        view = db.as_of(when)
        out: dict[str, Any] = {"t": ts(when)}
        for key, seed in (("ingest_service", "ingest-service"), ("ada", "Ada")):
            try:
                found = view.recall_2hop(seed, limit=20)
            except NotFoundError:
                found = []
            out[key] = {
                "seed": seed,
                "memories": [memory_dict(m, names.get(m.memory_id, [])) for m in found],
                "maintainer": maintainer_line(found),
            }
        return out


@app.get("/api/provenance/{memory_id}")
def provenance(memory_id: str) -> dict:
    db = get_db()
    with STATE.lock:
        try:
            p = db.provenance(pid(memory_id))
        except NotFoundError:
            raise HTTPException(404, "No memory with that id.")
        names = about_names(db)
        episodes = {e.episode_id: episode_dict(e) for e in p.episodes}
        chain = []
        for m in p.chain:
            entry = memory_dict(m, names.get(m.memory_id, []))
            entry["episode"] = episodes.get(m.episode_id)
            chain.append(entry)
        return {
            "memory_id": sid(p.memory_id),
            "chain": chain,
            "episodes": list(episodes.values()),
            "writers": list(p.writers),
            "edges": [
                {"id": sid(e.edge_id), "new_id": sid(e.src), "old_id": sid(e.dst), "writer": e.writer, "tx_from": ts(e.tx_from)}
                for e in p.edges
            ],
        }


@app.exception_handler(HTTPException)
async def plain_errors(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("STUDIO_PORT", "8765"))
    # access_log=False keeps query strings, which can carry recall text, out of the terminal
    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)
