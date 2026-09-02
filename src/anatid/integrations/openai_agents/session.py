"""``AnatidSession`` -- the OpenAI Agents SDK ``Session`` protocol backed by anatid/DuckDB.

Why this exists
---------------
The SDK ships SQLite, SQLAlchemy and Redis sessions.  All three store conversation history in a
*different* store from the agent's knowledge, so "what did we say" and "what do I know" can never
be answered by one query.  ``AnatidSession`` writes turns into a DuckDB table **in the same file
as the memory graph**, with anatid's system columns on it, which buys two things nothing else
in the SDK ecosystem has:

1. **History and knowledge are joinable.**  ``session.entities_mentioned()`` is one SQL statement
   joining ``agent_messages`` to ``entities``; ``session.memories_written_here()`` finds the
   memories this conversation produced.  See :meth:`entities_mentioned`.
2. **History is queryable, not an opaque blob.**  Every row carries ``item_type``, ``role``,
   ``tool_name``, the extracted text and the raw item JSON, plus ``valid_from/valid_to/tx_from/
   tx_to/writer``.  DuckDB aggregates then give you the analytics the SDK's
   ``AdvancedSQLiteSession`` offers -- turn counts per day, token rollups, tool-call counts --
   without a second database.

Protocol conformance
--------------------
Matched against the installed SDK (``openai-agents`` 0.22.0,
``agents/memory/session.py``): the protocol is **async**, items are ``TResponseInputItem``
*dicts*, and the four methods are ``get_items(limit=None)``, ``add_items(items)``,
``pop_item()``, ``clear_session()``.  ``session_id: str`` and ``session_settings`` are attributes.
No inheritance is required (``Session`` is a ``runtime_checkable`` ``Protocol``), so this class
subclasses nothing from the SDK and imports it only for :class:`SessionSettings`, which is
optional -- ``AnatidSession`` works with the SDK absent.

``limit`` semantics follow the documented protocol: ``None`` -> every item, ``N > 0`` -> the last
``N`` in chronological order, ``N <= 0`` -> ``[]``.  (``SQLiteSession`` additionally inherits
SQLite's "``LIMIT -1`` means unlimited" quirk; DuckDB has no such quirk and neither do we.)

Ordering contract
-----------------
Rows are ordered by ``(seq, message_id)``.  ``seq`` is allocated inside the insert transaction as
``max(seq) + 1`` for the session.  Under DuckDB's optimistic MVCC two *concurrent* transactions
appending to the same ``session_id`` can therefore be handed the same ``seq`` -- appends never
conflict, so neither aborts.  ``message_id`` (``anatid.ids.new_id()``, time-ordered) breaks the
tie, so the order is always total and deterministic; it is just not guaranteed to interleave the
two racing writers the way wall-clock did.  One conversation is written by one runner at a time,
so this is a documented corner, not the normal path.

Erasure
-------
Putting the transcript in the same file as the memory graph is the point of this class, and it
is also a right-to-erasure obligation: a tool result row quotes the memory's id *and* its
content verbatim, and ``get_items()`` would replay it into the model's context after the memory
itself had been purged.  The constructor therefore registers an erasure hook on the handle
(:meth:`anatid.Anatid.register_erasure_hook`), so ``db.forget(mid, hard=True)`` deletes the
matching ``agent_messages`` rows in the same transaction and reports them as the receipt's
``extra_rows_deleted``.  A *soft* forget deliberately leaves the transcript alone -- it is not
an erasure.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
from typing import Any, Iterable, Mapping, Sequence

from ...database import Anatid
from ...ids import new_id
from ...schema import quote_ident
from ...types import Memory, Namespace, utcnow

log = logging.getLogger("anatid.integrations.openai_agents")

__all__ = ["AnatidSession", "MESSAGE_COLUMNS", "classify_item"]

try:  # pragma: no cover - exercised only by the installed-SDK path
    from agents.memory.session_settings import (  # type: ignore[import-not-found]
        SessionSettings,
        resolve_session_limit as _resolve_session_limit,
    )
except Exception:  # pragma: no cover - the SDK is an optional dependency
    SessionSettings = None  # type: ignore[assignment]

    def _resolve_session_limit(explicit_limit, settings):  # type: ignore[misc]
        if explicit_limit is not None:
            return explicit_limit
        return getattr(settings, "limit", None) if settings is not None else None


#: ``agent_messages`` columns beyond ``message_id``/``tenant_id`` and the system columns.
MESSAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("session_id", "VARCHAR"),
    ("seq", "BIGINT"),
    ("turn", "BIGINT"),
    ("item_type", "VARCHAR"),
    ("role", "VARCHAR"),
    ("tool_name", "VARCHAR"),
    ("item_text", "VARCHAR"),
    ("item_json", "VARCHAR"),
    ("created_at", "TIMESTAMP"),
)

_SESSION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("session_id", "VARCHAR"),
    ("created_at", "TIMESTAMP"),
    ("updated_at", "TIMESTAMP"),
    ("turns", "BIGINT"),
    ("items", "BIGINT"),
)

_USAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("session_id", "VARCHAR"),
    ("turn", "BIGINT"),
    ("requests", "BIGINT"),
    ("input_tokens", "BIGINT"),
    ("output_tokens", "BIGINT"),
    ("total_tokens", "BIGINT"),
    ("cached_tokens", "BIGINT"),
    ("reasoning_tokens", "BIGINT"),
    ("created_at", "TIMESTAMP"),
)

#: anatid's current-state predicate, one conjunct more than "``valid_to IS NULL``" because that
#: is what the bitemporal model actually means (see :func:`anatid.schema.temporal_predicate`).
_CURRENT = "valid_to IS NULL AND tx_to IS NULL"


def _json_default(value: Any) -> Any:
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return str(value)


def _dumps(item: Any) -> str:
    try:
        return json.dumps(item, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(item, ensure_ascii=False, default=_json_default)


def _as_mapping(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # pragma: no cover - defensive
            return {}
    return {}


def _text_of(value: Any) -> str:
    """Best-effort plain text for a message ``content`` field (str, or list of parts)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "input_text", "output_text", "content", "refusal"):
            got = value.get(key)
            if isinstance(got, str):
                return got
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(part for part in (_text_of(v) for v in value) if part)
    return ""


def classify_item(item: Any) -> tuple[str, str | None, str | None, str]:
    """Return ``(item_type, role, tool_name, text)`` for one ``TResponseInputItem``.

    Purely descriptive: the raw item JSON is always stored verbatim, so a classification this
    function gets wrong costs a nicer ``GROUP BY``, never a round trip.
    """
    data = _as_mapping(item)
    item_type = data.get("type")
    role = data.get("role")
    if not isinstance(role, str):
        role = None
    if not isinstance(item_type, str):
        item_type = "message" if role else "unknown"

    tool_name = data.get("name")
    if not isinstance(tool_name, str) or "call" not in item_type:
        tool_name = None

    text = _text_of(data.get("content"))
    if not text:
        for key in ("output", "arguments", "text", "input"):
            text = _text_of(data.get(key))
            if text:
                break
    return item_type, role, tool_name, text


def _is_user_message(item_type: str, role: str | None) -> bool:
    return role == "user" and item_type in ("message", "unknown")


class AnatidSession:
    """A ``Session`` for the OpenAI Agents SDK that stores turns in an anatid database.

    ``AnatidSession("conv-1", db)`` shares an already-open :class:`~anatid.Anatid` handle -- the
    normal case, because the point is that history and memory live in one file.  ``AnatidSession
    ("conv-1", path="agent.anatid", tenant=7)`` opens (and owns) one itself.  With neither, an
    in-memory database is created, which is fine for tests and lost at exit.

    Tenancy: rows are stamped with the handle's ``tenant_id`` and every statement filters on it.
    That is *scoping*, not isolation -- DuckDB has no row-level security, and raw SQL through
    ``session.db.connection`` sees every tenant in the file.  For real isolation give each tenant
    its own file with :class:`anatid.DatabasePool`.

    ``clear_mode``
        ``"delete"`` (default) makes :meth:`clear_session` and :meth:`pop_item` remove rows, which
        is exactly what ``SQLiteSession`` does.  ``"close"`` instead stamps ``valid_to``/``tx_to``
        so the turn stops being visible to the agent but stays in the table, where your own
        ``valid_from``/``valid_to`` filter can still read it.  (``db.as_of(t)`` is a *memory* verb
        and does not cover these tables.)  A real erasure request needs ``"delete"``.
    """

    #: Declared for the SDK protocol; ``None`` means "no default limit".
    session_settings: Any = None

    def __init__(
        self,
        session_id: str,
        db: Anatid | None = None,
        *,
        path: str | None = None,
        tenant: int | Namespace | None = None,
        writer: str | None = None,
        messages_table: str = "agent_messages",
        sessions_table: str = "agent_sessions",
        usage_table: str = "agent_turn_usage",
        clear_mode: str = "delete",
        session_settings: Any = None,
        **open_kwargs: Any,
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if clear_mode not in ("delete", "close"):
            raise ValueError('clear_mode must be "delete" or "close"')
        if db is not None and path is not None:
            raise ValueError("pass either db= or path=, not both")

        self.session_id = session_id
        self._owns_db = db is None
        if db is None:
            self.db = Anatid.open(path or ":memory:", tenant=0 if tenant is None else tenant,
                                  **open_kwargs)
        else:
            if open_kwargs:
                raise ValueError("open_kwargs are only accepted together with path=")
            self.db = db
        self.namespace = self.db.resolve_tenant(tenant)
        self.tenant_id = self.namespace.tenant_id
        self.writer = writer or f"session:{session_id}"
        self.clear_mode = clear_mode
        self.session_settings = session_settings

        self._messages = messages_table
        self._sessions = sessions_table
        self._usage = usage_table
        self._m = quote_ident(messages_table)
        self._s = quote_ident(sessions_table)
        self._u = quote_ident(usage_table)
        self._ensure_tables()
        self.db.register_erasure_hook(self._purge_memory_from_transcript)

    # ------------------------------------------------------------------ schema

    def _ensure_tables(self) -> None:
        """Create the three tables if absent.  Idempotent; safe to call from many sessions."""
        self.db.create_node_label(self._messages, MESSAGE_COLUMNS, id_column="message_id")
        self.db.create_node_label(self._sessions, _SESSION_COLUMNS, id_column="agent_session_id")
        self.db.create_node_label(self._usage, _USAGE_COLUMNS, id_column="turn_usage_id")

    def __repr__(self) -> str:
        return (f"<AnatidSession {self.session_id!r} path={self.db.path!r} "
                f"tenant={self.tenant_id} clear_mode={self.clear_mode}>")

    def close(self) -> None:
        """Close the database **only if this session opened it**.  Idempotent."""
        if self._owns_db:
            self.db.close()

    def __enter__(self) -> "AnatidSession":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ erasure

    def _purge_memory_from_transcript(self, db: Anatid, memory_id: int, tenant_id: int,
                                      content: str | None = None) -> int:
        """Erasure hook: drop transcript rows that quote a hard-purged memory.

        Registered on the handle in :meth:`__init__`, so ``db.forget(mid, hard=True)`` removes
        them **inside its own transaction**.  Without it, ``forget(hard=True)`` deleted the
        memory row while the tool call that created it and the tool result that echoed its
        ``content`` verbatim stayed in ``agent_messages`` in the same file -- and
        :meth:`get_items` handed them straight back to the model on the next turn.  A right to
        erasure that a conversation replay defeats is not one.

        A row goes if its stored JSON or extracted text contains the memory's decimal **id** or
        its **content** verbatim (plain substring, via DuckDB ``contains`` -- no LIKE wildcards,
        so punctuation in the content is literal).  Both are needed: the tool *call* carries the
        content without the id, the tool *result* carries the id.

        Two honest consequences, neither hidden:

        * It is scoped to this session's messages table and the memory's tenant, but **not** to
          this ``session_id`` -- another conversation that quoted the same text is erased too,
          which is the point.
        * A very short memory ("hi") will match unrelated rows.  Erasure is destructive by
          definition, and anatid resolves the tie towards erasing: a false positive costs one
          transcript row, a false negative costs the erasure.
        """
        needles = [str(int(memory_id))]
        if content:
            needles.append(str(content))
        where = " OR ".join(
            "contains(coalesce(item_json, ''), ?) OR contains(coalesce(item_text, ''), ?)"
            for _ in needles)
        params: list[Any] = [int(tenant_id)]
        for n in needles:
            params += [n, n]
        cur = db.execute(f"DELETE FROM {self._m} WHERE tenant_id = ? AND ({where})", params)
        try:
            row = cur.fetchone()
        except Exception:                          # pragma: no cover - driver dependent
            return 0
        return 0 if row is None or row[0] is None else int(row[0])

    # ------------------------------------------------------------------ helpers

    @property
    def _visible(self) -> str:
        """Extra predicate for rows a soft clear/pop has closed."""
        return f" AND {_CURRENT}" if self.clear_mode == "close" else ""

    def sql(self, query: str, params: Sequence[Any] | None = None) -> list[tuple]:
        """Run one read against the session's database and return the rows.

        The escape hatch for anything the helpers below do not cover.  Raw SQL bypasses the
        tenant filter -- ``tenant_id`` is a column here, not a security boundary.
        """
        return self.db.execute(query, params).fetchall()

    def _decode(self, rows: Iterable[tuple]) -> list[Any]:
        items: list[Any] = []
        for (payload,) in rows:
            try:
                items.append(json.loads(payload))
            except (TypeError, ValueError):
                log.warning("session %r: skipping unreadable item_json", self.session_id)
                continue
        return items

    # ------------------------------------------------------------------ Session protocol

    async def get_items(self, limit: int | None = None) -> list[Any]:
        """The conversation history, oldest first.

        ``limit=None`` returns everything, ``limit=N>0`` the newest ``N`` items still in
        chronological order, ``limit<=0`` an empty list.
        """
        session_limit = _resolve_session_limit(limit, self.session_settings)
        return await asyncio.to_thread(self._get_items_sync, session_limit)

    def _get_items_sync(self, session_limit: int | None) -> list[Any]:
        base = (f"SELECT item_json FROM {self._m} "
                f"WHERE tenant_id = ? AND session_id = ?{self._visible}")
        if session_limit is None:
            rows = self.db.execute(f"{base} ORDER BY seq, message_id",
                                   [self.tenant_id, self.session_id]).fetchall()
            return self._decode(rows)
        if session_limit <= 0:
            return []
        # Widen the window when unreadable rows sit among the newest, so `limit` counts items
        # the caller can actually use -- the behaviour SQLiteSession/SQLAlchemySession settled on.
        window = session_limit
        while True:
            rows = self.db.execute(
                f"{base} ORDER BY seq DESC, message_id DESC LIMIT ?",
                [self.tenant_id, self.session_id, window]).fetchall()
            items = self._decode(reversed(rows))
            if len(items) >= session_limit:
                return items[-session_limit:]
            if len(rows) < window:
                return items
            window *= 2

    async def add_items(self, items: list[Any]) -> None:
        """Append items to the history.  One transaction for the whole batch."""
        if not items:
            return
        await asyncio.to_thread(self._add_items_sync, list(items))

    def _add_items_sync(self, items: list[Any]) -> None:
        now = utcnow()
        with self.db.transaction():
            row = self.db.execute(
                f"SELECT COALESCE(MAX(seq), 0), COALESCE(MAX(turn), 0) FROM {self._m} "
                f"WHERE tenant_id = ? AND session_id = ?",
                [self.tenant_id, self.session_id]).fetchone()
            seq = int(row[0] or 0)
            turn = int(row[1] or 0)
            added_turns = 0
            for item in items:
                item_type, role, tool_name, text = classify_item(item)
                if _is_user_message(item_type, role):
                    turn += 1
                    added_turns += 1
                seq += 1
                self.db.execute(
                    f"INSERT INTO {self._m} (message_id, tenant_id, session_id, seq, turn, "
                    "item_type, role, tool_name, item_text, item_json, created_at, "
                    "valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, NULL, 1.0)",
                    [new_id(), self.tenant_id, self.session_id, seq, turn, item_type, role,
                     tool_name, text, _dumps(item), now, now, now, self.writer])
            self._touch_session(now, turns=added_turns, items=len(items))

    def _touch_session(self, now: _dt.datetime, *, turns: int, items: int) -> None:
        updated = self.db.execute(
            f"UPDATE {self._s} SET updated_at = ?, turns = turns + ?, items = items + ? "
            f"WHERE tenant_id = ? AND session_id = ? AND {_CURRENT}",
            [now, turns, items, self.tenant_id, self.session_id])
        if (updated.fetchone() or [0])[0]:
            return
        self.db.execute(
            f"INSERT INTO {self._s} (agent_session_id, tenant_id, session_id, created_at, "
            "updated_at, turns, items, valid_from, valid_to, tx_from, tx_to, writer, "
            "episode_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, NULL, 1.0)",
            [new_id(), self.tenant_id, self.session_id, now, now, turns, items, now, now,
             self.writer])

    async def pop_item(self) -> Any | None:
        """Remove and return the newest item, or ``None`` when the session is empty.

        With ``clear_mode="close"`` the row is closed rather than deleted: it leaves the
        conversation the agent sees, and stays in the table for audit.
        """
        return await asyncio.to_thread(self._pop_item_sync)

    def _pop_item_sync(self) -> Any | None:
        with self.db.transaction():
            while True:
                row = self.db.execute(
                    f"SELECT message_id, item_json FROM {self._m} "
                    f"WHERE tenant_id = ? AND session_id = ?{self._visible} "
                    "ORDER BY seq DESC, message_id DESC LIMIT 1",
                    [self.tenant_id, self.session_id]).fetchone()
                if row is None:
                    return None
                message_id, payload = int(row[0]), row[1]
                self._remove(message_id)
                try:
                    return json.loads(payload)
                except (TypeError, ValueError):
                    # Unreadable row: it is gone now, keep looking for a usable item.
                    log.warning("session %r: dropped unreadable item %s", self.session_id,
                                message_id)

    def _remove(self, message_id: int) -> None:
        if self.clear_mode == "delete":
            self.db.execute(f"DELETE FROM {self._m} WHERE tenant_id = ? AND message_id = ?",
                            [self.tenant_id, message_id])
        else:
            now = utcnow()
            self.db.execute(
                f"UPDATE {self._m} SET valid_to = ?, tx_to = ? "
                f"WHERE tenant_id = ? AND message_id = ? AND {_CURRENT}",
                [now, now, self.tenant_id, message_id])

    async def clear_session(self) -> None:
        """Remove every item in this session (and its ``agent_sessions`` row)."""
        await asyncio.to_thread(self._clear_session_sync)

    def _clear_session_sync(self) -> None:
        with self.db.transaction():
            if self.clear_mode == "delete":
                for table in (self._m, self._s, self._u):
                    self.db.execute(
                        f"DELETE FROM {table} WHERE tenant_id = ? AND session_id = ?",
                        [self.tenant_id, self.session_id])
            else:
                now = utcnow()
                for table in (self._m, self._s, self._u):
                    self.db.execute(
                        f"UPDATE {table} SET valid_to = ?, tx_to = ? "
                        f"WHERE tenant_id = ? AND session_id = ? AND {_CURRENT}",
                        [now, now, self.tenant_id, self.session_id])

    # ------------------------------------------------------------------ usage / analytics

    async def store_run_usage(self, result: Any, *, turn: int | None = None) -> bool:
        """Record ``result.context_wrapper.usage`` against the session's current turn.

        Call it after ``Runner.run()``, exactly as with the SDK's ``AdvancedSQLiteSession``.
        Returns whether a row was written.  Never raises for a result without usage -- a missing
        rollup must not fail a run that already succeeded.
        """
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if usage is None:
            return False
        return await asyncio.to_thread(self._store_usage_sync, usage, turn)

    def _store_usage_sync(self, usage: Any, turn: int | None) -> bool:
        now = utcnow()
        if turn is None:
            row = self.db.execute(
                f"SELECT COALESCE(MAX(turn), 0) FROM {self._m} "
                f"WHERE tenant_id = ? AND session_id = ?",
                [self.tenant_id, self.session_id]).fetchone()
            turn = int((row or [0])[0] or 0)
        details_in = getattr(usage, "input_tokens_details", None)
        details_out = getattr(usage, "output_tokens_details", None)
        self.db.execute(
            f"INSERT INTO {self._u} (turn_usage_id, tenant_id, session_id, turn, requests, "
            "input_tokens, output_tokens, total_tokens, cached_tokens, reasoning_tokens, "
            "created_at, valid_from, valid_to, tx_from, tx_to, writer, episode_id, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, NULL, 1.0)",
            [new_id(), self.tenant_id, self.session_id, int(turn),
             int(getattr(usage, "requests", 0) or 0),
             int(getattr(usage, "input_tokens", 0) or 0),
             int(getattr(usage, "output_tokens", 0) or 0),
             int(getattr(usage, "total_tokens", 0) or 0),
             int(getattr(details_in, "cached_tokens", 0) or 0),
             int(getattr(details_out, "reasoning_tokens", 0) or 0),
             now, now, now, self.writer])
        return True

    async def usage_totals(self, *, all_sessions: bool = False) -> dict[str, int] | None:
        """Token/request rollup for this session (or every session in the tenant)."""
        return await asyncio.to_thread(self._usage_totals_sync, all_sessions)

    def _usage_totals_sync(self, all_sessions: bool) -> dict[str, int] | None:
        where, params = self._scope(all_sessions)
        row = self.db.execute(
            "SELECT COALESCE(SUM(requests), 0), COALESCE(SUM(input_tokens), 0), "
            "COALESCE(SUM(output_tokens), 0), COALESCE(SUM(total_tokens), 0), "
            "COALESCE(SUM(cached_tokens), 0), COALESCE(SUM(reasoning_tokens), 0), "
            f"COUNT(*) FROM {self._u} WHERE {where}", params).fetchone()
        if row is None or not row[6]:
            return None
        return {"requests": int(row[0]), "input_tokens": int(row[1]),
                "output_tokens": int(row[2]), "total_tokens": int(row[3]),
                "cached_tokens": int(row[4]), "reasoning_tokens": int(row[5]),
                "rows": int(row[6])}

    async def usage_by_day(self, *, all_sessions: bool = False) -> list[dict[str, Any]]:
        """Token totals per calendar day (UTC), oldest first."""
        return await asyncio.to_thread(self._usage_by_day_sync, all_sessions)

    def _usage_by_day_sync(self, all_sessions: bool) -> list[dict[str, Any]]:
        where, params = self._scope(all_sessions)
        rows = self.db.execute(
            "SELECT CAST(created_at AS DATE) AS day, COUNT(*) AS runs, "
            "COALESCE(SUM(requests), 0), COALESCE(SUM(input_tokens), 0), "
            "COALESCE(SUM(output_tokens), 0), COALESCE(SUM(total_tokens), 0) "
            f"FROM {self._u} WHERE {where} GROUP BY day ORDER BY day", params).fetchall()
        return [{"day": r[0], "runs": int(r[1]), "requests": int(r[2]),
                 "input_tokens": int(r[3]), "output_tokens": int(r[4]),
                 "total_tokens": int(r[5])} for r in rows]

    async def turn_counts_by_day(self, *, all_sessions: bool = False) -> list[dict[str, Any]]:
        """User turns and stored items per calendar day (UTC), oldest first."""
        return await asyncio.to_thread(self._turn_counts_sync, all_sessions)

    def _turn_counts_sync(self, all_sessions: bool) -> list[dict[str, Any]]:
        where, params = self._scope(all_sessions, visible=True)
        rows = self.db.execute(
            "SELECT CAST(created_at AS DATE) AS day, COUNT(DISTINCT turn) AS turns, "
            "COUNT(*) AS items, COUNT(DISTINCT session_id) AS sessions "
            f"FROM {self._m} WHERE {where} GROUP BY day ORDER BY day", params).fetchall()
        return [{"day": r[0], "turns": int(r[1]), "items": int(r[2]), "sessions": int(r[3])}
                for r in rows]

    async def item_type_counts(self, *, all_sessions: bool = False) -> list[dict[str, Any]]:
        """How many items of each ``item_type``/``role``, most frequent first."""
        return await asyncio.to_thread(self._item_type_counts_sync, all_sessions)

    def _item_type_counts_sync(self, all_sessions: bool) -> list[dict[str, Any]]:
        where, params = self._scope(all_sessions, visible=True)
        rows = self.db.execute(
            f"SELECT item_type, role, COUNT(*) AS n FROM {self._m} WHERE {where} "
            "GROUP BY item_type, role ORDER BY n DESC, item_type", params).fetchall()
        return [{"item_type": r[0], "role": r[1], "count": int(r[2])} for r in rows]

    async def tool_usage(self, *, all_sessions: bool = False) -> list[dict[str, Any]]:
        """Tool-call counts by tool name, most-called first."""
        return await asyncio.to_thread(self._tool_usage_sync, all_sessions)

    def _tool_usage_sync(self, all_sessions: bool) -> list[dict[str, Any]]:
        where, params = self._scope(all_sessions, visible=True)
        rows = self.db.execute(
            f"SELECT tool_name, COUNT(*) AS calls, COUNT(DISTINCT turn) AS turns "
            f"FROM {self._m} WHERE {where} AND tool_name IS NOT NULL "
            "GROUP BY tool_name ORDER BY calls DESC, tool_name", params).fetchall()
        return [{"tool_name": r[0], "calls": int(r[1]), "turns": int(r[2])} for r in rows]

    def _scope(self, all_sessions: bool, *, visible: bool = False) -> tuple[str, list[Any]]:
        where = "tenant_id = ?"
        params: list[Any] = [self.tenant_id]
        if not all_sessions:
            where += " AND session_id = ?"
            params.append(self.session_id)
        if visible:
            where += self._visible
        return where, params

    # ------------------------------------------------------------------ history x knowledge

    async def entities_mentioned(self, *, limit: int = 20,
                                 all_sessions: bool = False) -> list[dict[str, Any]]:
        """Which known entities this conversation talked about, and how often.

        **This is the joint query the SQLite session cannot write**: one statement over
        ``agent_messages`` (history) and ``entities`` (knowledge), because they are tables in the
        same DuckDB file.  The match is a case-insensitive substring test on the entity name --
        cheap, obvious, and good enough to rank; it is not entity linking, and a name that is
        also a common word will over-match.
        """
        return await asyncio.to_thread(self._entities_mentioned_sync, limit, all_sessions)

    def _entities_mentioned_sync(self, limit: int, all_sessions: bool) -> list[dict[str, Any]]:
        params: list[Any] = [self.tenant_id]
        session_clause = ""
        if not all_sessions:
            session_clause = " AND m.session_id = ?"
            params.append(self.session_id)
        visible = " AND m.valid_to IS NULL AND m.tx_to IS NULL" if self.clear_mode == "close" else ""
        params.append(int(limit))
        rows = self.db.execute(
            "SELECT e.entity_id, e.name, e.kind, COUNT(DISTINCT m.message_id) AS mentions\n"
            f"FROM {self._m} AS m\n"
            "JOIN entities AS e\n"
            "  ON e.tenant_id = m.tenant_id\n"
            " AND e.valid_to IS NULL AND e.tx_to IS NULL\n"
            " AND e.name IS NOT NULL AND e.name <> ''\n"
            " AND contains(lower(m.item_text), lower(e.name))\n"
            "WHERE m.tenant_id = ?" + session_clause + visible + "\n"
            "  AND m.item_text IS NOT NULL AND m.item_text <> ''\n"
            "GROUP BY e.entity_id, e.name, e.kind\n"
            "ORDER BY mentions DESC, e.name\n"
            "LIMIT ?", params).fetchall()
        return [{"entity_id": int(r[0]), "name": r[1], "kind": r[2], "mentions": int(r[3])}
                for r in rows]

    async def memories_written_here(self, *, limit: int = 20) -> list[Memory]:
        """Memories written by this session's tools (``writer = session:<id>``), newest first."""
        return await asyncio.to_thread(self._memories_written_sync, limit)

    def _memories_written_sync(self, limit: int) -> list[Memory]:
        from ...schema import MEMORY_COLUMNS

        cols = ", ".join(MEMORY_COLUMNS)
        rows = self.db.execute(
            f"SELECT {cols} FROM memories WHERE tenant_id = ? AND writer = ? AND {_CURRENT} "
            "ORDER BY created_at DESC, memory_id DESC LIMIT ?",
            [self.tenant_id, self.writer, int(limit)]).fetchall()
        return [Memory.from_row(r) for r in rows]

    async def transcript(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """The history as readable rows (``seq, turn, item_type, role, tool_name, text``).

        For humans and dashboards; :meth:`get_items` is what the SDK consumes.
        """
        return await asyncio.to_thread(self._transcript_sync, limit)

    def _transcript_sync(self, limit: int | None) -> list[dict[str, Any]]:
        sql = (f"SELECT seq, turn, item_type, role, tool_name, item_text, created_at "
               f"FROM {self._m} WHERE tenant_id = ? AND session_id = ?{self._visible} "
               "ORDER BY seq, message_id")
        params: list[Any] = [self.tenant_id, self.session_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self.db.execute(sql, params).fetchall()
        return [{"seq": int(r[0]), "turn": int(r[1]), "item_type": r[2], "role": r[3],
                 "tool_name": r[4], "text": r[5], "created_at": r[6]} for r in rows]
