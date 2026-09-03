"""Durable human-in-the-loop: park an interrupted run in the anatid file, resume it later.

The OpenAI Agents SDK already gives you the mechanism -- a tool with ``needs_approval`` stops the
run, ``RunResult.interruptions`` lists the ``ToolApprovalItem``s, ``result.to_state()`` captures
everything and ``RunState.to_string()`` serialises it.  What it does not give you is somewhere to
*put* that string, which is why every SDK example writes it to a file next to the script.

:class:`RunStateStore` puts it in the **same DuckDB file as the agent's memory**, in a table with
anatid's system columns, so:

* an approval can be answered minutes or days later, by another process, on the same file;
* saving a new state for a run closes the previous row instead of overwriting it, so the whole
  approval conversation is auditable and ``db.as_of(t)`` reads it back;
* ``store.pending()`` is a work queue -- one SQL statement, no extra service.

.. code-block:: python

    # process A -- the run stops on an approval
    result = await Runner.run(agent, "remember that Ada prefers DuckDB", session=session)
    if result.interruptions:
        run_id = store.save_result(result, session_id=session.session_id)

    # process B -- minutes or days later
    state = await store.resume(agent, run_id)          # RunState.from_string under the hood
    for item in state.get_interruptions():
        state.approve(item)                            # or state.reject(item)
    result = await Runner.run(agent, state)
    store.mark_resolved(run_id, status="approved")

What is in the string
---------------------
``RunState.to_string()`` contains the conversation so far, the pending tool calls and their
arguments, and your run context.  It is **not** encrypted and anatid does not encrypt it -- treat
the anatid file as being as sensitive as the conversation.  The SDK leaves the tracing API key
out unless you pass ``include_tracing_api_key=True``; :meth:`RunStateStore.save` never passes it.

Erasure
-------
Because that string is a verbatim copy of the conversation, ``agent_run_states`` is a copy of
every memory the conversation quoted -- so :meth:`RunStateStore.__init__` registers an erasure
hook on the handle for its table.  Before that, ``db.forget(mid, hard=True)`` returned a receipt
saying the memory had been erased while this table still held the id *and* the text, and
:meth:`resume` would hand them back to a model days later.  A parked run that quotes an erased
memory is deleted whole: a partially redacted ``RunState`` would not deserialise, and keeping it
would defeat the erasure.  See :mod:`anatid.integrations.erasure`.

Resuming needs the *same agent* (same name, tools and handoffs) that produced the state:
``RunState.from_string(agent, s)`` rebuilds the run against the agent you hand it, so process B
must construct an identical ``Agent``.  A state saved by a different version of your agent code
is not automatically compatible, which is why :meth:`save` records ``agent_name`` and lets you
attach your own ``metadata`` (a build id, a prompt version) to check against.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from ...database import Anatid
from ...errors import NotFoundError
from ...ids import new_id
from ...schema import quote_ident
from ...types import Namespace, utcnow
from ..erasure import register_table_erasure_hooks

log = logging.getLogger("anatid.integrations.openai_agents")

__all__ = ["RunStateStore", "StoredRun", "RUN_STATE_COLUMNS"]

#: ``agent_run_states`` columns beyond ``run_state_id``/``tenant_id`` and the system columns.
RUN_STATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_id", "VARCHAR"),
    ("session_id", "VARCHAR"),
    ("agent_name", "VARCHAR"),
    ("status", "VARCHAR"),
    ("pending_tools", "VARCHAR"),
    ("state_json", "VARCHAR"),
    ("metadata", "VARCHAR"),
    ("created_at", "TIMESTAMP"),
    ("updated_at", "TIMESTAMP"),
)

_CURRENT = "valid_to IS NULL AND tx_to IS NULL"

#: Status of a saved run.  Free-form -- these are the ones anatid itself writes.
PENDING = "pending_approval"


@dataclass(frozen=True)
class StoredRun:
    """One saved run state, without the (potentially large) state string."""

    run_id: str
    tenant_id: int
    status: str
    agent_name: str | None = None
    session_id: str | None = None
    pending_tools: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    writer: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.status == PENDING


def _interruption_tool_names(result: Any) -> tuple[str, ...]:
    names: list[str] = []
    for item in getattr(result, "interruptions", ()) or ():
        name = getattr(item, "tool_name", None)
        if not isinstance(name, str):
            raw = getattr(item, "raw_item", None)
            name = raw.get("name") if isinstance(raw, Mapping) else getattr(raw, "name", None)
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)


class RunStateStore:
    """Save, list and reload ``RunState`` strings in an anatid database.

    ``RunStateStore(db)`` uses the handle's tenant; ``RunStateStore(db, tenant=7)`` pins another.
    As everywhere in anatid, ``tenant_id`` here is *scoping*, not isolation -- one file per tenant
    (:class:`anatid.DatabasePool`) is the isolation story.

    Every method is synchronous except :meth:`resume`, which awaits the SDK's async
    ``RunState.from_string``.
    """

    def __init__(self, db: Anatid, *, tenant: int | Namespace | None = None,
                 table: str = "agent_run_states", writer: str | None = None) -> None:
        self.db = db
        self.namespace = db.resolve_tenant(tenant)
        self.tenant_id = self.namespace.tenant_id
        self.table_name = table
        self._t = quote_ident(table)
        self.writer = writer
        self.db.create_node_label(table, RUN_STATE_COLUMNS, id_column="run_state_id")
        #: ``state_json`` is a verbatim copy of the conversation, so a hard forget has to reach
        #: it.  Deduplicated by table name, so two stores on one handle register one hook.
        self.erasure_hooks = register_table_erasure_hooks(self.db, (table,))

    def __repr__(self) -> str:
        return (f"<RunStateStore path={self.db.path!r} tenant={self.tenant_id} "
                f"table={self.table_name!r}>")

    # ------------------------------------------------------------------ writing

    def save(
        self,
        state: Any,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        status: str = PENDING,
        pending_tools: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        writer: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """Persist a ``RunState`` (or an already-serialised state string) and return its ``run_id``.

        Saving again under the same ``run_id`` closes the previous row (``valid_to``/``tx_to``)
        and appends a new one, so nothing is overwritten and the history of an approval is
        readable with ``db.as_of(t)``.
        """
        if isinstance(state, str):
            state_json = state
        else:
            to_string = getattr(state, "to_string", None)
            if not callable(to_string):
                raise TypeError(
                    "save() takes a RunState or the string from RunState.to_string(), "
                    f"got {type(state).__name__}")
            state_json = to_string()
        if not isinstance(state_json, str):  # pragma: no cover - defensive
            raise TypeError("RunState.to_string() did not return a string")

        rid = run_id or f"run_{uuid.uuid4().hex}"
        at = now or utcnow()
        agent_name = agent_name or self._agent_name_of(state)
        row_writer = writer if writer is not None else self.writer

        with self.db.transaction():
            existing = self.db.execute(
                f"SELECT created_at FROM {self._t} WHERE tenant_id = ? AND run_id = ? "
                f"AND {_CURRENT} ORDER BY updated_at DESC LIMIT 1",
                [self.tenant_id, rid]).fetchone()
            created_at = existing[0] if existing else at
            if existing:
                self.db.execute(
                    f"UPDATE {self._t} SET valid_to = ?, tx_to = ? "
                    f"WHERE tenant_id = ? AND run_id = ? AND {_CURRENT}",
                    [at, at, self.tenant_id, rid])
            self.db.execute(
                f"INSERT INTO {self._t} (run_state_id, tenant_id, run_id, session_id, "
                "agent_name, status, pending_tools, state_json, metadata, created_at, "
                "updated_at, valid_from, valid_to, tx_from, tx_to, writer, episode_id, "
                "confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, NULL, 1.0)",
                [new_id(), self.tenant_id, rid, session_id, agent_name, status,
                 json.dumps(list(pending_tools)), state_json,
                 json.dumps(dict(metadata or {}), default=str), created_at, at, at, at,
                 row_writer])
        return rid

    def save_result(self, result: Any, **kwargs: Any) -> str:
        """:meth:`save` for a ``RunResult`` that stopped on an approval.

        Calls ``result.to_state()`` and records the interrupted tool names, so
        :meth:`pending` can tell you what is waiting without deserialising the state.
        """
        to_state = getattr(result, "to_state", None)
        if not callable(to_state):
            raise TypeError(f"save_result() needs a RunResult, got {type(result).__name__}")
        kwargs.setdefault("pending_tools", _interruption_tool_names(result))
        return self.save(to_state(), **kwargs)

    def mark_resolved(self, run_id: str, *, status: str = "resolved",
                      metadata: Mapping[str, Any] | None = None,
                      now: datetime | None = None) -> bool:
        """Update the current row's ``status`` in place.  Returns whether a row matched.

        This is the one deliberate in-place update: the decision outcome belongs to the row that
        recorded the question.  Under DuckDB's optimistic MVCC two processes resolving the same
        run at the same moment make the second one lose with a retryable
        :class:`~anatid.errors.ConflictError` -- which is the correct answer.
        """
        at = now or utcnow()
        if metadata is None:
            changed = self.db.execute(
                f"UPDATE {self._t} SET status = ?, updated_at = ? "
                f"WHERE tenant_id = ? AND run_id = ? AND {_CURRENT}",
                [status, at, self.tenant_id, run_id])
        else:
            changed = self.db.execute(
                f"UPDATE {self._t} SET status = ?, updated_at = ?, metadata = ? "
                f"WHERE tenant_id = ? AND run_id = ? AND {_CURRENT}",
                [status, at, json.dumps(dict(metadata), default=str), self.tenant_id, run_id])
        return bool((changed.fetchone() or [0])[0])

    def delete(self, run_id: str, *, hard: bool = False, now: datetime | None = None) -> int:
        """Remove a saved run.

        ``hard=False`` closes every row for the run (history preserved, ``load`` stops finding
        it).  ``hard=True`` deletes them, including the serialised conversation inside -- which is
        what a right-to-erasure request over an abandoned approval actually needs.
        """
        if hard:
            deleted = self.db.execute(
                f"DELETE FROM {self._t} WHERE tenant_id = ? AND run_id = ?",
                [self.tenant_id, run_id])
        else:
            at = now or utcnow()
            deleted = self.db.execute(
                f"UPDATE {self._t} SET valid_to = ?, tx_to = ? "
                f"WHERE tenant_id = ? AND run_id = ? AND {_CURRENT}",
                [at, at, self.tenant_id, run_id])
        return int((deleted.fetchone() or [0])[0])

    # ------------------------------------------------------------------ reading

    def load(self, run_id: str) -> str | None:
        """The current serialised state string for a run, or ``None``."""
        row = self.db.execute(
            f"SELECT state_json FROM {self._t} WHERE tenant_id = ? AND run_id = ? "
            f"AND {_CURRENT} ORDER BY updated_at DESC LIMIT 1",
            [self.tenant_id, run_id]).fetchone()
        return None if row is None else row[0]

    def get(self, run_id: str) -> StoredRun | None:
        """Metadata for a run (no state string), or ``None``."""
        row = self.db.execute(
            f"SELECT {_META_COLUMNS} FROM {self._t} WHERE tenant_id = ? AND run_id = ? "
            f"AND {_CURRENT} ORDER BY updated_at DESC LIMIT 1",
            [self.tenant_id, run_id]).fetchone()
        return None if row is None else _stored_run(row)

    def pending(self, *, status: str = PENDING, session_id: str | None = None,
                limit: int = 50) -> list[StoredRun]:
        """The approval work queue: saved runs with ``status``, oldest first."""
        sql = (f"SELECT {_META_COLUMNS} FROM {self._t} "
               f"WHERE tenant_id = ? AND status = ? AND {_CURRENT}")
        params: list[Any] = [self.tenant_id, status]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at, run_id LIMIT ?"
        params.append(int(limit))
        return [_stored_run(row) for row in self.db.execute(sql, params).fetchall()]

    def history(self, run_id: str) -> list[StoredRun]:
        """Every version of a run's row, oldest first -- including the closed ones."""
        rows = self.db.execute(
            f"SELECT {_META_COLUMNS} FROM {self._t} WHERE tenant_id = ? AND run_id = ? "
            "ORDER BY updated_at, run_state_id", [self.tenant_id, run_id]).fetchall()
        return [_stored_run(row) for row in rows]

    async def resume(self, agent: Any, run_id: str, **from_string_kwargs: Any) -> Any:
        """Rebuild the ``RunState`` for ``run_id`` against ``agent``.

        Thin wrapper over the SDK's ``RunState.from_string(agent, state_string)``; extra keyword
        arguments (``context_override``, ``context_deserializer``, ``strict_context``) are passed
        straight through.  Raises :class:`~anatid.errors.NotFoundError` when the run is unknown or
        already closed.
        """
        state_json = self.load(run_id)
        if state_json is None:
            raise NotFoundError(
                f"no saved run {run_id!r} in tenant {self.tenant_id} of {self.db.path!r}")
        try:
            from agents import RunState
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "resume() needs the OpenAI Agents SDK: pip install \"anatid[agents]\""
            ) from exc
        return await RunState.from_string(agent, state_json, **from_string_kwargs)

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _agent_name_of(state: Any) -> str | None:
        for attr in ("_current_agent", "current_agent"):
            agent = getattr(state, attr, None)
            name = getattr(agent, "name", None)
            if isinstance(name, str) and name:
                return name
        return None


_META_COLUMNS = ("run_id, tenant_id, status, agent_name, session_id, pending_tools, metadata, "
                 "created_at, updated_at, writer")


def _loads(payload: Any, fallback: Any) -> Any:
    if not isinstance(payload, str) or not payload:
        return fallback
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return fallback


def _stored_run(row: Iterable[Any]) -> StoredRun:
    (run_id, tenant_id, status, agent_name, session_id, pending_tools, metadata,
     created_at, updated_at, writer) = tuple(row)
    tools = _loads(pending_tools, [])
    meta = _loads(metadata, {})
    return StoredRun(
        run_id=run_id,
        tenant_id=int(tenant_id),
        status=status,
        agent_name=agent_name,
        session_id=session_id,
        pending_tools=tuple(t for t in tools if isinstance(t, str)),
        metadata=meta if isinstance(meta, dict) else {},
        created_at=created_at,
        updated_at=updated_at,
        writer=writer,
    )
