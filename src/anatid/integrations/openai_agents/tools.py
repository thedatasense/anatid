"""Approval-gated anatid memory tools for the OpenAI Agents SDK.

Six tools, in the shape agent-memory tools have converged on (a text ``content`` to write, a
text ``query`` to search, ids to amend or erase), so swapping a Cognee / neo4j-memory style
tool-set for this one is a change of names, not of protocol:

===================== ====== ==============================================================
tool                  write? what it does
===================== ====== ==============================================================
``anatid_remember``   yes    write one memory, optionally about named entities
``anatid_recall``     no     hybrid recall (vector + BM25 + graph, fused with RRF)
``anatid_context``    no     what is known about one entity
``anatid_supersede``  yes    replace a memory with a corrected one, keeping the old version
``anatid_forget``     yes    stop believing a memory (soft) or erase it (hard)
``anatid_provenance`` no     where a memory came from: its chain, episodes and writers
===================== ====== ==============================================================

**Reads are not gated.  Writes are.**  The three write tools are created with
``needs_approval=<policy>``, so the SDK interrupts the run before the tool body executes and
hands you a ``ToolApprovalItem`` on ``RunResult.interruptions``; nothing touches the database
until someone calls ``state.approve(...)``.  The policy is a callable, so a caller can
auto-approve low-risk writes and still require a human for hard deletes:

.. code-block:: python

    from anatid.integrations.openai_agents import create_memory_tools, approve_low_risk

    tools = create_memory_tools(db, approval_policy=approve_low_risk())

:func:`always_require_approval` is the default.  Whatever policy you pass,
``anatid_forget(hard=True)`` still requires approval unless you explicitly opt out with
``force_approval_for_hard_forget=False`` -- a hard forget deletes the row, its edges, its
embedding and its provenance, and that is not a decision to hand to a model.

Every tool returns a JSON string.  anatid errors are returned as ``{"error": ..., "message":
...}`` rather than raised, so the model can correct itself instead of failing the run;
everything else propagates.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ...database import Anatid
from ...errors import AnatidError
from ...types import Namespace

log = logging.getLogger("anatid.integrations.openai_agents")

# Imported at module scope on purpose: ``from __future__ import annotations`` turns the tool
# signatures into strings, and the SDK resolves them with ``typing.get_type_hints`` against this
# module's globals -- so ``RunContextWrapper`` has to live here, not inside the factory.  The SDK
# stays optional: without it these are None and :func:`create_memory_tools` raises before it
# builds anything.
try:  # pragma: no cover - exercised by the installed-SDK path
    from agents import RunContextWrapper, function_tool  # type: ignore[import-not-found]

    _SDK_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - the SDK is an optional dependency
    RunContextWrapper = None  # type: ignore[assignment,misc]
    function_tool = None  # type: ignore[assignment]
    _SDK_IMPORT_ERROR = _exc

__all__ = [
    "ApprovalRequest",
    "ApprovalPolicy",
    "always_require_approval",
    "never_require_approval",
    "approve_low_risk",
    "create_memory_tools",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "TOOL_NAMES",
]

READ_TOOLS: tuple[str, ...] = ("anatid_recall", "anatid_context", "anatid_provenance")
WRITE_TOOLS: tuple[str, ...] = ("anatid_remember", "anatid_supersede", "anatid_forget")
TOOL_NAMES: tuple[str, ...] = WRITE_TOOLS + READ_TOOLS


# --------------------------------------------------------------------------- approval policy

@dataclass(frozen=True)
class ApprovalRequest:
    """One "may I run this write?" question, handed to an :data:`ApprovalPolicy`.

    ``arguments`` is the model's parsed tool call.  It has already been JSON-decoded by the SDK;
    if the model produced arguments that could not be decoded the SDK does not call the policy at
    all and requires approval, which is the safe direction.
    """

    tool_name: str
    arguments: Mapping[str, Any]
    call_id: str
    context: Any = None

    @property
    def is_hard_forget(self) -> bool:
        """True for ``anatid_forget(hard=True)`` -- the irreversible one."""
        return self.tool_name == "anatid_forget" and bool(self.arguments.get("hard"))


#: Return True to require a human, False to let the call through.  Sync or async.
ApprovalPolicy = Callable[[ApprovalRequest], "bool | Awaitable[bool]"]


def always_require_approval(request: ApprovalRequest) -> bool:
    """The default: every write waits for a human."""
    return True


def never_require_approval(request: ApprovalRequest) -> bool:
    """Approve nothing to a human -- every write runs immediately.

    Only sane for a trusted batch job.  Note that ``create_memory_tools`` still forces approval
    for a hard forget unless you pass ``force_approval_for_hard_forget=False``.
    """
    return False


def approve_low_risk(
    *,
    remember: bool = True,
    soft_forget: bool = False,
    supersede: bool = False,
    max_content_chars: int | None = 2000,
) -> ApprovalPolicy:
    """A policy that lets low-risk writes through and stops the rest.

    ``remember=True`` (default) auto-approves ``anatid_remember`` -- an append that
    ``anatid_forget`` can undo.  ``soft_forget`` and ``supersede`` default to False because both
    change what the agent believes is currently true.  A hard forget is never auto-approved by
    this policy.  ``max_content_chars`` sends unusually large writes to a human anyway; pass
    ``None`` to disable that check.
    """

    def policy(request: ApprovalRequest) -> bool:
        if request.is_hard_forget:
            return True
        content = request.arguments.get("content")
        if (max_content_chars is not None and isinstance(content, str)
                and len(content) > max_content_chars):
            return True
        if request.tool_name == "anatid_remember":
            return not remember
        if request.tool_name == "anatid_forget":
            return not soft_forget
        if request.tool_name == "anatid_supersede":
            return not supersede
        return True

    return policy


def _needs_approval_adapter(
    tool_name: str,
    policy: ApprovalPolicy,
    *,
    force_hard_forget: bool,
) -> Callable[[Any, dict[str, Any], str], Awaitable[bool]]:
    """Wrap an :data:`ApprovalPolicy` in the signature the SDK calls.

    The SDK invokes ``needs_approval(run_context, parsed_arguments, call_id)`` and awaits the
    result if it is awaitable (``agents.util._approvals.evaluate_needs_approval_setting``).  It
    does not pass the tool name, which is why anatid binds one adapter per tool.
    """

    async def needs_approval(context: Any, arguments: dict[str, Any], call_id: str) -> bool:
        request = ApprovalRequest(tool_name=tool_name, arguments=arguments or {},
                                  call_id=call_id, context=context)
        if force_hard_forget and request.is_hard_forget:
            return True
        verdict = policy(request)
        if hasattr(verdict, "__await__"):
            verdict = await verdict  # type: ignore[misc]
        return bool(verdict)

    needs_approval.__name__ = f"{tool_name}_needs_approval"
    return needs_approval


# --------------------------------------------------------------------------- serialisation

def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, (_dt.datetime, _dt.date)) else value


def _memory_json(memory: Any, *, about: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "memory_id": memory.memory_id,
        "content": memory.content,
        "kind": memory.kind,
        "created_at": _iso(memory.created_at),
        "confidence": memory.confidence,
        "access_count": memory.access_count,
        "writer": memory.writer,
        "about": list(about),
    }


def _ok(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _err(exc: Exception) -> str:
    return json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False)


# --------------------------------------------------------------------------- the factory

def create_memory_tools(
    db: Anatid,
    *,
    tenant: int | Namespace | None = None,
    writer: str | None = None,
    session: Any = None,
    approval_policy: ApprovalPolicy | None = None,
    force_approval_for_hard_forget: bool = True,
    embedder: Callable[[str], Sequence[float]] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    default_k: int = 8,
    default_limit: int = 10,
) -> list[Any]:
    """Build the anatid memory tools for an ``Agent``.

    ``db``
        The open :class:`~anatid.Anatid` handle the tools read and write.  Tool bodies are sync
        functions; the SDK runs them in a worker thread, and anatid keeps one DuckDB connection
        per thread, so this is safe.
    ``writer`` / ``session``
        What to stamp in the ``writer`` provenance column.  With ``session=`` (an
        :class:`~anatid.integrations.openai_agents.AnatidSession`) it defaults to that session's
        writer tag, which is what makes ``session.memories_written_here()`` work.  Otherwise it
        falls back to ``agent:<name>`` from the run context, then to ``None``.
    ``approval_policy``
        See :data:`ApprovalPolicy`.  Defaults to :func:`always_require_approval`.
    ``embedder``
        Optional ``str -> vector`` callable.  Given one, ``anatid_remember`` stores an embedding
        and ``anatid_recall`` runs the vector arm; without one, recall is BM25 + graph only, and
        says so in its result (``"arms"``).  anatid never calls an embedding API itself.
    ``include`` / ``exclude``
        Tool-name filters, e.g. ``exclude=["anatid_forget"]`` for an agent that may not delete.

    Returns the ``FunctionTool`` list, in the order of :data:`TOOL_NAMES`.
    """
    if function_tool is None:  # pragma: no cover - depends on the environment
        raise ImportError(
            "anatid.integrations.openai_agents.tools needs the OpenAI Agents SDK: "
            'pip install "anatid[agents]"'
        ) from _SDK_IMPORT_ERROR

    policy = approval_policy or always_require_approval
    if session is not None and writer is None:
        writer = getattr(session, "writer", None)
    if session is not None and tenant is None:
        tenant = getattr(session, "namespace", None)
    ns = db.resolve_tenant(tenant)

    wanted = set(include) if include is not None else set(TOOL_NAMES)
    wanted -= set(exclude or ())
    unknown = wanted - set(TOOL_NAMES)
    if unknown:
        raise ValueError(f"unknown tool name(s): {sorted(unknown)}")

    def writer_for(ctx: Any) -> str | None:
        if writer is not None:
            return writer
        agent = getattr(ctx, "agent", None)
        name = getattr(agent, "name", None)
        return f"agent:{name}" if isinstance(name, str) and name else None

    def gate(tool_name: str) -> Any:
        return _needs_approval_adapter(tool_name, policy,
                                       force_hard_forget=force_approval_for_hard_forget)

    # ---------------------------------------------------------------- write tools

    @function_tool(name_override="anatid_remember", needs_approval=gate("anatid_remember"))
    def anatid_remember(
        ctx: RunContextWrapper[Any],
        content: str,
        entities: list[str] | None = None,
        kind: str | None = None,
    ) -> str:
        """Save one durable fact to long-term memory.

        Args:
            content: The fact to remember, as one self-contained sentence.
            entities: Names this fact is about, e.g. ["Ada Lovelace", "DuckDB"]. They become
                graph nodes so later recall can walk from an entity to everything about it.
            kind: A label for the memory, e.g. "fact", "preference", "decision". Defaults to
                "fact".
        """
        try:
            memory = db.remember(
                content,
                entities=tuple(entities or ()),
                kind=kind or "fact",
                embedding=list(embedder(content)) if embedder is not None else None,
                writer=writer_for(ctx),
                tenant=ns,
            )
            return _ok({"saved": True, **_memory_json(memory, about=entities or ())})
        except AnatidError as exc:
            return _err(exc)

    @function_tool(name_override="anatid_supersede", needs_approval=gate("anatid_supersede"))
    def anatid_supersede(
        ctx: RunContextWrapper[Any],
        memory_id: int,
        content: str,
        entities: list[str] | None = None,
        kind: str | None = None,
    ) -> str:
        """Replace a memory that is now wrong with a corrected version.

        The old memory is kept and marked as superseded, so the correction is auditable and
        anatid_provenance can walk back to what was believed before.

        Args:
            memory_id: The id of the memory being corrected.
            content: The corrected fact.
            entities: Names the corrected fact is about. Omit to keep the old memory's entities.
            kind: Label for the new memory. Omit to keep the old one's.
        """
        try:
            memory = db.supersede(
                int(memory_id),
                content,
                entities=None if entities is None else tuple(entities),
                kind=kind,
                embedding=list(embedder(content)) if embedder is not None else None,
                writer=writer_for(ctx),
                tenant=ns,
            )
            return _ok({"superseded": int(memory_id), **_memory_json(memory)})
        except AnatidError as exc:
            return _err(exc)

    @function_tool(name_override="anatid_forget", needs_approval=gate("anatid_forget"))
    def anatid_forget(
        ctx: RunContextWrapper[Any],
        memory_id: int,
        hard: bool | None = None,
        reason: str | None = None,
    ) -> str:
        """Stop believing a memory.

        Args:
            memory_id: The id of the memory to forget.
            hard: False (the default) closes the memory so it stops being recalled but stays in
                history. True permanently erases the row, its edges, its embedding and its
                provenance -- use only for an actual erasure request.
            reason: Why, recorded in the audit trail.
        """
        try:
            receipt = db.forget(int(memory_id), hard=bool(hard), reason=reason,
                                writer=writer_for(ctx), tenant=ns)
            return _ok({
                "memory_id": receipt.memory_id,
                "hard": receipt.hard,
                "at": _iso(receipt.at),
                "rows_removed": receipt.rows_removed,
                "reason": receipt.reason,
            })
        except AnatidError as exc:
            return _err(exc)

    # ---------------------------------------------------------------- read tools

    @function_tool(name_override="anatid_recall")
    def anatid_recall(
        ctx: RunContextWrapper[Any],
        query: str,
        k: int | None = None,
        seed_entity: str | None = None,
        hops: int | None = None,
    ) -> str:
        """Search long-term memory for anything relevant to a question.

        Args:
            query: What to look for, in natural language.
            k: How many memories to return. Defaults to 8.
            seed_entity: Optional entity name to start a graph walk from, which pulls in
                memories that share entities with it even when the words do not match.
            hops: How far to walk from seed_entity, 1 or 2. Defaults to 2.
        """
        try:
            notes: list[str] = []
            seed: Any = None
            if seed_entity:
                try:
                    seed = db.entity_id(seed_entity, tenant=ns, create=False)
                except AnatidError:
                    notes.append(f"unknown entity {seed_entity!r}; graph arm skipped")
            hits = db.recall(
                query,
                k=int(k) if k else default_k,
                embedding=list(embedder(query)) if embedder is not None else None,
                seed_entity=seed,
                hops=int(hops) if hops else 2,
                tenant=ns,
            )
            return _ok({
                "hits": [
                    {**_memory_json(hit.memory, about=hit.about),
                     "score": round(hit.score, 6), "rank": hit.rank,
                     "sources": list(hit.sources)}
                    for hit in hits
                ],
                "arms": list(hits.arms),
                "bm25_stale": hits.bm25_stale,
                "pending_fts_rows": hits.pending_fts_rows,
                "notes": notes + list(hits.notes),
            })
        except AnatidError as exc:
            return _err(exc)

    @function_tool(name_override="anatid_context")
    def anatid_context(
        ctx: RunContextWrapper[Any],
        entity: str,
        limit: int | None = None,
        hops: int | None = None,
    ) -> str:
        """List what is known about one person, project or thing.

        Args:
            entity: The entity's name, e.g. "Ada Lovelace".
            limit: How many memories to return. Defaults to 10.
            hops: 0 (default) for memories directly about the entity, 1 to include its
                neighbours, 2 to include their neighbours too.
        """
        try:
            memories = db.context(entity, limit=int(limit) if limit else default_limit,
                                  hops=int(hops) if hops else 0, tenant=ns)
            return _ok({"entity": entity, "hops": int(hops) if hops else 0,
                        "memories": [_memory_json(m) for m in memories]})
        except AnatidError as exc:
            return _err(exc)

    @function_tool(name_override="anatid_provenance")
    def anatid_provenance(ctx: RunContextWrapper[Any], memory_id: int) -> str:
        """Explain where a memory came from: its correction chain, sources and writers.

        Args:
            memory_id: The id of the memory to explain.
        """
        try:
            prov = db.provenance(int(memory_id), tenant=ns)
            return _ok({
                "memory_id": prov.memory_id,
                "depth": prov.depth,
                "chain": [_memory_json(m) for m in prov.chain],
                "source_text": prov.source_text,
                "episodes": [{"episode_id": e.episode_id, "source": e.source,
                              "content": e.content, "created_at": _iso(e.created_at)}
                             for e in prov.episodes],
                "writers": list(prov.writers),
            })
        except AnatidError as exc:
            return _err(exc)

    built = {
        "anatid_remember": anatid_remember,
        "anatid_supersede": anatid_supersede,
        "anatid_forget": anatid_forget,
        "anatid_recall": anatid_recall,
        "anatid_context": anatid_context,
        "anatid_provenance": anatid_provenance,
    }
    return [built[name] for name in TOOL_NAMES if name in wanted]
