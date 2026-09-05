"""Approval-gated anatid memory tools for the OpenAI Agents SDK.

Nine tools, in the shape agent-memory tools have converged on (a text ``content`` to write, a
text ``query`` to search, ids to amend or erase), so swapping a Cognee / neo4j-memory style
tool-set for this one is a change of names, not of protocol:

===================== ====== ==============================================================
tool                  write? what it does
===================== ====== ==============================================================
``anatid_remember``   yes    write one memory, optionally about named entities
``anatid_relate``     yes    connect two entities, so facts about one are reachable from the other
``anatid_recall``     no     hybrid recall (vector + BM25 + graph, fused with RRF)
``anatid_context``    no     what is known about one entity
``anatid_supersede``  yes    replace a memory with a corrected one, keeping the old version
``anatid_correct``    yes    supersede a memory and close and open its edges, in one transaction
``anatid_unrelate``   yes    retire the connection between two entities
``anatid_forget``     yes    stop believing a memory (soft) or erase it (hard)
``anatid_provenance`` no     where a memory came from: its chain, episodes and writers
``anatid_ingest``     yes    a note in, a reviewed patch of facts and edges applied as one (optional)
===================== ====== ==============================================================

The tenth tool, ``anatid_ingest``, is built only when :func:`create_memory_tools` is given an
``extractor`` (:class:`anatid.ingest.Extractor`), because it hands text to a model.  It is a
write and carries ``needs_approval`` like the others; ``dry_run=True`` proposes the patch and
returns its diff without writing, and a dry run never waits for approval because it changes
nothing.  See :mod:`anatid.ingest` for the pipeline it runs.

The three graph tools exist because memories are filed under the entities they name and
nothing links those entities to each other until an edge says so.  Three facts stored as
"Ada leads Kestrel", "Kestrel owns the ingest service" and "Bo maintains the ingest service"
are three islands: a recall seeded on Ada finds the first and nothing else, until
``anatid_relate("Ada", "Kestrel")`` and ``anatid_relate("Kestrel", "ingest service")`` make the
other two reachable in two hops.  ``anatid_correct`` is for the correction that moves an edge
as well as a sentence, and it moves both in one transaction (:meth:`anatid.Anatid.correct`).

**Reads are not gated.  Writes are.**  The six write tools are created with
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

**Ids are decimal strings**, going out and coming back: ``"883768514279557120"``, never
``883768514279557120``.  anatid ids are 63-bit and a JSON number is a double once it reaches
any JavaScript in the chain, which silently rounds one.  The four tools that take an id
declare a string in their JSON schema and accept a plain integer as well; see
:mod:`anatid.integrations.wire`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Sequence

from ... import AUTO_SEED
from ...database import Anatid
from ...errors import AnatidError
from ...ingest import Extractor, MemoryPatch, ReviewHook
from ...ingest import propose as propose_patch
from ...types import Namespace
from ..wire import WireId, wire_id

log = logging.getLogger("anatid.integrations.openai_agents")

# Imported at module scope on purpose: ``from __future__ import annotations`` turns the tool
# signatures into strings, and the SDK resolves them with ``typing.get_type_hints`` against this
# module's globals -- so ``RunContextWrapper`` has to live here, not inside the factory.  The SDK
# stays optional: without it these are None and :func:`create_memory_tools` raises before it
# builds anything.
_SDK_IMPORT_ERROR: Exception | None = None
if TYPE_CHECKING:
    # The checker always sees the real SDK types; the run-time fallback below is what makes the
    # SDK optional, and it is invisible to the checker on purpose (a variable named
    # RunContextWrapper is not a type, and every tool signature below uses it as one).
    from agents import RunContextWrapper, function_tool
else:
    try:  # pragma: no cover - exercised by the installed-SDK path
        from agents import RunContextWrapper, function_tool
    except Exception as _exc:  # pragma: no cover - the SDK is an optional dependency
        RunContextWrapper = None
        function_tool = None
        _SDK_IMPORT_ERROR = _exc

__all__ = [
    "ApprovalRequest",
    "ApprovalPolicy",
    "always_require_approval",
    "never_require_approval",
    "approve_low_risk",
    "create_memory_tools",
    "Relation",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "TOOL_NAMES",
    "INGEST_TOOL",
]

READ_TOOLS: tuple[str, ...] = ("anatid_recall", "anatid_context", "anatid_provenance")
WRITE_TOOLS: tuple[str, ...] = (
    "anatid_remember",
    "anatid_relate",
    "anatid_supersede",
    "anatid_correct",
    "anatid_unrelate",
    "anatid_forget",
)
#: The nine tools every call to :func:`create_memory_tools` builds.
TOOL_NAMES: tuple[str, ...] = WRITE_TOOLS + READ_TOOLS
#: The tenth, built only when an ``extractor`` is given.  A write: it waits for approval.
INGEST_TOOL = "anatid_ingest"


@dataclass
class Relation:
    """One edge between two entities, as ``anatid_correct`` takes it.

    ``src`` and ``dst`` are entity names; ``rel_kind`` is an optional label such as "leads".
    The SDK builds this into the tool's JSON schema as an object with those three keys, and
    hands the tool body instances of this class.
    """

    src: str
    dst: str
    rel_kind: str | None = None


def _relation_tuple(value: Any) -> tuple[str, str, str | None]:
    """A :class:`Relation`, or the mapping a model sent for one, as the tuple the verb takes."""
    if isinstance(value, Mapping):
        return (str(value["src"]), str(value["dst"]), value.get("rel_kind"))
    return (str(value.src), str(value.dst), value.rel_kind)


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

    @property
    def is_dry_run(self) -> bool:
        """True for ``anatid_ingest(dry_run=True)``, which proposes a patch and writes nothing."""
        return self.tool_name == INGEST_TOOL and bool(self.arguments.get("dry_run"))


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
    relate: bool = False,
    soft_forget: bool = False,
    supersede: bool = False,
    ingest: bool = False,
    max_content_chars: int | None = 2000,
) -> ApprovalPolicy:
    """A policy that lets low-risk writes through and stops the rest.

    ``remember=True`` (default) auto-approves ``anatid_remember`` -- an append that
    ``anatid_forget`` can undo.  ``relate`` defaults to False: an edge is an append that
    ``anatid_unrelate`` can undo, but it changes what graph recall reaches from both entities.
    ``soft_forget`` and ``supersede`` default to False because both change what the agent
    believes is currently true.  ``anatid_unrelate`` and ``anatid_correct`` always go to a human
    under this policy, because both close edges and the second closes a memory as well.
    ``ingest`` defaults to False: a patch a model proposed can add, correct and close several
    things at once, and the diff is what a person should read.  A hard forget is never
    auto-approved by this policy.  ``max_content_chars`` sends unusually large writes to a human
    anyway (for ``anatid_ingest`` it applies to ``text``); pass ``None`` to disable that check.
    """

    def policy(request: ApprovalRequest) -> bool:
        if request.is_hard_forget:
            return True
        content = request.arguments.get("content")
        if request.tool_name == INGEST_TOOL:
            content = request.arguments.get("text")
        if (max_content_chars is not None and isinstance(content, str)
                and len(content) > max_content_chars):
            return True
        if request.tool_name == "anatid_remember":
            return not remember
        if request.tool_name == "anatid_relate":
            return not relate
        if request.tool_name == "anatid_forget":
            return not soft_forget
        if request.tool_name == "anatid_supersede":
            return not supersede
        if request.tool_name == INGEST_TOOL:
            return not ingest
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
        if request.is_dry_run:
            # A dry run proposes a patch and returns its diff; nothing reaches the database,
            # so there is nothing for a person to approve.  The real call is gated as a write.
            return False
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
    # memory_id goes out as a decimal string: 63 bits do not survive a JSON number, and this
    # is the value the model hands back to anatid_supersede, anatid_forget and
    # anatid_provenance.
    return {
        "memory_id": wire_id(memory.memory_id),
        "content": memory.content,
        "kind": memory.kind,
        "created_at": _iso(memory.created_at),
        "confidence": memory.confidence,
        "access_count": memory.access_count,
        "writer": memory.writer,
        "about": list(about),
    }


def _edge_json(edge: Any, *, src: str, dst: str) -> dict[str, Any]:
    # The names the model used go back beside the ids, so the result reads as "Ada -> Kestrel"
    # and the ids are still there for a later anatid_provenance or SQL.
    return {
        "edge_id": wire_id(edge.edge_id),
        "src": src,
        "dst": dst,
        "src_entity_id": wire_id(edge.src),
        "dst_entity_id": wire_id(edge.dst),
        "rel_kind": edge.rel_kind,
        "valid_from": _iso(edge.valid_from),
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
    extractor: Extractor | None = None,
    ingest_review: ReviewHook | None = None,
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
        and ``anatid_recall`` runs the vector arm; without one the handle's own embedder
        (``Anatid.open(embedder=...)``) is used when there is one, else recall is BM25 + graph
        and says so in its result (``"arms"``).  The tools never call an embedding API
        themselves.
    ``extractor``
        An :class:`anatid.ingest.Extractor`.  Given one, the tenth tool ``anatid_ingest`` is
        built: a note in, a patch of facts, corrections and edges proposed by the extractor,
        resolved against the graph and applied in one transaction, gated like every write.
    ``ingest_review``
        Optional hook that sees the prepared patch before ``anatid_ingest`` applies it and
        returns the patch to apply, an edited copy, or ``None`` to decline (the tool then
        reports ``"declined": true`` and writes nothing).  This is the diff-level gate; the
        SDK's approval gate sees the note.
    ``include`` / ``exclude``
        Tool-name filters, e.g. ``exclude=["anatid_forget"]`` for an agent that may not delete.

    Returns the ``FunctionTool`` list, in the order of :data:`TOOL_NAMES`: nine tools, six of
    them writes (``anatid_remember``, ``anatid_relate``, ``anatid_supersede``,
    ``anatid_correct``, ``anatid_unrelate``, ``anatid_forget``) and three reads
    (``anatid_recall``, ``anatid_context``, ``anatid_provenance``), then ``anatid_ingest`` when
    an ``extractor`` was given.
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

    available = TOOL_NAMES + ((INGEST_TOOL,) if extractor is not None else ())
    wanted = set(include) if include is not None else set(available)
    wanted -= set(exclude or ())
    unknown = wanted - set(TOOL_NAMES) - {INGEST_TOOL}
    if unknown:
        raise ValueError(f"unknown tool name(s): {sorted(unknown)}")
    if INGEST_TOOL in wanted and extractor is None:
        raise ValueError(f"{INGEST_TOOL} needs extractor=: an anatid.ingest.Extractor")

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
        memory_id: WireId,
        content: str,
        entities: list[str] | None = None,
        kind: str | None = None,
    ) -> str:
        """Replace a memory that is now wrong with a corrected version.

        The old memory is kept and marked as superseded, so the correction is auditable and
        anatid_provenance can walk back to what was believed before.

        Args:
            memory_id: The id of the memory being corrected, as the decimal string anatid
                gave you, e.g. "883768514279557120". anatid ids are 63-bit and a JSON
                number loses precision above 2**53, so ids travel as strings.
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
            return _ok({"superseded": wire_id(int(memory_id)), **_memory_json(memory)})
        except AnatidError as exc:
            return _err(exc)

    @function_tool(name_override="anatid_forget", needs_approval=gate("anatid_forget"))
    def anatid_forget(
        ctx: RunContextWrapper[Any],
        memory_id: WireId,
        hard: bool | None = None,
        reason: str | None = None,
    ) -> str:
        """Stop believing a memory.

        Args:
            memory_id: The id of the memory to forget, as the decimal string anatid gave
                you, e.g. "883768514279557120". anatid ids are 63-bit and a JSON
                number loses precision above 2**53, so ids travel as strings.
            hard: False (the default) closes the memory so it stops being recalled but stays in
                history. True permanently erases the row, its edges, its embedding and its
                provenance -- use only for an actual erasure request.
            reason: Why, recorded in the audit trail.
        """
        try:
            receipt = db.forget(int(memory_id), hard=bool(hard), reason=reason,
                                writer=writer_for(ctx), tenant=ns)
            return _ok({
                "memory_id": wire_id(receipt.memory_id),
                "hard": receipt.hard,
                "at": _iso(receipt.at),
                "rows_removed": receipt.rows_removed,
                "reason": receipt.reason,
            })
        except AnatidError as exc:
            return _err(exc)

    # ---------------------------------------------------------------- graph tools

    @function_tool(name_override="anatid_relate", needs_approval=gate("anatid_relate"))
    def anatid_relate(
        ctx: RunContextWrapper[Any],
        src: str,
        dst: str,
        rel_kind: str | None = None,
    ) -> str:
        """Connect two entities so that facts about one become reachable from the other.

        Memories are filed under the entities they name, and graph recall walks edges between
        entities, up to two hops from a seed. Without an edge, "Ada leads Kestrel" and
        "Kestrel owns the ingest service" are two islands: a recall seeded on Ada finds the
        first and nothing else. Add an edge when two entities are connected in the world (a
        person and their team, a team and the service it owns, a project and its owner), so
        that a walk from either one reaches the facts about the other. One edge serves both
        directions. Edges are not deduplicated, so check anatid_context(entity, hops=1) before
        adding one that may already exist.

        Args:
            src: One entity's name, e.g. "Ada". Created if it does not exist yet.
            dst: The other entity's name, e.g. "Kestrel". Created if it does not exist yet.
            rel_kind: Optional label for the edge, e.g. "leads", "owns", "maintains".
        """
        try:
            edge = db.relate(src, dst, rel_kind=rel_kind, writer=writer_for(ctx), tenant=ns)
            return _ok({"related": True, **_edge_json(edge, src=src, dst=dst)})
        except AnatidError as exc:
            return _err(exc)

    @function_tool(name_override="anatid_unrelate", needs_approval=gate("anatid_unrelate"))
    def anatid_unrelate(
        ctx: RunContextWrapper[Any],
        src: str,
        dst: str,
        rel_kind: str | None = None,
    ) -> str:
        """Retire the connection between two entities.

        Use it when a relationship has ended in the world: Bo no longer maintains the ingest
        service, Ada has left Kestrel. The edge is closed and stays in history: recall stops
        walking it from now on, and a read as of an earlier time still does. Retire an edge
        only when the relationship itself has ended. A fact that changed while the two
        entities stay connected calls for anatid_supersede or anatid_correct instead.

        Args:
            src: One entity's name. It must already exist.
            dst: The other entity's name. It must already exist.
            rel_kind: Close only the edges with this label. Omit to close every edge between
                the two entities.
        """
        try:
            closed = db.unrelate(src, dst, rel_kind=rel_kind, tenant=ns)
            payload: dict[str, Any] = {
                "src": src, "dst": dst, "rel_kind": rel_kind, "edges_closed": closed,
            }
            if not closed:
                payload["note"] = "no current edge between these two entities; nothing changed"
            return _ok(payload)
        except AnatidError as exc:
            return _err(exc)

    @function_tool(name_override="anatid_correct", needs_approval=gate("anatid_correct"))
    def anatid_correct(
        ctx: RunContextWrapper[Any],
        memory_id: WireId,
        content: str,
        entities: list[str] | None = None,
        kind: str | None = None,
        add_relations: list[Relation] | None = None,
        remove_relations: list[Relation] | None = None,
    ) -> str:
        """Replace a wrong memory and move the edges that change with it, as one unit.

        A correction is rarely only a sentence. When "Bo maintains the ingest service"
        becomes "Cy maintains the ingest service", the edge from Bo to the service has to
        close and an edge from Cy has to open, or recall keeps walking to Bo. Name the edges
        to close in remove_relations and the edges to open in add_relations. anatid runs the
        supersede, the closes and the opens in one transaction, so all of it lands or none
        of it does. Use this whenever a correction changes who is connected to what; use
        anatid_supersede when only the wording or a detail of the fact changes.

        Args:
            memory_id: The id of the memory being corrected, as the decimal string anatid
                gave you, e.g. "883768514279557120". anatid ids are 63-bit and a JSON
                number loses precision above 2**53, so ids travel as strings.
            content: The corrected fact, as one self-contained sentence.
            entities: Names the corrected fact is about. Omit to keep the old memory's entities.
            kind: Label for the new memory. Omit to keep the old one's.
            add_relations: Edges to open, each with src and dst entity names and an optional
                rel_kind label. Missing entities are created.
            remove_relations: Edges to close, in the same shape. Each entry closes every
                current edge between its two entities, or only those with its rel_kind when
                one is given. Both entities must already exist.
        """
        try:
            additions = [_relation_tuple(r) for r in add_relations or ()]
            removals = [_relation_tuple(r) for r in remove_relations or ()]
            receipt = db.correct(
                int(memory_id),
                content,
                entities=None if entities is None else tuple(entities),
                kind=kind,
                add_relations=additions,
                remove_relations=removals,
                embedding=list(embedder(content)) if embedder is not None else None,
                writer=writer_for(ctx),
                tenant=ns,
            )
            about = [e.name for e in db.entities_of(receipt.new.memory_id, tenant=ns)]
            closed = set(receipt.closed)
            return _ok({
                "corrected": True,
                "superseded": wire_id(int(memory_id)),
                **_memory_json(receipt.new, about=about),
                "opened": [_edge_json(edge, src=src, dst=dst)
                           for edge, (src, dst, _kind)
                           in zip(receipt.opened, additions, strict=True)],
                "closed": [{"src": src, "dst": dst, "rel_kind": rel_kind}
                           for src, dst, rel_kind in receipt.closed],
                "not_closed": [{"src": src, "dst": dst, "rel_kind": rel_kind}
                               for src, dst, rel_kind in removals
                               if (src, dst, rel_kind) not in closed],
                "edges_closed": receipt.edges_closed,
            })
        except AnatidError as exc:
            return _err(exc)

    # ---------------------------------------------------------------- ingestion

    @function_tool(name_override=INGEST_TOOL, needs_approval=gate(INGEST_TOOL))
    def anatid_ingest(
        ctx: RunContextWrapper[Any],
        text: str,
        source: str | None = None,
        dry_run: bool | None = None,
        patch: str | None = None,
    ) -> str:
        """Store what a note, message or document says, as one reviewed change to memory.

        The text and the facts already known about the entities it names go to an extraction
        model, which proposes a patch: facts to add, facts to correct (by memory_id), edges to
        open and close, and names that mean an existing entity. The patch is resolved against
        memory (duplicates dropped, aliases rewritten, each with a note) and applied in one
        transaction, with the text stored first as the episode every new row cites. Call with
        dry_run=true first to see the diff without writing; the result carries "patch_json",
        and passing it back as `patch` applies exactly that patch (edited if needed) instead
        of asking the model again.

        Args:
            text: The note, message or document, verbatim.
            source: Where it came from, e.g. "standup/2026-06-15" or a file name. Stored on
                the episode.
            dry_run: True proposes the patch and returns its diff without writing anything.
            patch: The "patch_json" string a dry run returned, possibly edited. When given,
                the model is not consulted and this patch is applied as is.
        """
        try:
            if extractor is None:  # pragma: no cover - the tool is not built without one
                raise ValueError(f"{INGEST_TOOL} needs an extractor")
            if not text or not text.strip():
                return _err(ValueError("anatid_ingest needs text: a note, a message or a document"))
            if patch:
                proposed = MemoryPatch.from_json(patch, source_text=text)
                how = "patch supplied by the caller"
            else:
                proposed = propose_patch(db, text, extractor=extractor, tenant=ns)
                how = "proposed by the extractor and resolved against memory"
            summary: dict[str, Any] = {
                "operations": proposed.operations,
                "is_empty": proposed.is_empty,
                "diff": proposed.describe(),
                "patch_json": proposed.to_json(indent=None),
                "notes": list(proposed.notes),
                "how": how,
            }
            if dry_run:
                return _ok({"applied": False, "dry_run": True, **summary})
            if ingest_review is not None:
                reviewed = ingest_review(proposed)
                if reviewed is None:
                    return _ok({"applied": False, "declined": True, **summary,
                                "note": "the review hook declined this patch; nothing was written"})
                if not isinstance(reviewed, MemoryPatch):
                    raise TypeError(
                        f"ingest_review must return a MemoryPatch or None, "
                        f"got {type(reviewed).__name__}"
                    )
                proposed = reviewed
                summary["diff"] = proposed.describe()
                summary["patch_json"] = proposed.to_json(indent=None)
                summary["operations"] = proposed.operations
            receipt = proposed.apply(
                db,
                writer=writer_for(ctx),
                episode=text,
                source=source,
                tenant=ns,
                embedder=embedder,
            )
            return _ok({
                "applied": True,
                **summary,
                "summary": receipt.describe(),
                "changes": receipt.changes,
                **receipt.to_dict(),
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

        Text search and a graph walk run together and are fused. Without seed_entity the
        walk starts from the entities the query names (longest name first, at most three),
        so "who maintains the ingest service" reaches facts filed under the ingest service
        and its neighbours even when their words differ; the result's "seeds" lists them.

        Args:
            query: What to look for, in natural language.
            k: How many memories to return. Defaults to 8.
            seed_entity: Optional entity name to start the graph walk from instead of the
                names found in the query, which pulls in memories that share entities with it
                even when the words do not match.
            hops: How far to walk from the seeds, 1 or 2. Defaults to 2.
        """
        try:
            notes: list[str] = []
            seed: Any = AUTO_SEED
            if seed_entity:
                try:
                    seed = db.entity_id(seed_entity, tenant=ns, create=False)
                except AnatidError:
                    notes.append(
                        f"unknown entity {seed_entity!r}; the graph arm was seeded from the "
                        f"query instead"
                    )
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
                "seeds": list(hits.seeds),
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
    def anatid_provenance(ctx: RunContextWrapper[Any], memory_id: WireId) -> str:
        """Explain where a memory came from: its correction chain, sources and writers.

        Args:
            memory_id: The id of the memory to explain, as the decimal string anatid gave
                you, e.g. "883768514279557120". anatid ids are 63-bit and a JSON
                number loses precision above 2**53, so ids travel as strings.
        """
        try:
            prov = db.provenance(int(memory_id), tenant=ns)
            return _ok({
                "memory_id": wire_id(prov.memory_id),
                "depth": prov.depth,
                "chain": [_memory_json(m) for m in prov.chain],
                "source_text": prov.source_text,
                "episodes": [{"episode_id": wire_id(e.episode_id), "source": e.source,
                              "content": e.content, "created_at": _iso(e.created_at)}
                             for e in prov.episodes],
                "writers": list(prov.writers),
            })
        except AnatidError as exc:
            return _err(exc)

    built = {
        "anatid_remember": anatid_remember,
        "anatid_relate": anatid_relate,
        "anatid_supersede": anatid_supersede,
        "anatid_correct": anatid_correct,
        "anatid_unrelate": anatid_unrelate,
        "anatid_forget": anatid_forget,
        "anatid_recall": anatid_recall,
        "anatid_context": anatid_context,
        "anatid_provenance": anatid_provenance,
        INGEST_TOOL: anatid_ingest,
    }
    return [built[name] for name in available if name in wanted]
