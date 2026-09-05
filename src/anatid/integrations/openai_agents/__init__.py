"""anatid for the OpenAI Agents SDK: one DuckDB file holding history, knowledge and approvals.

Three pieces, each usable on its own:

``AnatidSession``
    The SDK's ``Session`` protocol (async ``get_items``/``add_items``/``pop_item``/
    ``clear_session``) backed by an anatid database, so conversation turns are queryable rows in
    the same file as the memory graph -- with DuckDB analytics and history-to-knowledge joins the
    SQLite session cannot express.  See :mod:`~anatid.integrations.openai_agents.session`.

``create_memory_tools``
    Nine function tools -- ``anatid_remember``, ``anatid_relate``, ``anatid_recall``,
    ``anatid_context``, ``anatid_supersede``, ``anatid_correct``, ``anatid_unrelate``,
    ``anatid_forget``, ``anatid_provenance`` -- and a tenth, ``anatid_ingest``, when an
    extractor is given.  Reads run freely; the writes carry ``needs_approval``, driven by a
    policy you choose.  The three graph tools let an agent maintain the edges recall walks:
    connect two entities, retire a connection, and correct a memory together with the edges
    that change with it, in one transaction.  ``anatid_ingest`` takes a note and applies the
    patch of facts and edges a model proposes from it, as one reviewed transaction.  See
    :mod:`~anatid.integrations.openai_agents.tools`.

``RunStateStore``
    ``RunState.to_string()`` parked in the same anatid file, so an interrupted run can be
    approved minutes or days later by another process.  See
    :mod:`~anatid.integrations.openai_agents.approvals`.

.. code-block:: python

    from anatid import Anatid
    from anatid.integrations.openai_agents import (
        AnatidSession, RunStateStore, create_memory_tools,
    )

    db = Anatid.open("agent.anatid", tenant=1, embedding_dim=1536)
    session = AnatidSession("conv-1", db)
    agent = Agent(name="assistant", tools=create_memory_tools(db, session=session))
    store = RunStateStore(db)

Importing this package requires ``openai-agents`` only for :func:`create_memory_tools` and
``RunStateStore.resume``; :class:`AnatidSession` works without it.
"""

from __future__ import annotations

from .approvals import RUN_STATE_COLUMNS, RunStateStore, StoredRun
from .session import MESSAGE_COLUMNS, AnatidSession, classify_item
from .tools import (
    INGEST_TOOL,
    READ_TOOLS,
    TOOL_NAMES,
    WRITE_TOOLS,
    ApprovalPolicy,
    ApprovalRequest,
    Relation,
    always_require_approval,
    approve_low_risk,
    create_memory_tools,
    never_require_approval,
)

__all__ = [
    "AnatidSession",
    "MESSAGE_COLUMNS",
    "classify_item",
    "create_memory_tools",
    "Relation",
    "ApprovalRequest",
    "ApprovalPolicy",
    "always_require_approval",
    "never_require_approval",
    "approve_low_risk",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "TOOL_NAMES",
    "INGEST_TOOL",
    "RunStateStore",
    "StoredRun",
    "RUN_STATE_COLUMNS",
]
