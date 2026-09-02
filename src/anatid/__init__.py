"""anatid -- an embedded graph database for AI agents, built on DuckDB.

::

    from anatid import Anatid

    with Anatid.open("agent.anatid", tenant=1, embedding_dim=1536) as db:
        db.relate("Ada", "coffee")
        m = db.remember("Ada prefers dark roast", entities=["Ada"], embedding=vec)
        db.rebuild_fts_index()                       # BM25 is not incremental; you say when
        hits = db.recall("coffee", embedding=q, seed_entity="Ada", k=5)
        newer = db.supersede(m.memory_id, "Ada switched to decaf")
        db.as_of(t0).recall_2hop("Ada")              # the answer before the supersession
        db.provenance(newer.memory_id).source_text   # the evidence behind the belief

What anatid promises, and what it does not
------------------------------------------
* **Time travel** is anatid's own filter over ``valid_from``/``valid_to`` and
  ``tx_from``/``tx_to``.  DuckDB has no ``AS OF SYSTEM TIME``.
* **Tenant isolation** is file-per-tenant, enforced by this wrapper and the filesystem.  DuckDB
  has no schema- or row-level access control; ``tenant_id`` inside one file is *scoping*, not
  isolation.
* **BM25** rides DuckDB's ``fts`` index, which is **not incremental**: rows written after the
  last :meth:`~anatid.Anatid.rebuild_fts_index` are invisible to the text arm, and every
  :meth:`~anatid.Anatid.recall` result says so.
* **Transactions** are DuckDB's optimistic MVCC: snapshot isolation, not serializable.  Appends
  never conflict; two updates to the same row abort the second with a retryable
  :class:`~anatid.errors.ConflictError`.
* **Vector search** is a brute-force cosine scan -- fine to roughly 1e5 memories per tenant.
* **Erasure** (``forget(hard=True)``) really erases: row, edges, embedding and provenance.

The engine choice is settled by measurement, not taste: at 1,000,000 memories / 2.3M edges /
10 tenants, 2-hop recall runs at 2.88 ms p50 on plain DuckDB SQL (2.04 ms with the optional C++
CSR extension) against a tuned LadybugDB's 7.35 ms, with identical result id-lists on 200 verify
queries.
"""

from __future__ import annotations

from .csr import CsrBackend, CsrInfo, discover_extension_path
from .database import Anatid, DatabasePool, connect
from .errors import (
    AnatidError,
    ConflictError,
    EmbeddingDimensionError,
    ExtensionUnavailable,
    NotFoundError,
    SchemaVersionError,
    StaleIndexError,
    TenantIsolationError,
)
from .ids import new_id
from .recall import BRUTE_FORCE_CEILING, FTS_STALENESS_POLICY, RRF_K
from .schema import (
    CONTRACT_NOTES,
    DEFAULT_EMBEDDING_DIM,
    SCHEMA_VERSION,
    SchemaConfig,
)
from .types import (
    ABOUT,
    CURRENT,
    RELATES_TO,
    SUPERSEDES,
    AsOf,
    Edge,
    EdgeType,
    Entity,
    Episode,
    ForgetReceipt,
    FtsStatus,
    Isolation,
    Memory,
    Namespace,
    Provenance,
    PruneReport,
    RecallHit,
    RecallHits,
    SchemaInfo,
    to_utc_naive,
    utcnow,
)
from .verbs import AsOfView, as_of

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    # handles
    "Anatid",
    "DatabasePool",
    "connect",
    # value types
    "Memory",
    "Entity",
    "Edge",
    "EdgeType",
    "Episode",
    "RecallHit",
    "RecallHits",
    "Provenance",
    "ForgetReceipt",
    "PruneReport",
    "FtsStatus",
    "SchemaInfo",
    "Namespace",
    "Isolation",
    "AsOf",
    "AsOfView",
    "CURRENT",
    "ABOUT",
    "RELATES_TO",
    "SUPERSEDES",
    # helpers
    "as_of",
    "utcnow",
    "to_utc_naive",
    "new_id",
    # schema
    "SchemaConfig",
    "SCHEMA_VERSION",
    "DEFAULT_EMBEDDING_DIM",
    "CONTRACT_NOTES",
    # graph backend
    "CsrBackend",
    "CsrInfo",
    "discover_extension_path",
    # constants
    "RRF_K",
    "BRUTE_FORCE_CEILING",
    "FTS_STALENESS_POLICY",
    # errors
    "AnatidError",
    "SchemaVersionError",
    "ConflictError",
    "TenantIsolationError",
    "ExtensionUnavailable",
    "NotFoundError",
    "EmbeddingDimensionError",
    "StaleIndexError",
]
