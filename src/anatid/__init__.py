"""anatid -- an embedded graph database for AI agents, built on DuckDB.

::

    from anatid import Anatid

    with Anatid.open("agent.anatid", tenant=1, embedding_dim=1536) as db:
        db.relate("Ada", "coffee")
        m = db.remember("Ada prefers dark roast", entities=["Ada"], embedding=vec)
        hits = db.recall("coffee", embedding=q, seed_entity="Ada", k=5)   # finds m, nothing built
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
* **BM25** rides DuckDB's ``fts`` index, which is not incremental, so :mod:`anatid.fts` runs
  the text arm as a derived index instead: a published base generation plus a journal written
  in the same transaction as the memory.  A write is **searchable by the very next**
  :meth:`~anatid.Anatid.recall` with nothing rebuilt, on the handle that wrote it and on any
  other handle on the file, and a ``supersede`` or a ``forget`` leaves the text results on that
  same read.  :meth:`~anatid.Anatid.rebuild_fts_index` compacts the journal into a new
  generation, which buys read latency rather than visibility.  ``Anatid.open()`` attaches this
  by default; ``accelerators=False`` keeps 0.1.1's single file-wide index, and there a write
  really is invisible until a rebuild.
* **Transactions** are DuckDB's optimistic MVCC: snapshot isolation, not serializable.  Appends
  never conflict; two updates to the same row abort the second with a retryable
  :class:`~anatid.errors.ConflictError`.
* **Vector search** is a brute-force cosine scan -- fine to roughly 1e5 memories per tenant,
  and :data:`BRUTE_FORCE_CEILING` is enforced: past it ``recall(embedding=...)`` raises
  :class:`~anatid.errors.BruteForceCeilingError` unless ``allow_slow=True``.
* **Erasure** (``forget(hard=True)``) really erases: row, edges, embedding and provenance.

The engine choice is settled by measurement, not taste: at 1,000,000 memories / 2.3M edges /
10 tenants, 2-hop recall runs at 2.88 ms p50 on plain DuckDB SQL (2.04 ms with the optional C++
CSR extension) against a tuned LadybugDB's 7.35 ms, with identical result id-lists on 200 verify
queries.
"""

from __future__ import annotations

from . import atomic, csr, fts, vector
from .atomic import Attempt, AtomicOutcome
from .csr import CsrBackend, CsrIndex, CsrInfo, ExpandPath, discover_extension_path
from .database import Anatid, DatabasePool, PoolEvent, connect
from .fts import FtsIndex, FtsSearch
from .derived import (
    DerivedIndex,
    ErasureResult,
    Generation,
    HealthReason,
    HealthReport,
    IndexDefinition,
    IndexEvent,
    IndexRegistry,
    MaintenancePolicy,
    MaintenanceReport,
    ValidationReport,
    maintain,
)
from .errors import (
    AnatidError,
    BruteForceCeilingError,
    ConflictError,
    DuplicateIdError,
    EmbeddingDimensionError,
    EmbeddingValueError,
    ExtensionUnavailable,
    IndexGenerationError,
    IndexValidationError,
    IntegrityError,
    NotFoundError,
    RangeError,
    SchemaVersionError,
    StaleIndexError,
    TenantIsolationError,
    ValidationError,
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
    DoctorFinding,
    DoctorReport,
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
    Severity,
    to_utc_naive,
    utcnow,
)
from .vector import VectorIndex, VectorSearch
from .verbs import AsOfView, as_of
from .visibility import Visibility, visible_at

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # handles
    "Anatid",
    "DatabasePool",
    "PoolEvent",
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
    "DoctorReport",
    "DoctorFinding",
    "Severity",
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
    # visibility
    "Visibility",
    "visible_at",
    # derived indexes
    "DerivedIndex",
    "ErasureResult",
    "Generation",
    "HealthReason",
    "HealthReport",
    "IndexDefinition",
    "IndexEvent",
    "IndexRegistry",
    "MaintenancePolicy",
    "MaintenanceReport",
    "ValidationReport",
    "maintain",
    # the accelerators, and the modules that configure them
    "fts",
    "FtsIndex",
    "FtsSearch",
    "vector",
    "VectorIndex",
    "VectorSearch",
    "csr",
    "CsrBackend",
    "CsrIndex",
    "CsrInfo",
    "ExpandPath",
    "discover_extension_path",
    # conflict handling
    "atomic",
    "Attempt",
    "AtomicOutcome",
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
    "ValidationError",
    "RangeError",
    "DuplicateIdError",
    "EmbeddingDimensionError",
    "EmbeddingValueError",
    "IntegrityError",
    "StaleIndexError",
    "BruteForceCeilingError",
    "IndexGenerationError",
    "IndexValidationError",
]
