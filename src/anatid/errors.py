"""Exception hierarchy for anatid.

Everything anatid raises on purpose derives from :class:`AnatidError`.  Errors coming straight
out of DuckDB are left alone except for the two cases anatid gives a stable meaning to:

* a write-write conflict from DuckDB's optimistic MVCC -> :class:`ConflictError` (retryable),
* a cross-tenant access on a file-per-tenant handle -> :class:`TenantIsolationError`.
"""

from __future__ import annotations

__all__ = [
    "AnatidError",
    "SchemaVersionError",
    "ConflictError",
    "TenantIsolationError",
    "ExtensionUnavailable",
    "NotFoundError",
    "EmbeddingDimensionError",
    "StaleIndexError",
]


class AnatidError(Exception):
    """Base class for every error anatid raises deliberately."""


class SchemaVersionError(AnatidError):
    """The database file's ``anatid_meta.schema_version`` cannot be used by this build.

    Raised when the file was written by a NEWER anatid than the one running (anatid never
    downgrades a file), or when a requested migration step does not exist.
    """

    def __init__(self, message: str, *, found: int | None = None, expected: int | None = None) -> None:
        super().__init__(message)
        self.found = found
        self.expected = expected


class ConflictError(AnatidError):
    """DuckDB's optimistic MVCC aborted this transaction because another transaction touched
    the same row first.

    DuckDB detects write-write conflicts on the SAME row: the loser sees
    ``TransactionContext Error: Conflict on update!`` (raised at statement time, not at COMMIT).
    Appends never conflict.  The transaction is dead at this point -- roll back and retry the
    whole unit of work.  This error is retryable by construction; nothing has been committed.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    retryable = True


class TenantIsolationError(AnatidError):
    """A tenant boundary that anatid's wrapper enforces was crossed.

    DuckDB itself has NO schema-level or row-level access control, so this is an application
    level check: a handle opened for one tenant file refuses to read or write another tenant's
    rows.  See :class:`anatid.database.Anatid` for the full isolation contract.
    """


class ExtensionUnavailable(AnatidError):
    """The optional C++ ``anatid`` DuckDB extension could not be loaded.

    Only raised when the caller explicitly demanded the extension path
    (``require_extension=True``).  By default anatid falls back to the pure-SQL expansion,
    which returns identical results.
    """


class NotFoundError(AnatidError):
    """A memory / entity / episode id does not exist in this database (or this tenant)."""


class EmbeddingDimensionError(AnatidError):
    """An embedding was supplied whose length is not the database's configured dimension."""

    def __init__(self, message: str, *, expected: int | None = None, got: int | None = None) -> None:
        super().__init__(message)
        self.expected = expected
        self.got = got


class StaleIndexError(AnatidError):
    """The BM25 (fts) index is stale and the caller asked for it to be an error.

    DuckDB's ``fts`` extension index is NOT incremental: rows inserted after
    ``PRAGMA create_fts_index`` are invisible to BM25 until the index is rebuilt.  By default
    anatid reports staleness on the result object instead of raising; pass
    ``on_stale_fts="error"`` to :meth:`anatid.Anatid.recall` to get this exception.
    """
