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
    "BackupDestinationExists",
    "ExtensionUnavailable",
    "NotFoundError",
    "ValidationError",
    "RangeError",
    "EmbeddingDimensionError",
    "EmbeddingValueError",
    "DuplicateIdError",
    "IntegrityError",
    "StaleIndexError",
    "BruteForceCeilingError",
    "IndexGenerationError",
    "IndexValidationError",
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
    """Two writers wanted the same row and one of them has to be told.

    Two shapes reach this class, and ``retryable`` is what tells them apart.

    The **engine** shape: DuckDB's optimistic MVCC aborted this transaction because another
    transaction touched the same row first.  DuckDB detects write-write conflicts on the SAME
    row: the loser sees ``TransactionContext Error: Conflict on update!`` (raised at statement
    time, not at COMMIT).  Appends never conflict.  The transaction is dead at this point --
    roll back and retry the whole unit of work.  ``retryable`` is True, because nothing was
    committed and the next attempt re-reads.

    The **compare-and-swap** shape: the caller asked to write only while the row was at the
    version it had read (``memory.update(..., expected_version=7)``,
    ``relate(..., if_current=True)``), and it is not at that version any more.  ``resource``
    names the row, ``expected_version`` is what the caller held and ``current_version`` what
    the database has.  ``retryable`` is False: the version the caller reasoned about is gone,
    so an identical retry fails identically.  Re-read, decide whether the change still applies,
    and write again against what is there now.

    ``attempt`` is set by :func:`anatid.atomic.run` (:meth:`anatid.Anatid.atomic`) to the
    attempt number that raised, so a caller that catches the error out of a retry loop can
    report how hard anatid tried.  It is None for an error that never went through one.
    """

    #: Class-level default so ``ConflictError.retryable`` is answerable without an instance,
    #: and so code written against 0.1.1 (where it was only a class attribute) keeps working.
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        resource: str | None = None,
        expected_version: int | None = None,
        current_version: int | None = None,
        retryable: bool | None = None,
        attempt: int | None = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.resource = resource
        self.expected_version = expected_version
        self.current_version = current_version
        self.attempt = attempt
        if retryable is not None:
            self.retryable = bool(retryable)


class TenantIsolationError(AnatidError):
    """A tenant boundary that anatid's wrapper enforces was crossed.

    DuckDB itself has NO schema-level or row-level access control, so this is an application
    level check: a handle opened for one tenant file refuses to read or write another tenant's
    rows.  See :class:`anatid.database.Anatid` for the full isolation contract.
    """


class BackupDestinationExists(AnatidError, FileExistsError):
    """A backup was asked to write over a file that is already there, without ``overwrite``.

    It is a ``FileExistsError`` so ordinary filesystem-shaped handling still catches it, and an
    :class:`AnatidError` so a caller with its own vocabulary can tell it apart.  That second half
    is the reason the class exists: a command line catching ``OSError`` reported this as a
    run-time failure when it is a bad argument with nothing written, and an HTTP layer that
    recognises anatid's errors answered 500 where 400 was the honest answer.

    ``path`` is the destination that already exists.
    """

    def __init__(self, message: str, *, path: object = None) -> None:
        super().__init__(message)
        self.path = path


class ExtensionUnavailable(AnatidError):
    """The optional C++ ``anatid`` DuckDB extension could not be loaded.

    Only raised when the caller explicitly demanded the extension path
    (``require_extension=True``).  By default anatid falls back to the pure-SQL expansion,
    which returns identical results.
    """


class NotFoundError(AnatidError):
    """A memory / entity / episode id does not exist in this database (or this tenant)."""


class ValidationError(AnatidError, ValueError):
    """A verb was called with an argument anatid refuses to write.

    These are *caller* errors caught at the verb boundary, before any statement runs: a
    non-finite embedding value, a confidence outside ``[0, 1]``, a ``k`` of zero, a
    ``memory_id`` that already exists in the tenant.  anatid 0.1.0 accepted all of those and
    wrote the bad row; this class is what it raises instead.

    It is also a :class:`ValueError`, so code written against 0.1.0 that catches ``ValueError``
    around a verb keeps working.  Nothing here is retryable -- fix the argument.
    """

    retryable = False


class RangeError(ValidationError):
    """A numeric argument is outside the range anatid accepts.

    ``field`` names the argument, ``value`` is what was passed, and ``low``/``high`` are the
    inclusive bounds (``None`` = unbounded on that side).
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: object = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.value = value
        self.low = low
        self.high = high


class EmbeddingDimensionError(ValidationError):
    """An embedding was supplied whose length is not the database's configured dimension."""

    def __init__(self, message: str, *, expected: int | None = None, got: int | None = None) -> None:
        super().__init__(message)
        self.expected = expected
        self.got = got


class EmbeddingValueError(ValidationError):
    """An embedding contains a value that is not a finite number (NaN or +/-inf).

    DuckDB stores them happily and ``array_cosine_similarity`` then returns NaN for *every*
    comparison against that row, which sorts unpredictably and can silently displace real hits
    from the vector arm.  ``index`` is the offending position and ``value`` what was there.
    """

    def __init__(self, message: str, *, index: int | None = None, value: object = None) -> None:
        super().__init__(message)
        self.index = index
        self.value = value


class DuplicateIdError(ValidationError):
    """An explicit id passed to a verb is already in use in that tenant.

    ``memory_id`` is unique per *tenant*, not per file: the same id in two tenants is legal and
    the tests rely on it.  Two rows with one id inside one tenant are not: ``get()`` returns an
    arbitrary one of them, ``supersede`` closes both, and the BM25 source table has to
    de-duplicate them.  Passing ``memory_id=`` an id that already exists raises this.
    """

    def __init__(self, message: str, *, table: str | None = None, id: int | None = None,
                 tenant_id: int | None = None) -> None:
        super().__init__(message)
        self.table = table
        self.id = id
        self.tenant_id = tenant_id


class StaleIndexError(AnatidError):
    """The BM25 (fts) arm could not answer exactly and the caller asked for that to raise.

    On the derived text index that :meth:`anatid.Anatid.open` attaches by default, a write is
    searchable by the very next ``recall()`` with nothing rebuilt, so this is raised only when
    no generation was usable AND the tenant's corpus is above :data:`anatid.fts.SCAN_CEILING`,
    which made the exact fallback scan too expensive to run.  On 0.1.1's file-wide index
    (``accelerators=False``) it means what it always meant: rows inserted after
    ``PRAGMA create_fts_index`` are invisible to BM25 until the index is rebuilt.  Either way
    anatid reports the condition on the result object by default; pass ``on_stale_fts="error"``
    to :meth:`anatid.Anatid.recall` to get this exception, and read the message for which of the
    two it was.
    """


class BruteForceCeilingError(AnatidError):
    """The vector arm was asked to scan more of a tenant's memories than it is cheap for.

    anatid has no ANN index: :func:`anatid.recall.vector_arm` is a full cosine scan of the
    tenant's visible embeddings, linear in their number.  :data:`anatid.BRUTE_FORCE_CEILING`
    (100,000) is where that stops being cheap -- roughly 9-11 ms per recall at the ceiling on
    the spike hardware -- and since 0.1.1 :meth:`anatid.Anatid.recall` refuses to run the arm
    past it rather than quietly getting slower with every write.  ``rows`` is how many rows the
    scan would have covered, ``ceiling`` the limit it exceeded.

    Pass ``allow_slow=True`` to run the scan anyway; leave ``embedding=`` out to answer from the
    text and graph arms alone; or shard the tenant into its own file
    (:class:`anatid.DatabasePool`) and keep each one under the ceiling.  Never retryable: the
    corpus does not shrink by asking again.
    """

    retryable = False

    def __init__(self, message: str, *, tenant_id: int | None = None, rows: int | None = None,
                 ceiling: int | None = None) -> None:
        super().__init__(message)
        self.tenant_id = tenant_id
        self.rows = rows
        self.ceiling = ceiling


class IndexGenerationError(AnatidError):
    """A derived-index generation was asked for a transition its state does not allow.

    Raised by :mod:`anatid.derived` when a generation that was never validated is published
    without ``force=True``, when a build is started while another build of the same index and
    tenant is in progress, when a published generation is retired, or when an index that has no
    implementation registered (:class:`anatid.derived.NullIndex`) is asked to build.  ``index``
    names the index and ``generation`` the generation number when one is involved.  Never
    retryable on its own: the state has to change first.
    """

    retryable = False

    def __init__(self, message: str, *, index: str | None = None,
                 generation: int | None = None) -> None:
        super().__init__(message)
        self.index = index
        self.generation = generation


class IndexValidationError(IndexGenerationError):
    """A derived-index generation failed validation against the SQL oracle.

    ``report`` is the :class:`anatid.derived.ValidationReport`.  Raised by
    :func:`anatid.derived.maintain` only when asked to (``raise_on_failure=True``); by default
    the failed generation is retired, the previously published one stays in service and the
    :class:`anatid.derived.MaintenanceReport` says what happened.
    """

    def __init__(self, message: str, *, report=None, index: str | None = None,
                 generation: int | None = None) -> None:
        super().__init__(message, index=index, generation=generation)
        self.report = report


class IntegrityError(AnatidError):
    """:meth:`anatid.Anatid.doctor` found faults and was asked to raise.

    ``report`` is the full :class:`~anatid.types.DoctorReport`; ``findings`` is the subset with
    ``severity == "error"``.  Never raised by ``doctor()`` in its default reporting mode.
    """

    def __init__(self, message: str, *, report=None, findings=()) -> None:
        super().__init__(message)
        self.report = report
        self.findings = tuple(findings)
