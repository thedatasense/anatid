"""Conflict primitives: re-running a unit of work, and compare-and-swap on a version.

DuckDB gives optimistic snapshot isolation and aborts a transaction that loses a write-write
race.  It does not give serializable application invariants, so anatid does not claim them.
What it offers instead is in this module and is deliberately small:

* :func:`run` (:meth:`anatid.Anatid.atomic`) re-runs the **whole** caller-supplied callback in a
  fresh transaction after a retryable :class:`~anatid.errors.ConflictError`, with jittered
  backoff.  Re-running one statement inside an aborted transaction is meaningless: DuckDB marks
  the transaction dead at the first conflict, so nothing more can be read or written on it.
* :func:`version_of` and :func:`version_conflict` are the compare-and-swap half.  A caller that
  read version 7 asks to write only while the row is still at version 7; when it is not, the
  error says which version it found instead, and says the write must not simply be retried.
* :func:`current_ids` is the endpoint guard behind ``relate(..., if_current=True)``: the
  endpoints are checked in the transaction that writes the edge, which is snapshot isolation
  and not serializability, and the verb's docstring says exactly that.

What is retryable and what is not
---------------------------------
:class:`~anatid.errors.ConflictError` carries ``retryable``.  It is True for an engine-level
abort, because nothing was committed and the next attempt re-reads and usually wins.  It is
False for a compare-and-swap failure, because the version the caller expected is gone for good
and the same callback would fail the same way on every attempt: that one belongs to the caller,
who has to re-read and decide whether the change still applies.  :func:`run` retries the first
kind and re-raises the second untouched.

Nothing here retries a non-conflict error.  A ``ValidationError``, a ``NotFoundError`` or a
DuckDB planner error means the unit of work was wrong, and running it again keeps it wrong.
"""

from __future__ import annotations

import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, TypeVar

from .errors import ConflictError
from .schema import quote_ident, version_expr
from .verbs import _is_entity_race
from .visibility import Visibility

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._typing import VerbHost

log = logging.getLogger("anatid.atomic")

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_BACKOFF",
    "DEFAULT_MAX_BACKOFF",
    "Attempt",
    "AtomicOutcome",
    "atomic",
    "run",
    "is_retryable",
    "backoff_delay",
    "version_of",
    "version_conflict",
    "current_ids",
    "missing_or_closed",
]

_T = TypeVar("_T")

#: How many times :func:`run` runs the callback before giving up.  Three is the design's
#: number: a conflict that survives two re-reads is contention the caller should see.
DEFAULT_MAX_ATTEMPTS = 3

#: First backoff step in seconds; each further attempt doubles it, capped at
#: :data:`DEFAULT_MAX_BACKOFF`.  The actual sleep is uniform in ``[0, step)`` (full jitter), so
#: writers that collided do not wake together and collide again.
DEFAULT_BACKOFF = 0.005
DEFAULT_MAX_BACKOFF = 0.2


@dataclass(frozen=True, slots=True)
class Attempt:
    """One run of an :func:`run` callback, passed in when the callback accepts an argument.

    ``number`` counts from 1.  ``last_error`` is the conflict that ended the previous attempt,
    so a callback can log what it is re-running after, or take a different path on a retry.
    """

    number: int
    max_attempts: int
    last_error: BaseException | None = None

    @property
    def first(self) -> bool:
        return self.number == 1

    @property
    def final(self) -> bool:
        """True on the last attempt this call will make."""
        return self.number >= self.max_attempts


@dataclass(frozen=True, slots=True)
class AtomicOutcome:
    """What :func:`run` did: the callback's result and how many attempts it took.

    ``conflicts`` holds the errors that caused each re-run, oldest first, so a caller can
    report contention without instrumenting the callback.
    """

    result: Any
    attempts: int
    conflicts: tuple[BaseException, ...] = field(default_factory=tuple)

    @property
    def retried(self) -> bool:
        return self.attempts > 1


def is_retryable(exc: BaseException) -> bool:
    """True when ``exc`` is a conflict that re-running the unit of work can resolve.

    Two things qualify.  A :class:`~anatid.errors.ConflictError` whose ``retryable`` flag is
    set, which is the engine's write-write abort: nothing was committed and the next attempt
    re-reads.  And a lost entity-creation race
    (:func:`anatid.verbs._is_entity_race`), where ``UNIQUE (tenant_id, entity_key)`` refused a
    second row for a name another writer committed first, at the INSERT or at the COMMIT; the
    verbs already re-run their own transaction on that one, and a callback that opened its own
    has to be re-run here instead.

    A compare-and-swap failure is a conflict that no number of identical attempts can win, so
    it does not qualify.  Neither does anything that is not a conflict at all.
    """
    if isinstance(exc, ConflictError):
        return bool(getattr(exc, "retryable", True))
    return _is_entity_race(exc)


def backoff_delay(
    attempt: int,
    *,
    base: float = DEFAULT_BACKOFF,
    cap: float = DEFAULT_MAX_BACKOFF,
    rng: Callable[[], float] = random.random,
) -> float:
    """Full-jitter exponential backoff: uniform in ``[0, min(cap, base * 2**(attempt-1)))``.

    Jitter is not decoration.  Two writers that abort at the same instant and sleep the same
    fixed interval collide again on the next attempt; drawing the wait uniformly spreads them.
    """
    if attempt < 1:
        raise ValueError("attempt counts from 1")
    step = min(float(cap), float(base) * (2 ** (int(attempt) - 1)))
    if step <= 0:
        return 0.0
    return float(rng()) * step


def _caller(callback: Callable[..., _T]) -> Callable[[Attempt], _T]:
    """Decide once whether ``callback`` takes the :class:`Attempt`, and bind the call shape.

    A callback that can be called with no arguments is called with none; anything else is
    passed the attempt.  The decision is made before the first run so a signature that fits
    neither fails on attempt 1 with its own ``TypeError`` rather than after a retry.
    """
    if not callable(callback):
        raise TypeError(f"atomic() needs a callable, got {type(callback).__name__}")
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):  # builtins and C callables have no signature
        return lambda _attempt: callback()
    try:
        sig.bind()
    except TypeError:
        return lambda attempt: callback(attempt)
    return lambda _attempt: callback()


def run(
    db: "VerbHost",
    callback: Callable[..., _T],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    sleep: Callable[[float], Any] | None = None,
    rng: Callable[[], float] | None = None,
) -> AtomicOutcome:
    """Run ``callback`` inside one transaction, re-running it on a retryable conflict.

    Returns an :class:`AtomicOutcome`; :func:`atomic` is the same thing returning only the
    result.  Each attempt opens its own transaction through ``db.transaction()``, so a
    conflict rolls the whole attempt back before the next one starts and the callback never
    sees half of its own previous run.

    When the caller already has a transaction open this steps aside and runs the callback once
    inside it.  Only the caller can decide to re-run the caller's transaction: anatid does not
    know what else went into it, and DuckDB has no usable savepoint to roll back to.

    What counts as a conflict is :func:`is_retryable`: the engine's write-write abort and the
    lost entity-creation race.  Everything else propagates on the first attempt.

    ``sleep`` and ``rng`` exist so a test can make the backoff deterministic.
    """
    attempts = int(max_attempts)
    if attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if backoff < 0 or max_backoff < 0:
        raise ValueError("backoff and max_backoff must be >= 0")
    call = _caller(callback)
    if getattr(db, "in_transaction", False):
        return AtomicOutcome(call(Attempt(1, attempts)), 1, ())
    nap = time.sleep if sleep is None else sleep
    dice = random.random if rng is None else rng
    conflicts: list[BaseException] = []
    for number in range(1, attempts + 1):
        attempt = Attempt(number, attempts, conflicts[-1] if conflicts else None)
        try:
            with db.transaction():
                result = call(attempt)
        except Exception as exc:
            if isinstance(exc, ConflictError):
                # The attempt number rides on the error the caller finally sees.  Only
                # anatid's own error carries it: a raw DuckDB exception is not ours to
                # decorate.
                exc.attempt = number
            if not is_retryable(exc) or number == attempts:
                raise
            conflicts.append(exc)
            delay = backoff_delay(number, base=backoff, cap=max_backoff, rng=dice)
            log.debug(
                "conflict on attempt %d/%d, re-running after %.4fs: %s",
                number,
                attempts,
                delay,
                exc,
            )
            if delay:
                nap(delay)
            continue
        return AtomicOutcome(result, number, tuple(conflicts))
    raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover


def atomic(db: "VerbHost", callback: Callable[..., _T], **kwargs: Any) -> _T:
    """:func:`run` returning the callback's result.  See :meth:`anatid.Anatid.atomic`."""
    return run(db, callback, **kwargs).result


def version_of(
    db: "VerbHost",
    table: str,
    id_column: str,
    row_id: int,
    *,
    tenant_id: int,
) -> int | None:
    """The version number of a logical id's **live** row, or None when there is none.

    The live row is the one no correction has closed on the transaction axis, whatever its
    valid interval, so a superseded memory still has one and its version is the number a
    compare-and-swap has to match.  ``None`` means the id does not exist in this tenant (or
    every version of it was purged).
    """
    live, params = Visibility(int(tenant_id)).live()
    # version_expr, not _VERSION: a handle that could not migrate the file (read_only, or
    # ensure=False) may be looking at a schema-v3 table with no version column, where every
    # row is version 1.
    row = db.execute(
        f"SELECT {version_expr(db.connection, table)} FROM {quote_ident(table)} "
        f"WHERE {quote_ident(id_column)} = ? AND {live}",
        [int(row_id)] + params,
    ).fetchone()
    return None if row is None else int(row[0])


def version_conflict(
    resource: str,
    *,
    expected_version: int | None,
    current_version: int | None,
    action: str = "write",
    retryable: bool = False,
) -> ConflictError:
    """The structured error a failed compare-and-swap raises.

    Not retryable by default, and that is the point: the caller asked to write only while the
    row was at a version that has since been replaced, so an identical retry fails identically.
    Re-read the row, decide whether the change still applies, and write again with the version
    that is there now.
    """
    return ConflictError(
        f"{resource} is at version {current_version} but the {action} expected version "
        f"{expected_version}; another writer corrected it first. Re-read it, decide whether "
        f"the change still applies, and write again against the version you found.",
        resource=resource,
        expected_version=expected_version,
        current_version=current_version,
        retryable=retryable,
    )


def current_ids(
    db: "VerbHost",
    table: str,
    id_column: str,
    ids: Iterable[int],
    *,
    tenant_id: int,
) -> set[int]:
    """Which of ``ids`` have a currently-valid, live row in this tenant.

    The guard behind ``relate(..., if_current=True)``.  It runs inside the transaction that
    writes the edge, so it sees that transaction's own writes and one consistent snapshot; it
    is snapshot isolation, not serializability, and the docstring of
    :meth:`anatid.Anatid.relate` says so.

    The id list is interpolated after ``int()`` rather than bound: DuckDB does not take a list
    parameter for ``IN``, and only digits can reach the statement.  Same narrow exception the
    verbs document.
    """
    wanted = [int(i) for i in ids]
    if not wanted:
        return set()
    where, params = Visibility(int(tenant_id)).predicate()
    rows = db.execute(
        f"SELECT {quote_ident(id_column)} FROM {quote_ident(table)} "
        f"WHERE {quote_ident(id_column)} IN ({', '.join(str(i) for i in wanted)}) AND {where}",
        params,
    ).fetchall()
    return {int(r[0]) for r in rows}


def missing_or_closed(found: set[int], wanted: Sequence[int]) -> list[int]:
    """The ids of ``wanted`` that :func:`current_ids` did not find, in the order given."""
    seen: set[int] = set()
    out: list[int] = []
    for i in wanted:
        n = int(i)
        if n not in found and n not in seen:
            seen.add(n)
            out.append(n)
    return out
