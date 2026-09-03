"""Visibility: the one place anatid writes its tenant and time predicates.

DuckDB has no ``AS OF SYSTEM TIME`` and no row-level access control.  Time travel and tenant
scoping are therefore predicates anatid compiles into every read, and a read that forgets one
of them returns another tenant's rows or a row that was not visible at the requested instant.
This module exists so that no read path writes those predicates by hand: every read in
``anatid.recall``, ``anatid.csr``, ``anatid.verbs`` and ``anatid.database`` obtains them from
:class:`Visibility`, and ``tests/test_visibility.py`` scans the source tree for the literal
fragments and fails when one appears anywhere outside this file.

The rule, stated once
---------------------
A row of a bitemporal table is visible to :class:`Visibility` ``(tenant_id, valid_at, tx_at)``
when all of the following hold:

* ``row.tenant_id = tenant_id``;
* valid time: ``valid_at`` is ``None`` and ``row.valid_to IS NULL`` (the row is currently
  believed), or ``row.valid_from <= valid_at < row.valid_to`` with a ``NULL`` ``valid_to``
  meaning open;
* transaction time: ``tx_at`` is ``None`` and ``row.tx_to IS NULL`` (the row is live), or
  ``row.tx_from <= tx_at < row.tx_to`` with a ``NULL`` ``tx_to`` meaning open.

Intervals are half-open ``[from, to)``: a row closed exactly at T is already invisible at T, and
a row opened exactly at T is already visible.

Versions
--------
Rows of ``memories``, ``edges_about`` and ``edges_relates`` are immutable versions.  The logical
id (``memory_id``, ``edge_id``) is what callers hold; ``version`` numbers the physical rows of
one logical id from 1.  A correction (``supersede``, soft ``forget``, ``unrelate``, a
``reinforce`` that changes ``confidence``) never rewrites the row it corrects: it sets that
version's ``tx_to`` to the correction instant and inserts the next version, with ``tx_from`` at
the same instant and the corrected valid interval, in the same transaction.  The rule above
then selects at most one version of a logical id for any ``(valid_at, tx_at)``: the version
that was live at ``tx_at`` and believed at ``valid_at``.  Asking what the database believed
on Jan 2 about Jan 4, after a forget on Jan 3, returns the open-ended version 1; asking on
Jan 4 returns nothing, because version 2 ends its validity on Jan 3.

A lookup that has no time axis, ``get()`` by id, returns the **live** version: the one no
correction has closed on the transaction axis (:func:`live_row_sql`).  That version carries
the memory's current valid interval, so a superseded memory is returned with ``valid_to`` set
and ``is_current`` false.  The usage counters ``access_count`` and ``last_access_at`` are not
bitemporal: ``reinforce`` bumps them in place on the live version, and an older version keeps
the values it had when it was closed.

Derived indexes and this module
-------------------------------
An index built over current state cannot answer what was visible at an earlier instant, so an
accelerator only ever narrows the candidate set.  The candidate rows are then filtered with the
predicate from this module, or with :meth:`Visibility.admits` when the filtering happens in
Python.  See ``docs/design/derived-index-framework.md`` and :mod:`anatid.derived`.

The integer-literal fast path
-----------------------------
:meth:`Visibility.predicate` renders the tenant as a literal when asked
(``inline_tenant=True``).  The value has already been through ``int()``, and a Python ``int``
renders as digits and an optional minus sign, so nothing but a number can reach the statement.
duckdb-python 1.5.5 spends about 0.6 ms per call marshalling bound parameters, which is half the
cost of the benchmarked 2-hop recall at 100k memories, so the current-state 2-hop query runs
with no parameters at all.  Timestamps are always bound.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

from .types import CURRENT, AsOf, to_utc_naive

__all__ = [
    "Visibility",
    "visible_at",
    "temporal_predicate",
    "tenant_sql",
    "CURRENT_ROW_SQL",
    "LIVE_ROW_SQL",
    "current_row_sql",
    "live_row_sql",
]

#: The current-state predicate with no alias: the row is believed now and live now.  Used by
#: the ``relates_undirected`` view DDL, by the doctor's duplicate-live-edge check, and as the
#: write-side compare-and-swap guard: ``supersede``, soft ``forget`` and ``unrelate`` close
#: exactly the version this selects, and a matched-row count of zero is how they detect that
#: another writer closed it first.  The guard needs both halves: a version that a correction
#: has closed keeps ``valid_to IS NULL`` (it is immutable) and is told apart by its ``tx_to``.
CURRENT_ROW_SQL = "valid_to IS NULL AND tx_to IS NULL"

#: The live version of a logical object: the one no correction has closed on the transaction
#: axis.  No valid-time filter, so a superseded or forgotten memory still has exactly one live
#: version, whose ``valid_to`` says when belief in it ended.  Id lookups with no time axis
#: (``get()``) and the usage counters (``reinforce``) address this version.
LIVE_ROW_SQL = "tx_to IS NULL"


def _col(alias: str | None, name: str) -> str:
    return f"{alias}.{name}" if alias else name


def current_row_sql(alias: str | None = None) -> str:
    """:data:`CURRENT_ROW_SQL` qualified with ``alias`` (no bind parameters)."""
    return f"{_col(alias, 'valid_to')} IS NULL AND {_col(alias, 'tx_to')} IS NULL"


def live_row_sql(alias: str | None = None) -> str:
    """:data:`LIVE_ROW_SQL` qualified with ``alias`` (no bind parameters)."""
    return f"{_col(alias, 'tx_to')} IS NULL"


def _render_tenant(tenant_id: int, inline: bool) -> tuple[str, list]:
    return (str(int(tenant_id)), []) if inline else ("?", [int(tenant_id)])


def tenant_sql(alias: str | None = None) -> str:
    """``alias.tenant_id = ?`` with the tenant left as a bind parameter.

    For the lookups that identify a row by id inside one tenant and apply no time filter on
    purpose, over tables that carry no versions (entities, episodes, the fts sidecars).  The
    caller binds the tenant id itself.  The same lookup over a versioned table takes
    :meth:`Visibility.live` so it addresses one version; everything with a time axis uses
    :meth:`Visibility.predicate` instead.
    """
    return f"{_col(alias, 'tenant_id')} = ?"


@dataclass(frozen=True, slots=True)
class Visibility:
    """Which rows of one tenant a read may return, at which valid time and transaction time.

    ``valid_at`` / ``tx_at`` of ``None`` mean the current state on that axis.  Both ``None`` is
    the current state, and is what every read uses unless it was given an ``as_of``.

    The object is a value: build it with :meth:`at` from a verb's ``tenant`` and ``as_of``
    arguments and hand it to every statement the read issues.
    """

    tenant_id: int
    valid_at: _dt.datetime | None = None
    tx_at: _dt.datetime | None = None

    def __post_init__(self) -> None:
        if isinstance(self.tenant_id, bool):
            raise TypeError("tenant_id must be an int, not bool")
        object.__setattr__(self, "tenant_id", int(self.tenant_id))
        object.__setattr__(self, "valid_at", to_utc_naive(self.valid_at))
        object.__setattr__(self, "tx_at", to_utc_naive(self.tx_at))

    # ------------------------------------------------------------------ construction

    @classmethod
    def at(cls, tenant_id: int, as_of: AsOf | _dt.datetime | None = None) -> Visibility:
        """Build from a tenant id and anything :meth:`anatid.types.AsOf.coerce` accepts."""
        scope = AsOf.coerce(as_of)
        return cls(tenant_id=int(tenant_id), valid_at=scope.valid_time, tx_at=scope.tx_time)

    @property
    def as_of(self) -> AsOf:
        """The time half of this visibility as an :class:`~anatid.types.AsOf`."""
        if self.valid_at is None and self.tx_at is None:
            return CURRENT
        return AsOf(valid_time=self.valid_at, tx_time=self.tx_at)

    @property
    def is_current(self) -> bool:
        """True when this is the current state on both axes."""
        return self.valid_at is None and self.tx_at is None

    # ------------------------------------------------------------------ SQL

    def tenant(self, alias: str | None = None, *, inline: bool = False) -> tuple[str, list]:
        """``alias.tenant_id = ?`` and its parameter, or the literal form with ``inline=True``."""
        sql, params = _render_tenant(self.tenant_id, inline)
        return f"{_col(alias, 'tenant_id')} = {sql}", params

    def live(self, alias: str | None = None, *, inline: bool = False) -> tuple[str, list]:
        """The tenant predicate plus :func:`live_row_sql`: this tenant's live versions.

        For lookups that identify a row by id and apply no time axis on purpose.  ``get()``
        by id returns a superseded memory and says so through ``Memory.is_current``; what it
        must not return is a version that a later correction has replaced, which is what the
        transaction-time half here excludes.
        """
        t_sql, t_params = self.tenant(alias, inline=inline)
        return f"{t_sql} AND {live_row_sql(alias)}", t_params

    def temporal(self, alias: str | None = None, *, valid_only: bool = False) -> tuple[str, list]:
        """The time half of the predicate: valid time, and transaction time unless
        ``valid_only`` (for tables that carry only valid-time columns)."""
        return temporal_predicate(alias, self.as_of, valid_only=valid_only)

    def predicate(
        self,
        alias: str | None = None,
        *,
        inline_tenant: bool = False,
        valid_only: bool = False,
    ) -> tuple[str, list]:
        """The full WHERE fragment and its bind parameters: tenant, then valid time, then
        transaction time, in that order.

        Callers append it with ``AND``; the fragment contains no leading or trailing keyword.
        """
        t_sql, t_params = self.tenant(alias, inline=inline_tenant)
        w_sql, w_params = self.temporal(alias, valid_only=valid_only)
        return f"{t_sql} AND {w_sql}", t_params + w_params

    # ------------------------------------------------------------------ Python mirror

    def admits(
        self,
        row: Any = None,
        *,
        tenant_id: int | None = None,
        valid_from: _dt.datetime | None = None,
        valid_to: _dt.datetime | None = None,
        tx_from: _dt.datetime | None = None,
        tx_to: _dt.datetime | None = None,
        valid_only: bool = False,
    ) -> bool:
        """The same rule as :meth:`predicate`, evaluated in Python.

        Pass a row object with ``tenant_id``, ``valid_from``, ``valid_to``, ``tx_from`` and
        ``tx_to`` attributes (every value type in :mod:`anatid.types` qualifies), or the fields
        by keyword.  The SQL is the authority; this mirrors it for accelerators that filter in
        Python and for tests.
        """
        if row is not None:
            tenant_id = getattr(row, "tenant_id", tenant_id)
            valid_from = getattr(row, "valid_from", valid_from)
            valid_to = getattr(row, "valid_to", valid_to)
            tx_from = getattr(row, "tx_from", tx_from)
            tx_to = getattr(row, "tx_to", tx_to)
        if tenant_id is None or int(tenant_id) != self.tenant_id:
            return False
        if self.valid_at is None:
            if valid_to is not None:
                return False
        else:
            if valid_from is None or valid_from > self.valid_at:
                return False
            if valid_to is not None and valid_to <= self.valid_at:
                return False
        if valid_only:
            return True
        if self.tx_at is None:
            if tx_to is not None:
                return False
        else:
            if tx_from is None or tx_from > self.tx_at:
                return False
            if tx_to is not None and tx_to <= self.tx_at:
                return False
        return True


def visible_at(
    tenant_id: int,
    valid_time: _dt.datetime | None = None,
    transaction_time: _dt.datetime | None = None,
) -> Visibility:
    """The design document's ``visible_at(valid_time, transaction_time)``, for one tenant."""
    return Visibility(tenant_id=int(tenant_id), valid_at=valid_time, tx_at=transaction_time)


def temporal_predicate(
    alias: str | None, as_of: AsOf | None, *, valid_only: bool = False
) -> tuple[str, list]:
    """WHERE fragment and bind parameters selecting the rows visible under ``as_of``.

    This, and not any engine feature, is anatid's time travel:

    * current (``as_of`` is ``None`` or :data:`anatid.types.CURRENT`):
      ``valid_to IS NULL AND tx_to IS NULL``
    * as of T (valid time): ``valid_from <= T AND (valid_to IS NULL OR valid_to > T)``
    * as of T (transaction time): ``tx_from <= T AND (tx_to IS NULL OR tx_to > T)``

    ``valid_only=True`` skips the transaction-time half for tables that only carry valid-time
    columns.  ``alias`` of ``None`` renders bare column names.  Kept as a module-level function
    because :mod:`anatid.schema` re-exports it under the same name.
    """
    vt = _col(alias, "valid_to")
    vf = _col(alias, "valid_from")
    tt = _col(alias, "tx_to")
    tf = _col(alias, "tx_from")
    if as_of is None or as_of.is_current:
        return (f"{vt} IS NULL" if valid_only else f"{vt} IS NULL AND {tt} IS NULL"), []
    parts: list[str] = []
    params: list = []
    if as_of.valid_time is not None:
        parts.append(f"{vf} <= ? AND ({vt} IS NULL OR {vt} > ?)")
        params += [as_of.valid_time, as_of.valid_time]
    else:
        parts.append(f"{vt} IS NULL")
    if not valid_only:
        if as_of.tx_time is not None:
            parts.append(f"{tf} <= ? AND ({tt} IS NULL OR {tt} > ?)")
            params += [as_of.tx_time, as_of.tx_time]
        else:
            parts.append(f"{tt} IS NULL")
    return " AND ".join(parts), params
