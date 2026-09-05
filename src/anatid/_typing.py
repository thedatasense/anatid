"""Structural types for anatid's internal seams -- no runtime behaviour lives here.

``MemoryVerbs`` (:mod:`anatid.verbs`) is a mixin: it calls ``execute()``, ``transaction()``,
``resolve_tenant()``, ``connection``, ``config`` and ``csr``, but it does not define them.
:class:`anatid.database.Anatid` does.  Until now the mixin declared those seams as bodyless
placeholders::

    def execute(self, sql, params=None, *, con=None): ...  # infers -> None
    def transaction(self, con=None): ...  # infers -> None

which is worse than no annotation at all: a type checker believes the return type really *is*
``None``, so every ``self.execute(...).fetchone()`` is an attribute access on ``None`` and every
``with self.transaction():`` is a context manager that is ``None``.  Those two lines alone
account for 32 of the 58 pyright errors reported against anatid 0.1.0.

:class:`VerbHost` states the contract properly, and :data:`VerbHostMixin` is the standard
"typed mixin" base: it is :class:`VerbHost` to a type checker and plain :class:`object` at run
time, so declaring the mixin as::

    from ._typing import VerbHostMixin


    class MemoryVerbs(VerbHostMixin): ...

adds exactly one empty, ``__slots__ = ()`` base to the MRO and changes no ``isinstance``
result and no behaviour -- ``VerbHostMixin`` defines nothing, so attribute lookup on a
``MemoryVerbs`` instance resolves precisely as it did with ``object`` -- while giving the
checker the real signatures.  Deleting
the placeholder ``def``\\ s along with the class-level ``config: Any`` / ``csr: Any`` /
``connection: Any`` annotations is the point: they are what shadow the host's real types.

:class:`VerbHost` is :func:`~typing.runtime_checkable`, so ``isinstance(obj, VerbHost)`` works,
but note what :pep:`544` guarantees there -- presence of the attributes, never their signatures.
It is a debugging convenience, not a validator.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Type-only: ``from __future__ import annotations`` keeps every annotation below a string,
    # and :func:`~typing.runtime_checkable` only ever looks at attribute *names*, so none of
    # these is needed at import time.  Keeping them here means this module never participates
    # in an import cycle with the modules it describes.
    import duckdb

    from .csr import CsrBackend
    from .embed import Embedder
    from .schema import SchemaConfig
    from .types import Namespace

__all__ = [
    "Row",
    "Rows",
    "SqlParams",
    "VerbHost",
    "VerbHostMixin",
]

#: One row as DuckDB's Python client hands it back.
Row = tuple[Any, ...]
#: A materialised result set.
Rows = list[Row]
#: What the verbs bind into a statement.  Never caller text -- see :mod:`anatid.verbs`.
SqlParams = Sequence[Any]


@runtime_checkable
class VerbHost(Protocol):
    """What :class:`anatid.verbs.MemoryVerbs` requires of the object it is mixed into.

    :class:`anatid.database.Anatid` is the only implementation in the tree.  A test double only
    has to provide these six members to drive the verbs.
    """

    #: The DDL knobs the database was opened with (``embedding_dim`` is read on every write).
    config: SchemaConfig
    #: Graph-expansion backend: the C++ extension when it loaded, otherwise recursive SQL.
    csr: CsrBackend
    #: The embedding model ``Anatid.open(embedder=...)`` stored, or None.  With one set, the
    #: write verbs embed content they were not given an embedding for and ``recall`` embeds
    #: the query.
    embedder: Embedder | None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """This thread's DuckDB connection."""
        ...

    def execute(
        self,
        sql: str,
        params: SqlParams | None = None,
        *,
        con: duckdb.DuckDBPyConnection | None = None,
    ) -> duckdb.DuckDBPyConnection:
        """Run one statement, translating a DuckDB write-write conflict into
        :class:`~anatid.errors.ConflictError`.

        Returns the connection the statement ran on, which is what DuckDB's Python client
        returns and what makes ``self.execute(...).fetchone()`` legal.
        """
        ...

    def transaction(
        self, con: duckdb.DuckDBPyConnection | None = None
    ) -> AbstractContextManager[duckdb.DuckDBPyConnection]:
        """Re-entrant explicit transaction on this thread's connection."""
        ...

    def resolve_tenant(self, tenant: int | Namespace | None = None) -> Namespace:
        """Resolve a verb's ``tenant=`` argument, enforcing the file-per-tenant boundary."""
        ...


if TYPE_CHECKING:
    #: Base class for the verb mixin.  ``VerbHost`` while type checking, ``object`` at run time.
    VerbHostMixin = VerbHost
else:

    class VerbHostMixin:
        """Run-time stand-in for :class:`VerbHost`: an empty base, identical to ``object``.

        Inheriting from a :class:`~typing.Protocol` at run time would make the mixin abstract
        and add ``Protocol`` to its MRO.  Neither is wanted; only the checker needs the types.
        """

        __slots__ = ()
