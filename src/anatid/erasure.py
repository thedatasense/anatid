"""Reaching every copy of an erased memory: erasure hooks and the bundled integration tables.

``forget(memory_id, hard=True)`` is a right-to-erasure purge.  The verbs know the memory graph
-- ``memories``, its edges, its episode, its audit rows, the BM25 index tables -- and nothing
else, yet other code puts copies of memory text in the same file.  anatid's own bundled
integrations do it in four tables:

===================  =====================================================================
``agent_messages``   the conversation transcript.  A tool *call* row carries the memory's
                     content; the tool *result* row carries its id **and** its content.
``agent_run_states`` a whole serialised ``RunState``: the conversation so far, the pending
                     tool calls and their arguments, verbatim, in one ``state_json`` string.
``agent_sessions``   / ``agent_turn_usage`` -- counters today, but they are integration
                     tables in the same file and are covered on the same terms, so a column
                     added later cannot quietly reopen the hole.
===================  =====================================================================

Two mechanisms cover them, and the difference between them is the point of this module:

* **Hooks** (:meth:`anatid.Anatid.register_erasure_hook`) live on one handle.  A session or
  store registers a :class:`TableErasureHook` for each table it creates, and the purge runs it.
  That is the general mechanism, and the only one for tables anatid has never heard of.
* **The bundled tables are covered by name**, from the catalog, by ``forget`` itself
  (:func:`purge_bundled_tables`), whether or not this handle constructed the integration that
  wrote them.  Before 0.1.1 they were covered only by hooks, which meant a hard forget issued
  from any *other* handle -- a second process, a maintenance script, the ``anatid-mcp``
  ``forget`` tool on a file the Agents SDK integration also writes -- deleted nothing from
  ``agent_messages`` or ``agent_run_states``: the receipt said ``extra_rows_deleted=0`` and the
  id and the text stayed, ready for ``store.resume(...)`` to replay.  A right to erasure that
  depends on which process asks is not one.  :data:`BUNDLED_INTEGRATION_TABLES` is the list;
  a test asserts it matches the tables the integrations actually create.

How a row is matched
--------------------
A row goes if **any** of its text columns contains the memory's decimal id or its content
verbatim -- a plain substring test (DuckDB ``contains``, so no LIKE wildcards and punctuation is
literal), against the columns discovered from ``PRAGMA table_info`` rather than a hard-coded
list, so a column added to one of these tables later is covered without touching this file.

Three honest consequences, none of them hidden:

* **The whole row goes**, not the matching field.  A parked approval whose conversation quoted
  an erased memory is deleted, so ``store.load(run_id)`` stops finding it; a half-redacted
  ``RunState`` would not deserialise anyway.
* **A short content string over-matches.**  "hi" will take unrelated rows with it.  Erasure is
  destructive by definition and anatid resolves the tie towards erasing: a false positive costs
  a transcript row, a false negative costs the erasure.
* **Scoped to the memory's tenant**, on tables that have a ``tenant_id`` column, but not to one
  session or one run: another conversation in the same tenant that quoted the same text is
  erased too, which is the point.

Everything here runs inside ``forget``'s transaction, so a failure aborts the purge rather than
committing a half-erased file, and the rows removed are reported as the receipt's
``extra_rows_deleted``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from .schema import quote_ident, table_names

log = logging.getLogger("anatid.erasure")

__all__ = [
    "BUNDLED_INTEGRATION_TABLES",
    "TEXT_TYPE_PREFIXES",
    "TableErasureHook",
    "memory_needles",
    "purge_bundled_tables",
    "purge_rows_containing",
    "register_table_erasure_hooks",
    "text_columns",
]

#: The tables anatid's bundled integrations create, in the default names they create them
#: under.  ``forget(hard=True)`` purges every one of these it finds in the catalog, on every
#: handle.  A session or store constructed with a custom table name is covered by the hook it
#: registers on its own handle, as before.
BUNDLED_INTEGRATION_TABLES: tuple[str, ...] = (
    "agent_messages",  # AnatidSession: the transcript
    "agent_sessions",  # AnatidSession: per-session counters
    "agent_turn_usage",  # AnatidSession: per-turn token usage
    "agent_run_states",  # RunStateStore: serialised RunState strings
)

#: DuckDB type names that can hold a verbatim copy of a memory's text.  ``duckdb_columns`` /
#: ``PRAGMA table_info`` report ``VARCHAR`` for every string type, but the aliases are listed so
#: a table created by hand with ``TEXT``/``JSON`` columns is still covered.
TEXT_TYPE_PREFIXES = ("VARCHAR", "TEXT", "STRING", "CHAR", "BPCHAR", "JSON")


def text_columns(db: Any, table: str) -> tuple[str, ...]:
    """The text-typed column names of ``table``, from the catalog.  ``()`` if it does not exist.

    Read from ``PRAGMA table_info`` on every call rather than cached: these tables are created
    by whichever integration object was constructed first, and a hook registered by one of them
    may run against a table another one has since added a column to.
    """
    try:
        rows = db.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    except Exception:  # noqa: BLE001 - unknown table, closed handle
        return ()
    out: list[str] = []
    for row in rows:
        name, type_name = row[1], str(row[2] or "").upper()
        if any(type_name.startswith(prefix) for prefix in TEXT_TYPE_PREFIXES):
            out.append(str(name))
    return tuple(out)


def _has_column(db: Any, table: str, column: str) -> bool:
    try:
        rows = db.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    except Exception:  # noqa: BLE001 - unknown table, closed handle
        return False
    return any(str(row[1]) == column for row in rows)


def memory_needles(memory_id: int, content: str | None) -> list[str]:
    """The strings whose presence in a row means that row is a copy of the erased memory.

    Both are needed and neither is redundant: a tool *call* row quotes the content and never the
    id, a tool *result* row quotes the id and usually the content.  Empty content is dropped --
    ``contains(x, '')`` is true for every row, which would erase the table.
    """
    needles = [str(int(memory_id))]
    if content:
        text = str(content)
        if text:
            needles.append(text)
    return needles


def purge_rows_containing(
    db: Any,
    table: str,
    needles: Sequence[str],
    *,
    tenant_id: int | None = None,
) -> int:
    """Delete every row of ``table`` whose text columns contain any of ``needles``.

    Returns the number of rows deleted; ``0`` when the table does not exist, has no text columns,
    or nothing matched.  ``tenant_id`` is applied only if the table has such a column.
    """
    cleaned = [n for n in needles if n]
    if not cleaned:
        return 0
    columns = text_columns(db, table)
    if not columns:
        return 0

    tests = " OR ".join(
        f"contains(coalesce({quote_ident(c)}, ''), ?)" for _ in cleaned for c in columns
    )
    params: list[Any] = []
    where = ""
    if tenant_id is not None and _has_column(db, table, "tenant_id"):
        where = "tenant_id = ? AND "
        params.append(int(tenant_id))
    for needle in cleaned:
        params.extend([needle] * len(columns))

    cur = db.execute(f"DELETE FROM {quote_ident(table)} WHERE {where}({tests})", params)
    try:
        row = cur.fetchone()
    except Exception:  # pragma: no cover - driver dependent
        return 0
    return 0 if row is None or row[0] is None else int(row[0])


class TableErasureHook:
    """An erasure hook for one table.

    Equality is by table name so that :meth:`anatid.Anatid.register_erasure_hook` -- which
    ignores a hook it already holds -- registers one hook per table however many sessions or
    stores are constructed on the same handle.  Without that, two ``RunStateStore``s on one
    database would each delete the same rows and the receipt would double-count them.  The same
    equality lets ``forget`` skip the bundled-table pass for a table a hook already covers.
    """

    __slots__ = ("table",)

    def __init__(self, table: str) -> None:
        self.table = str(table)

    def __call__(self, db: Any, memory_id: int, tenant_id: int, content: str | None = None) -> int:
        deleted = purge_rows_containing(
            db, self.table, memory_needles(memory_id, content), tenant_id=tenant_id
        )
        if deleted:
            log.info(
                "erasure: removed %d row(s) from %s for memory %s", deleted, self.table, memory_id
            )
        return deleted

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TableErasureHook) and other.table == self.table

    def __hash__(self) -> int:
        return hash((TableErasureHook, self.table))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TableErasureHook {self.table!r}>"


def register_table_erasure_hooks(db: Any, tables: Iterable[str]) -> tuple[TableErasureHook, ...]:
    """Register one :class:`TableErasureHook` per table on ``db``.  Idempotent per table.

    Returns the hooks, so a caller can keep them (the handle already does) or assert on them.
    """
    hooks = tuple(TableErasureHook(t) for t in dict.fromkeys(tables))
    for hook in hooks:
        db.register_erasure_hook(hook)
    return hooks


def purge_bundled_tables(
    db: Any,
    memory_id: int,
    tenant_id: int,
    content: str | None,
    *,
    skip: Iterable[str] = (),
) -> int:
    """Erase ``memory_id`` from every :data:`BUNDLED_INTEGRATION_TABLES` member in the catalog.

    Called by ``forget(hard=True)`` on **every** handle, after the handle's registered hooks,
    with ``skip`` naming the tables those hooks already covered (so a row is neither deleted
    twice nor counted twice).  Returns the number of rows deleted.  A table that is absent --
    the usual case on a file no integration has touched -- costs one catalog query.
    """
    present = table_names(db.connection if hasattr(db, "connection") else db)
    covered = set(skip)
    needles = memory_needles(memory_id, content)
    deleted = 0
    for table in BUNDLED_INTEGRATION_TABLES:
        if table in covered or table not in present:
            continue
        n = purge_rows_containing(db, table, needles, tenant_id=tenant_id)
        if n:
            log.info(
                "erasure: removed %d row(s) from bundled table %s for memory %s",
                n,
                table,
                memory_id,
            )
        deleted += n
    return deleted
