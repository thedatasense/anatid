"""The read-only SQL escape hatch used by the anatid MCP server.

Why an escape hatch exists at all
---------------------------------
``recall`` / ``context`` / ``provenance`` answer the questions anatid was designed for.
An agent that needs something else -- "how many memories per ``kind``?", "which entities
have the most ABOUT edges?", "show me the raw ``anatid_audit`` trail" -- would otherwise
be stuck.  So the server exposes one tool that runs arbitrary SQL, and that tool is
**read-only with respect to the database**.

How read-only is enforced
-------------------------
Three layers, and *none of them is a regular expression over the SQL text*.  Two of them
are DuckDB deciding, and the third is DuckDB refusing.

1. **DuckDB's parser/binder classifies the statement.**
   :func:`duckdb.DuckDBPyConnection.extract_statements` returns one ``duckdb.Statement``
   per statement in the text, each carrying a ``duckdb.StatementType``.  Only
   :attr:`~duckdb.StatementType.SELECT` and :attr:`~duckdb.StatementType.EXPLAIN` are
   accepted, and **every** statement in the text must pass -- so
   ``SELECT 1; DELETE FROM memories`` is rejected on the second one rather than
   half-executed.  Because this is DuckDB's own classification rather than ours, it sees
   through things a regex cannot: ``PRAGMA create_fts_index(...)`` is expanded at bind
   time into the ``CREATE``/``INSERT``/``UPDATE`` statements it really is, and is rejected
   on those; ``ATTACH``, ``COPY``, ``INSTALL``/``LOAD``, ``CALL``, ``SET``, ``PREPARE``,
   ``VACUUM`` and ``BEGIN`` each get their own type and none of them is on the allowlist.

   ``EXPLAIN ANALYZE <statement>`` *executes* the statement it explains, and DuckDB still
   types the whole thing ``EXPLAIN``.  So an ``EXPLAIN`` is accepted only when the
   statement it wraps is itself classified ``SELECT`` -- the ``EXPLAIN`` keyword and its
   option list are lexed off (see :func:`_strip_explain`) and the remainder is handed back
   to ``extract_statements``.  The *decision* is still DuckDB's; only the keyword removal
   is lexical.

2. **DuckDB's AST is scanned for anything that reads outside the database.**
   ``json_serialize_sql()`` turns the statement into DuckDB's own parse tree, and two
   things in it are checked:

   * every ``function_name``, against :data:`DENIED_FUNCTIONS` (``read_csv``,
     ``read_parquet``, ``glob``, ``postgres_scan``, ``duckdb_secrets``, and ``query`` /
     ``query_table``, which take SQL as a *string* and would otherwise hide a denied
     function from this scan entirely);
   * every ``BASE_TABLE`` name, which must be a plain identifier
     (:func:`_is_plain_identifier`).  This is not pedantry: DuckDB's **replacement scan**
     means ``SELECT * FROM '/etc/passwd.csv'`` is a perfectly ordinary ``SELECT`` whose AST
     contains *no* ``function_name`` at all -- the path is the table name.  The same trick
     reaches globs (``FROM '/data/*.parquet'``) and, via httpfs autoloading, arbitrary URLs
     (``FROM 'https://attacker.example/x.csv'``), which would make the MCP host issue
     outbound requests a prompt-injected model chose.  A function deny-list cannot see any
     of that, so the table names are allow-listed by shape instead.

   This layer is about *reading* things outside the database, not about writing -- layers 1
   and 3 own writes.  If DuckDB cannot serialize the statement the scan is skipped and the
   result says so (``ast_scanned: false``); layers 1 and 3 still hold.

3. **DuckDB's transaction manager refuses the write.**  The accepted statement runs on a
   private cursor inside ``BEGIN TRANSACTION READ ONLY``, and the transaction is *always*
   rolled back.  Any attempt to write the database -- including one smuggled through
   ``EXPLAIN ANALYZE`` -- fails with ``TransactionContext Error: Cannot write to database
   ... transaction is launched in read-only mode``.  Verified on DuckDB 1.5.5 for
   ``INSERT``, ``UPDATE``, ``DELETE``, ``CREATE``, ``DROP``, ``COPY ... FROM`` and
   ``EXPLAIN ANALYZE INSERT``.

   Measured caveat, which is exactly why layer 1 is not optional: a read-only *transaction*
   does **not** by itself stop ``ATTACH``, ``COPY ... TO 'file'``, ``INSTALL`` or
   ``CHECKPOINT``, because those do not write the current database.  Layer 1 rejects all
   four by statement type before layer 3 is ever reached.

What is *not* enforced
----------------------
* The gateway is **not tenant-filtered**.  ``Isolation.SCOPED`` is a column predicate that
  anatid's verbs add; raw SQL does not get it, so a query here can read every tenant in the
  file.  Real isolation is one file per tenant (:class:`anatid.DatabasePool`); with that,
  and with layer 2's table-name check closing the replacement scan, this tool can only see
  the file the server was pointed at.
* A ``SELECT`` can still be *slow*.  Row output is capped (``limit``), but a full scan of a
  large table costs what it costs.

Set ``ANATID_SQL_TOOL=off`` to remove the tool from the server entirely.
"""

from __future__ import annotations

import json
import math
import datetime as _dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

import duckdb

__all__ = [
    "SqlNotAllowed",
    "SqlGateway",
    "GateDecision",
    "ALLOWED_STATEMENT_TYPES",
    "DENIED_FUNCTIONS",
    "ENFORCEMENT",
]


ALLOWED_STATEMENT_TYPES = frozenset(
    {duckdb.StatementType.SELECT, duckdb.StatementType.EXPLAIN}
)
"""The only ``duckdb.StatementType`` values this gateway will run."""


DENIED_FUNCTIONS = frozenset(
    {
        # local filesystem readers
        "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json",
        "read_json_auto", "read_json_objects", "read_json_objects_auto", "read_ndjson",
        "read_ndjson_auto", "read_ndjson_objects", "read_text", "read_blob", "glob",
        "sniff_csv", "parquet_metadata", "parquet_schema", "parquet_file_metadata",
        "parquet_kv_metadata", "parquet_bloom_probe", "read_xlsx", "st_read", "st_readosm",
        "st_read_meta",
        # lakehouse / external connectors
        "iceberg_scan", "iceberg_metadata", "iceberg_snapshots", "delta_scan",
        "postgres_scan", "postgres_scan_pushdown", "postgres_query", "mysql_scan",
        "mysql_query", "sqlite_scan", "sqlite_query", "read_gsheet",
        # extension / secret / process surface
        "load_extension", "install_extension", "duckdb_secrets", "which_secret",
        "shell", "system",
        # take SQL as a STRING: whatever they run is a VALUE_CONSTANT in the AST, so a denied
        # function nested inside one is invisible to the function scan below.  Measured:
        # query('SELECT * FROM read_csv(''/x.csv'')') returned the file while the un-wrapped
        # read_csv was refused.
        "query", "query_table",
    }
)
"""Function names rejected by the AST scan (layer 2).  These read outside the database."""


ENFORCEMENT = (
    "read-only, enforced by DuckDB in three layers: (1) duckdb's own parser classifies every "
    "statement in the text and only StatementType.SELECT / EXPLAIN are run (an EXPLAIN must "
    "wrap a SELECT, because EXPLAIN ANALYZE executes what it explains); (2) the statement's "
    "json_serialize_sql AST is scanned for filesystem/external-connector functions AND for base "
    "table names that are not plain identifiers (DuckDB's replacement scan makes "
    "SELECT * FROM '/path/file.csv' an ordinary SELECT); (3) the "
    "statement runs on a private cursor inside BEGIN TRANSACTION READ ONLY and is always "
    "ROLLBACKed, so DuckDB itself refuses any write. Not a regex. Not tenant-filtered."
)


class SqlNotAllowed(ValueError):
    """The submitted SQL was rejected before anything ran."""


@dataclass(frozen=True)
class GateDecision:
    """What the gate concluded about one submission (used by the tests and the tool result)."""

    allowed: bool
    statement_types: tuple[str, ...] = ()
    reason: str | None = None
    ast_scanned: bool = False
    functions: tuple[str, ...] = field(default=())
    tables: tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------- lexing helpers


def _skip_trivia(text: str, i: int) -> int:
    """Advance past whitespace and SQL comments.  Used only to find the EXPLAIN keyword."""
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif text.startswith("--", i):
            nl = text.find("\n", i)
            i = n if nl == -1 else nl + 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            break
    return i


def _read_word(text: str, i: int) -> tuple[str, int]:
    j = i
    while j < len(text) and (text[j].isalnum() or text[j] == "_"):
        j += 1
    return text[i:j], j


def _skip_parens(text: str, i: int) -> int:
    """``text[i]`` must be ``(``; return the index just past its matching ``)``."""
    depth = 0
    n = len(text)
    while i < n:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        elif text[i] == "'":
            i += 1
            while i < n and text[i] != "'":
                i += 1
        i += 1
    return n


def _strip_explain(text: str) -> str | None:
    """Return the statement an ``EXPLAIN`` wraps, or ``None`` if ``text`` is not an EXPLAIN.

    Handles ``EXPLAIN x``, ``EXPLAIN ANALYZE x``, ``EXPLAIN (ANALYZE, FORMAT JSON) x`` and
    leading comments.  Only the keyword removal is lexical -- the *classification* of what
    is left is done by DuckDB in :meth:`SqlGateway.inspect`.
    """
    i = _skip_trivia(text, 0)
    word, i = _read_word(text, i)
    if word.upper() != "EXPLAIN":
        return None
    while True:
        i = _skip_trivia(text, i)
        if i < len(text) and text[i] == "(":
            i = _skip_parens(text, i)
            continue
        save = i
        word, j = _read_word(text, i)
        if word.upper() in ("ANALYZE", "VERBOSE"):
            i = j
            continue
        i = save
        break
    return text[i:]


# --------------------------------------------------------------------------- value coercion


def _jsonable(value: Any) -> Any:
    """Coerce one DuckDB cell into something ``json.dumps`` and pydantic both accept."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


# --------------------------------------------------------------------------- the gateway


class SqlGateway:
    """Runs read-only SQL against an open :class:`anatid.Anatid` database.

    ``connection_factory`` is called for each submission and must return a DuckDB
    connection on the target database; the gateway makes its own ``cursor()`` from it so
    the caller's transaction state is never disturbed.  See the module docstring for the
    enforcement contract.
    """

    def __init__(self, connection_factory, *, max_rows: int = 200) -> None:
        self._factory = connection_factory
        self.max_rows = int(max_rows)

    # -- layer 1 + 2 ---------------------------------------------------------

    def inspect(self, sql: str) -> GateDecision:
        """Classify ``sql`` without running it.  Never executes the statement."""
        text = (sql or "").strip()
        if not text:
            return GateDecision(False, reason="empty statement")

        con = self._factory()
        try:
            statements = con.extract_statements(text)
        except duckdb.Error as exc:
            first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            return GateDecision(False, reason=f"DuckDB could not parse this: {first}")

        if not statements:
            return GateDecision(False, reason="no statement found (comments only?)")

        # NB: `duckdb.StatementType` is a pybind11 enum, so `st.type is StatementType.X` is
        # False even when they are the same value (each attribute access builds a fresh
        # wrapper object).  Compare with `==` / set membership only -- an `is` here silently
        # disables the check.
        types = tuple(s.type.name for s in statements)
        for st in statements:
            if st.type not in ALLOWED_STATEMENT_TYPES:
                return GateDecision(
                    False,
                    types,
                    reason=(
                        f"DuckDB classified this as {st.type.name}; the anatid sql tool is "
                        f"read-only and runs only SELECT / EXPLAIN. "
                        f"Statement types found: {', '.join(types)}."
                    ),
                )

        # EXPLAIN ANALYZE runs what it explains, and DuckDB still types the whole thing
        # EXPLAIN -- so re-classify the wrapped statement.
        inner_texts: list[str] = []
        for st in statements:
            if st.type != duckdb.StatementType.EXPLAIN:
                inner_texts.append(st.query)
                continue
            inner = _strip_explain(st.query)
            if inner is None or not inner.strip():
                return GateDecision(False, types, reason="could not read the statement this EXPLAIN wraps")
            try:
                nested = con.extract_statements(inner)
            except duckdb.Error as exc:
                first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
                return GateDecision(False, types, reason=f"could not classify the EXPLAINed statement: {first}")
            if len(nested) != 1 or nested[0].type != duckdb.StatementType.SELECT:
                found = ", ".join(n.type.name for n in nested) or "nothing"
                return GateDecision(
                    False, types,
                    reason=(f"EXPLAIN is allowed only over a SELECT (EXPLAIN ANALYZE executes what "
                            f"it explains); the wrapped statement is {found}."),
                )
            inner_texts.append(nested[0].query)

        # -- layer 2: walk DuckDB's AST for anything that reads outside the database
        scanned = True
        seen: set[str] = set()
        tables: set[str] = set()
        for piece in inner_texts:
            try:
                blob = con.execute("SELECT json_serialize_sql(?)", [piece]).fetchone()[0]
                tree = json.loads(blob)
            except Exception:                     # noqa: BLE001 - serializer is best-effort
                scanned = False
                continue
            if tree.get("error"):
                scanned = False
                continue
            _collect_ast(tree, seen, tables)

        denied = sorted(n for n in seen if n.lower() in DENIED_FUNCTIONS)
        if denied:
            return GateDecision(
                False, types, ast_scanned=scanned, functions=tuple(sorted(seen)),
                tables=tuple(sorted(tables)),
                reason=(f"rejected: {', '.join(denied)} read data from outside the database. "
                        f"The anatid sql tool may only read the anatid database itself."),
            )

        # DuckDB's replacement scan turns a quoted path or URL into a BASE_TABLE whose name IS
        # the path, with no function anywhere in the tree.  Allow-list the shape.
        odd = sorted(t for t in tables if not _is_plain_identifier(t))
        if odd:
            return GateDecision(
                False, types, ast_scanned=scanned, functions=tuple(sorted(seen)),
                tables=tuple(sorted(tables)),
                reason=(f"rejected: {', '.join(repr(o) for o in odd)} is not a table name in this "
                        f"database. DuckDB's replacement scan would read it as a file or URL, and "
                        f"the anatid sql tool may only read the anatid database itself."),
            )

        return GateDecision(True, types, ast_scanned=scanned, functions=tuple(sorted(seen)),
                            tables=tuple(sorted(tables)))

    # -- layer 3 -------------------------------------------------------------

    def run(self, sql: str, *, limit: int | None = None) -> dict[str, Any]:
        """Gate ``sql`` and, if it passes, run it in a read-only transaction.

        Raises :class:`SqlNotAllowed` when the gate rejects it -- nothing has run at that
        point.  DuckDB errors from a legitimately-allowed query propagate as
        ``duckdb.Error``.
        """
        decision = self.inspect(sql)
        if not decision.allowed:
            raise SqlNotAllowed(decision.reason or "rejected")

        cap = self.max_rows if limit is None else max(1, min(int(limit), self.max_rows))
        con = self._factory().cursor()
        try:
            con.execute("BEGIN TRANSACTION READ ONLY")
            try:
                result = con.execute(sql)
                columns = [d[0] for d in (result.description or [])]
                rows = result.fetchmany(cap)
                truncated = len(result.fetchmany(1)) > 0
            finally:
                try:
                    con.execute("ROLLBACK")
                except duckdb.Error:              # already aborted / never opened
                    pass
        finally:
            try:
                con.close()
            except duckdb.Error:                  # pragma: no cover - close is best effort
                pass

        return {
            "columns": columns,
            "rows": [[_jsonable(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
            "statement_types": list(decision.statement_types),
            "ast_scanned": decision.ast_scanned,
            "enforcement": ENFORCEMENT,
        }


_IDENT_EXTRA = set("_$")


def _is_plain_identifier(name: str) -> bool:
    """True for ``memories``, ``fts_main_memories``, a CTE alias -- false for a path or URL.

    Deliberately an allow-list: a deny-list of ``/``, ``://``, ``*`` and data-file suffixes
    would have to keep up with every future replacement scan DuckDB adds.
    """
    if not name or len(name) > 128:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c in _IDENT_EXTRA for c in name)


def _collect_ast(node: Any, functions: set[str], tables: set[str], key: str | None = None) -> None:
    """Walk DuckDB's serialized parse tree collecting function names and base-table names."""
    if isinstance(node, dict):
        if node.get("type") == "BASE_TABLE":
            for k in ("table_name", "schema_name", "catalog_name"):
                v = node.get(k)
                if isinstance(v, str) and v:
                    tables.add(v)
        for k, v in node.items():
            _collect_ast(v, functions, tables, k)
    elif isinstance(node, list):
        for v in node:
            _collect_ast(v, functions, tables, key)
    elif isinstance(node, str) and key == "function_name":
        functions.add(node)


def _collect_function_names(node: Any, out: set[str], key: str | None = None) -> None:
    """Back-compatible wrapper: collect only ``function_name`` nodes."""
    _collect_ast(node, out, set(), key)


def sanitize_rows(rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """Public helper: coerce a DuckDB result set into JSON-safe nested lists."""
    return [[_jsonable(v) for v in row] for row in rows]
