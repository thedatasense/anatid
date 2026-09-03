"""The read-only SQL escape hatch used by the anatid MCP server.

Why an escape hatch exists at all
---------------------------------
``recall`` / ``context`` / ``provenance`` answer the questions anatid was designed for.
An agent that needs something else -- "how many memories per ``kind``?", "which entities
have the most ABOUT edges?", "show me the raw ``anatid_audit`` trail" -- would otherwise
be stuck.  So the server exposes one tool that runs arbitrary SQL, and that tool is
**read-only with respect to the database**.

It is also **off by default** (layer 0).  The tool is registered only when the operator
opts in with ``ANATID_ENABLE_SQL=1`` or ``build_server(db, ServerConfig(sql_tool=True))``;
a default-on escape hatch means every prompt-injected model that reaches an anatid MCP
server gets an arbitrary-SQL primitive it was never meant to have.

How read-only is enforced
-------------------------
Three layers, and *none of them is a regular expression over the SQL text*.  Two of them
are DuckDB deciding, and the third is DuckDB refusing.  **Every layer fails closed**: if a
check cannot be *completed*, the statement is refused rather than run unchecked.

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
     ``read_parquet``, ``glob``, ``postgres_scan``, ``duckdb_secrets``, the host-metadata
     functions such as ``duckdb_settings`` / ``current_setting`` / ``duckdb_databases`` that
     return filesystem paths and process configuration rather than memory data, and
     ``query`` / ``query_table``, which take SQL as a *string* and would otherwise hide a
     denied function from this scan entirely);
   * every ``BASE_TABLE`` name, which must be a plain identifier
     (:func:`_is_plain_identifier`).  This is not pedantry: DuckDB's **replacement scan**
     means ``SELECT * FROM '/etc/passwd.csv'`` is a perfectly ordinary ``SELECT`` whose AST
     contains *no* ``function_name`` at all -- the path is the table name.  The same trick
     reaches globs (``FROM '/data/*.parquet'``) and, via httpfs autoloading, arbitrary URLs
     (``FROM 'https://attacker.example/x.csv'``), which would make the MCP host issue
     outbound requests a prompt-injected model chose.  A function deny-list cannot see any
     of that, so the table names are allow-listed by shape instead.

   This layer is about *reading* things outside the database, not about writing -- layers 1
   and 3 own writes.  **If DuckDB cannot serialize the statement, the statement is refused.**
   It used to be skipped -- ``json_serialize_sql`` raising, or returning
   ``{"error": true, ...}``, set ``ast_scanned: false`` and the query ran anyway with only
   layers 1 and 3 behind it, which is precisely the "unscannable input is trusted input"
   shape.  Layers 1 and 3 do not cover what layer 2 covers: a replacement scan
   (``SELECT * FROM '/etc/passwd.csv'``) is a ``SELECT`` that writes nothing, so it passes
   both.  An unscannable statement is therefore an unenforceable one, and it is denied;
   ``GateDecision.ast_scanned`` is ``True`` on every accepted statement.

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

4. **DuckDB itself is hardened, and the query is bounded.**  Constructing a
   :class:`SqlGateway` (with ``harden=True``, the default) runs, on the target database:

   * ``SET enable_external_access=false`` -- DuckDB's own recommendation for an untrusted
     SQL surface (https://duckdb.org/docs/current/operations_manual/securing_duckdb/overview).
     This is belt-and-braces behind layer 2: even a replacement scan or an autoloaded
     ``httpfs`` that got past the AST check cannot reach the filesystem or the network.
   * ``SET memory_limit=<memory_limit>`` (default :data:`DEFAULT_MEMORY_LIMIT`), so one
     query cannot exhaust the host.

   Two honest consequences, because both settings are DuckDB **GLOBAL** scope -- there is no
   per-statement scope for either (``duckdb_settings()`` says ``GLOBAL`` for both on 1.5.5),
   so they apply to the whole database instance the gateway was pointed at, not just to the
   gateway's cursor:

   * ``enable_external_access`` is one-way: DuckDB refuses to re-enable it while the database
     is running ("Cannot enable external access while database is running").  After a gateway
     is built, that handle can no longer ``read_parquet``/``read_csv``, nor autoload an
     extension it has not already loaded.  anatid loads ``fts`` in :meth:`Anatid.open`, before
     any gateway exists, so BM25 keeps working; ``load_parquet`` on the same handle does not.
     This is why the SQL tool is opt-in: opting into arbitrary SQL opts into the hardening.
   * With external access off DuckDB cannot spill to its temp directory, so ``memory_limit`` is
     a hard cap: an over-large query fails with ``Out of Memory Error`` instead of spilling.

   The wall-clock bound is separate, because DuckDB has no ``statement_timeout`` setting: a
   watchdog thread calls :meth:`duckdb.DuckDBPyConnection.interrupt` after ``timeout`` seconds
   (default :data:`DEFAULT_TIMEOUT_SECONDS`), which raises ``duckdb.InterruptException`` in the
   running query; the gateway turns that into :class:`SqlTimeout`.

What is *not* enforced
----------------------
* The gateway is **not tenant-filtered**.  ``Isolation.SCOPED`` is a column predicate that
  anatid's verbs add; raw SQL does not get it, so a query here can read every tenant in the
  file.  Real isolation is one file per tenant (:class:`anatid.DatabasePool`); with that,
  and with layer 2's table-name check closing the replacement scan, this tool can only see
  the file the server was pointed at.
* A ``SELECT`` can still be *slow*, up to ``timeout``.  Row output is capped (``limit``), but
  a full scan of a large table costs what it costs until the watchdog fires.
* anatid implements no authentication of its own.  The stdio transport needs none; an HTTP
  transport bound to a non-loopback interface is refused unless the operator declares that an
  authenticating proxy fronts it (see ``anatid.integrations.mcp.server.check_transport_security``
  and https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization).

The tool is opt-in: ``ANATID_ENABLE_SQL=1`` (or ``--enable-sql``, or
``ServerConfig(sql_tool=True)``) registers it, and ``ANATID_SQL_TOOL=off`` still forces it off.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import datetime as _dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

import duckdb

log = logging.getLogger("anatid.integrations.mcp")

__all__ = [
    "SqlNotAllowed",
    "SqlTimeout",
    "SqlGateway",
    "GateDecision",
    "ALLOWED_STATEMENT_TYPES",
    "DENIED_FUNCTIONS",
    "ENFORCEMENT",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MEMORY_LIMIT",
]

#: Wall-clock bound for one submitted statement.  DuckDB has no ``statement_timeout`` setting,
#: so this is enforced with :meth:`duckdb.DuckDBPyConnection.interrupt` from a watchdog thread.
DEFAULT_TIMEOUT_SECONDS = 30.0

#: ``SET memory_limit`` applied to the database the gateway runs on.  GLOBAL scope in DuckDB.
DEFAULT_MEMORY_LIMIT = "1GB"


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
        # host metadata: read-only, but what they return is the host's configuration and
        # filesystem layout (home_directory, temp_directory, extension_directory, the paths of
        # every attached database, the spill files on disk, the platform triple), none of which
        # is memory data.  The tool's contract is "the anatid database itself".
        "duckdb_settings", "current_setting", "duckdb_databases", "duckdb_temporary_files",
        "duckdb_extensions", "duckdb_logs", "duckdb_log_contexts", "duckdb_memory",
        "pragma_database_size", "pragma_storage_info", "pragma_platform", "pragma_user_agent",
        "pragma_metadata_info", "duckdb_prepared_statements",
    }
)
"""Function names rejected by the AST scan (layer 2).  These read outside the database, or
report the host's configuration and filesystem layout rather than the database's contents."""


ENFORCEMENT = (
    "opt-in (off unless ANATID_ENABLE_SQL=1) and read-only, enforced by DuckDB in three layers, "
    "each of which fails CLOSED: (1) duckdb's own parser classifies every "
    "statement in the text and only StatementType.SELECT / EXPLAIN are run (an EXPLAIN must "
    "wrap a SELECT, because EXPLAIN ANALYZE executes what it explains); (2) the statement's "
    "json_serialize_sql AST is scanned for filesystem/external-connector/host-metadata functions "
    "AND for base "
    "table names that are not plain identifiers (DuckDB's replacement scan makes "
    "SELECT * FROM '/path/file.csv' an ordinary SELECT) -- a statement DuckDB cannot serialize "
    "is REFUSED, never run unscanned; (3) the "
    "statement runs on a private cursor inside BEGIN TRANSACTION READ ONLY and is always "
    "ROLLBACKed, so DuckDB itself refuses any write. The database also runs with "
    "enable_external_access=false and a memory_limit, and each statement has a wall-clock "
    "timeout. Not a regex. Not tenant-filtered."
)


class SqlNotAllowed(ValueError):
    """The submitted SQL was rejected before anything ran."""


class SqlTimeout(ValueError):
    """The submitted SQL ran longer than the gateway's ``timeout`` and was interrupted.

    A ``ValueError`` on purpose: the MCP server's ``_guard`` turns it into a ``ToolError`` the
    model can read and act on ("that query was too expensive, narrow it"), rather than a crash.
    Nothing was committed -- the statement ran inside the read-only transaction that is always
    rolled back.
    """


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

    ``timeout``
        Seconds before the running statement is interrupted (:class:`SqlTimeout`).
        ``None`` or ``0`` disables the watchdog.
    ``memory_limit``
        Passed to ``SET memory_limit``.  ``None`` leaves DuckDB's default alone.
    ``harden``
        Apply ``SET enable_external_access=false`` and ``memory_limit`` to the database, once,
        at construction.  Both are DuckDB **GLOBAL** settings, so they apply to the whole
        database instance and ``enable_external_access`` cannot be turned back on while that
        database is open -- see the module docstring.  ``harden=False`` exists so a caller who
        needs the handle's filesystem access (``load_parquet``) can keep it and accept that
        only layers 1-3 are in the way.

    :attr:`hardening` records what was actually applied, and the ``sql`` tool reports it in
    ``stats()``, so "we set enable_external_access=false" is checkable rather than claimed.
    """

    def __init__(
        self,
        connection_factory,
        *,
        max_rows: int = 200,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
        memory_limit: str | None = DEFAULT_MEMORY_LIMIT,
        harden: bool = True,
    ) -> None:
        self._factory = connection_factory
        self.max_rows = int(max_rows)
        self.timeout = None if not timeout else float(timeout)
        self.memory_limit = memory_limit or None
        self.hardening: dict[str, Any] = {
            "external_access_disabled": False,
            "memory_limit": None,
            "timeout_seconds": self.timeout,
            "errors": [],
        }
        if harden:
            self._harden()

    def _harden(self) -> None:
        """Apply DuckDB's own recommendations for an untrusted SQL surface.  Idempotent.

        Failures are recorded in :attr:`hardening` and logged rather than raised: a database
        that will not take these settings (an unusual build, a locked configuration) must still
        serve memory verbs, and layers 1-3 do not depend on this one.
        """
        con = self._factory()
        if self.memory_limit is not None:
            try:
                con.execute(f"SET memory_limit='{self.memory_limit}'")
                self.hardening["memory_limit"] = self.memory_limit
            except duckdb.Error as exc:                       # pragma: no cover - build dependent
                self.hardening["errors"].append(f"memory_limit: {exc}")
                log.warning("sql gateway could not set memory_limit=%r: %s", self.memory_limit, exc)
        try:
            con.execute("SET enable_external_access=false")
        except duckdb.Error as exc:                           # pragma: no cover - build dependent
            self.hardening["errors"].append(f"enable_external_access: {exc}")
            log.warning("sql gateway could not disable external access: %s", exc)
        try:
            self.hardening["external_access_disabled"] = not bool(
                con.execute("SELECT current_setting('enable_external_access')").fetchone()[0])
        except duckdb.Error as exc:                           # pragma: no cover - build dependent
            self.hardening["errors"].append(f"current_setting: {exc}")

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

        # -- layer 2: walk DuckDB's AST for anything that reads outside the database.
        # FAIL CLOSED.  A statement whose AST we cannot read is a statement we cannot check for
        # replacement scans, and layers 1 and 3 do not cover those (a file read is a SELECT that
        # writes nothing).  So "the serializer could not tell us" is a refusal, not a shrug.
        scanned = True
        seen: set[str] = set()
        tables: set[str] = set()
        for piece in inner_texts:
            try:
                blob = con.execute("SELECT json_serialize_sql(?)", [piece]).fetchone()[0]
                tree = json.loads(blob)
            except Exception as exc:              # noqa: BLE001 - any serializer failure denies
                return GateDecision(
                    False, types, ast_scanned=False,
                    reason=(f"refused: DuckDB could not serialize this statement's parse tree "
                            f"({str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__}), "
                            f"so the anatid sql tool cannot check it for file and URL reads. "
                            f"Rewrite it as a plain SELECT over the anatid tables."),
                )
            if not isinstance(tree, dict) or tree.get("error"):
                detail = tree.get("error_message") if isinstance(tree, dict) else "not an object"
                return GateDecision(
                    False, types, ast_scanned=False,
                    reason=(f"refused: DuckDB could not serialize this statement's parse tree "
                            f"({detail or 'unknown serialization error'}), so the anatid sql tool "
                            f"cannot check it for file and URL reads. Rewrite it as a plain "
                            f"SELECT over the anatid tables."),
                )
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
        point -- and :class:`SqlTimeout` when the statement outran ``timeout``.  DuckDB errors
        from a legitimately-allowed query propagate as ``duckdb.Error``.
        """
        decision = self.inspect(sql)
        if not decision.allowed:
            raise SqlNotAllowed(decision.reason or "rejected")

        cap = self.max_rows if limit is None else max(1, min(int(limit), self.max_rows))
        con = self._factory().cursor()
        timer: threading.Timer | None = None
        try:
            con.execute("BEGIN TRANSACTION READ ONLY")
            try:
                # DuckDB has no statement_timeout: interrupt() from a watchdog thread is the
                # supported way to stop a running query, and it raises InterruptException in
                # the thread that submitted it.  Cancelled the moment the rows are in hand.
                if self.timeout:
                    timer = threading.Timer(self.timeout, con.interrupt)
                    timer.daemon = True
                    timer.start()
                result = con.execute(sql)
                columns = [d[0] for d in (result.description or [])]
                rows = result.fetchmany(cap)
                truncated = len(result.fetchmany(1)) > 0
            except duckdb.InterruptException as exc:
                raise SqlTimeout(
                    f"query exceeded the anatid sql tool's {self.timeout:g}s limit and was "
                    f"interrupted; nothing was written. Narrow it (add a WHERE, a LIMIT, or an "
                    f"aggregate) and try again."
                ) from exc
            finally:
                if timer is not None:
                    timer.cancel()
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
            "limits": {"max_rows": cap, "timeout_seconds": self.timeout,
                       "memory_limit": self.hardening.get("memory_limit"),
                       "external_access_disabled": self.hardening.get(
                           "external_access_disabled", False)},
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
