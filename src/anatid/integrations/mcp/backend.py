"""Where ``anatid-mcp`` gets its memory: an embedded handle, or a running ``anatid-server``.

The MCP tools in :mod:`anatid.integrations.mcp.server` call verbs on one object, ``db``.  Until
0.3.0 that object was always an :class:`anatid.Anatid` opened on a file, which made every MCP
client a separate writer process on that file, and DuckDB gives one process exclusive use of a
database file.  Measured on duckdb 1.5.5 (``docs/server.md``, section 1), the second process is
refused even when it only wants to read::

    holder=read-write  second=read-write  -> IO Error: Could not set lock on file
    holder=read-write  second=read-only   -> IO Error: Could not set lock on file

So Claude Desktop and Claude Code pointed at the same ``ANATID_DB`` could not both be running,
and the second one to start died on that error with nothing to say about what to do instead.

:func:`open_backend` is the one place the choice of backend is made.  Given the resolved
:class:`~anatid.integrations.mcp.server.ServerConfig` it returns

* an :class:`anatid.Anatid` when the configuration names a file (``--db`` / ``ANATID_DB``, or
  the default path).  This is the embedded profile and it is unchanged: one process, no socket,
  the fastest arrangement anatid has.
* an :class:`anatid.server.client.AnatidClient` when it names a running server (``--socket`` /
  ``ANATID_SOCKET``, or ``--http-url`` plus a token).  This is the server profile: one
  ``anatid-server`` process holds the file and every ``anatid-mcp`` talks to it, reads and
  writes alike, so any number of MCP clients share one memory.

Both objects answer the same verbs with the same keywords and return the same types (0.3.0 made
the client a drop-in; ``tests/test_server_client.py`` compares the two signature by signature
and ``tests/test_mcp_over_server.py`` runs every MCP tool against both and compares the
answers), so the tool layer does not branch on which one it was given.

What the client does not carry is the part of the embedded handle that is about a file rather
than about memory.  The tool layer reads three of those: ``read_only`` decides whether the write
tools are registered, and ``path`` and ``expand_path`` are reported by ``stats``.
:class:`ServerHandle` is the client with those three filled in, and it is what the socket and
HTTP paths return.  It adds no verb.

The SQL escape hatch does not cross the wire.  It runs statements on the file's own DuckDB
connection, and over a socket the only such connection is in the server process, so
``--enable-sql`` together with a socket is refused at startup rather than registered as a tool
that fails on first use.  Two more settings belong to the process that holds the file and are
refused with a socket for the same reason: the embedder (``ANATID_EMBED_*``), because the
client forwards verbs and does not embed, and the extractor (``ANATID_EXTRACT_*``), because the
ingest pipeline runs several verbs in one transaction on the file's own connection.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from anatid import Anatid
from anatid.server.client import AnatidClient

from .embedding import (
    ENV_BASE_URL,
    ENV_HASH,
    ENV_MODEL,
    EmbedderConfigError,
    embedder_from_config,
)
from .ingest import ENV_EXTRACT_BASE_URL, ENV_EXTRACT_MODEL, extraction_configured

if TYPE_CHECKING:  # pragma: no cover - typing only; the runtime import would be circular
    from .server import ServerConfig

__all__ = [
    "BackendConfigError",
    "BackendError",
    "DatabaseLocked",
    "LOCK_MARKER",
    "ServerHandle",
    "check_backend_config",
    "check_ingest_backend",
    "check_sql_backend",
    "embedder_for",
    "open_backend",
    "resolve_token",
    "sharing_recipe",
    "start_hint",
    "suggested_socket_path",
]

#: The substring DuckDB puts in its ``IOException`` when another process holds the file.  The
#: same test :mod:`anatid.server.cli` uses to decide whether a restore has to wait.
LOCK_MARKER = "Could not set lock on file"


# --------------------------------------------------------------------------- errors


class BackendError(RuntimeError):
    """The configured backend cannot be used, for a reason the operator has to act on.

    :func:`anatid.integrations.mcp.server.main` prints the message on stderr and exits 2, the
    way it does for :class:`~anatid.integrations.mcp.server.InsecureTransport`.
    """


class BackendConfigError(BackendError):
    """The configuration names two backends, or asks one of them for something it cannot do."""


class DatabaseLocked(BackendError):
    """``--db`` named a file another process holds.

    The message is the explanation and the way out: DuckDB's lock rule in one paragraph, then
    the ``anatid-server start`` command that makes one process hold the file and the
    ``anatid-mcp --socket`` command every client runs instead of opening it.
    """

    def __init__(self, path: str, *, tenant: int, detail: str) -> None:
        self.path = path
        self.tenant = tenant
        self.detail = detail
        self.socket_path = suggested_socket_path()
        start, connect = sharing_recipe(path, tenant=tenant, socket_path=self.socket_path)
        super().__init__(
            f"another process holds {path}. DuckDB gives one process exclusive use of a "
            f"database file and refuses a second opener even for reading, so two anatid-mcp "
            f"processes cannot share a file by both opening it: the second one always fails "
            f"here, and a read-only open does not get around it. The way to share one memory "
            f"between clients is anatid's server profile: one anatid-server process holds the "
            f"file and every anatid-mcp talks to it over a Unix socket, reads and writes alike. "
            f"Start the server once:\n\n"
            f"    {start}\n\n"
            f"then start every anatid-mcp against the socket instead of the file:\n\n"
            f"    {connect}\n\n"
            f"In a client's JSON config block that is ANATID_SOCKET={self.socket_path} in place "
            f"of ANATID_DB. If the process holding the file is already an anatid-server, skip "
            f'the first command and use its socket. See docs/mcp.md, "Sharing one memory '
            f'between clients".\n'
            f"DuckDB's own message: {detail}"
        )


# --------------------------------------------------------------------------- the recipe


def suggested_socket_path() -> str:
    """A socket path that works when pasted.

    ``$XDG_RUNTIME_DIR/anatid/anatid.sock`` where a session runtime directory exists, which is
    the per-user place for a socket on a Linux desktop or under systemd.  Otherwise
    ``/tmp/anatid/anatid.sock``: short enough for ``sockaddr_un`` on every platform (104 bytes
    on macOS) and writable by an unprivileged user, which ``/run/anatid`` is not.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return os.path.join(runtime, "anatid", "anatid.sock")
    return "/tmp/anatid/anatid.sock"


def sharing_recipe(
    db_path: str, *, tenant: int = 0, socket_path: str | None = None
) -> tuple[str, str]:
    """The two commands that share one file between MCP clients.

    The first starts the process that holds the file; the second is what every client runs
    instead of ``anatid-mcp --db``.  Paths are shell-quoted so a macOS ``Application Support``
    path pastes correctly.
    """
    sock = socket_path or suggested_socket_path()
    start = (
        f"anatid-server start --socket {shlex.quote(sock)} --db {shlex.quote(db_path)} "
        f"--tenant {int(tenant)}"
    )
    connect = f"anatid-mcp --socket {shlex.quote(sock)} --tenant {int(tenant)}"
    return start, connect


def start_hint(config: ServerConfig) -> str | None:
    """What to run when nothing listens on the configured socket, or None for a file.

    The client's own error names the socket and the Python constructor; an operator running
    ``anatid-mcp`` wants the command.  The memory file is the server's to name, so it is left
    as a placeholder rather than guessed from ``ANATID_DB``, which a socket client does not use.
    """
    if config.socket is None:
        return None
    return (
        f"a server for that socket is started with: anatid-server start --socket "
        f"{shlex.quote(config.socket)} --db <the memory file> --tenant {int(config.tenant)}"
    )


# --------------------------------------------------------------------------- the remote handle


class ServerHandle(AnatidClient):
    """An :class:`~anatid.server.client.AnatidClient` with the three handle attributes the MCP
    tool layer reads.  It adds no verb.

    ``read_only``
        This MCP server's own setting, and what decides whether the write tools are registered
        here.  It restricts nothing on the server, which authenticates its own callers; a
        principal the server lets write can still write through another client.
    ``path``
        Where this process is connected, as in ``unix:/run/anatid/anatid.sock``.  The file is
        the server's; its path is in ``info().extras["path"]``.
    ``expand_path``
        The graph-expansion path the server's own handle forecasts for its next read
        (``"sql"``, ``"csr"`` or ``"extension"``), one ``info`` round trip away, or
        ``"unknown"`` from a server whose ``info`` does not report it.
    """

    def __init__(self, *, read_only: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self._read_only = bool(read_only)

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def path(self) -> str:
        return self.address

    @property
    def expand_path(self) -> str:
        value = self.info().extras.get("expand_path")
        return "unknown" if value is None else str(value)

    def __repr__(self) -> str:
        return (
            f"<ServerHandle {self.address} tenant={self.namespace.tenant_id}"
            f"{' read_only' if self._read_only else ''}{' closed' if self.closed else ''}>"
        )


# --------------------------------------------------------------------------- configuration checks


def check_backend_config(config: ServerConfig) -> None:
    """Refuse a configuration that names two backends, or asks one for what it cannot do.

    Raises :class:`BackendConfigError`.  Nothing is opened or connected before this passes, so
    a refusal creates no file and touches no socket.
    """
    if config.socket is not None and config.http_url is not None:
        raise BackendConfigError(
            f"--socket {config.socket!r} and --http-url {config.http_url!r} name two servers. "
            f"Pass one: the Unix socket for a server on this machine, the HTTP URL for one on "
            f"another host."
        )
    remote = config.socket if config.socket is not None else config.http_url
    if remote is None:
        return
    flag = (
        "--socket / ANATID_SOCKET" if config.socket is not None else "--http-url / ANATID_HTTP_URL"
    )
    if config.db_explicit:
        raise BackendConfigError(
            f"--db / ANATID_DB ({config.db!r}) and {flag} ({remote!r}) were both given, and "
            f"they are two different backends: --db opens the file in this process, and a "
            f"socket client does not open a file at all, because the server holding the socket "
            f"already has it. Drop one. To share this file between clients, leave --db out "
            f"here and give it to the server instead: "
            f"{sharing_recipe(config.db, tenant=config.tenant)[0]}"
        )
    if config.sql_tool:
        raise BackendConfigError(
            f"--enable-sql / ANATID_ENABLE_SQL cannot be combined with {flag}: the SQL escape "
            f"hatch runs statements on the file's own DuckDB connection, and over a socket the "
            f"only such connection is inside the server process. Leave it unset for an "
            f"anatid-mcp that talks to a server; it is available to an anatid-mcp that opens the "
            f"file itself (--db)."
        )
    if config.http_url is not None and not (config.token or config.token_file):
        raise BackendConfigError(
            f"--http-url {config.http_url!r} needs a bearer token: --token / ANATID_TOKEN, or "
            f"--token-file / ANATID_TOKEN_FILE. anatid-server refuses to serve HTTP without "
            f"one, and a port with no token is reachable by every user of the machine it is "
            f"bound on. The Unix socket needs no token; use --socket for a server on this "
            f"machine."
        )
    env = config.env
    if config.embed_hash or any(
        (env.get(name) or "").strip() for name in (ENV_BASE_URL, ENV_MODEL, ENV_HASH)
    ):
        raise BackendConfigError(
            f"an embedder ({ENV_BASE_URL} / {ENV_MODEL} / {ENV_HASH} / --embed-hash) cannot be "
            f"combined with {flag}: embedding happens in the process that holds the file, and a "
            f"socket client forwards verbs without embedding. Drop the embedding settings for "
            f"an anatid-mcp that talks to a server; they apply to an anatid-mcp that opens the "
            f"file itself (--db)."
        )
    if extraction_configured(env):
        raise BackendConfigError(
            f"the ingest tools ({ENV_EXTRACT_BASE_URL} / {ENV_EXTRACT_MODEL}) cannot be combined "
            f"with {flag}: the ingestion pipeline applies a whole patch in one transaction on "
            f"the file's own connection, which only the server process has. Drop the "
            f"extraction settings for an anatid-mcp that talks to a server; they apply to an "
            f"anatid-mcp that opens the file itself (--db)."
        )


def check_sql_backend(db: Anatid | AnatidClient, config: ServerConfig) -> None:
    """The half of :func:`check_backend_config` that :func:`build_server` can check itself.

    A program that connects its own :class:`~anatid.server.client.AnatidClient` and passes it to
    ``build_server`` with ``sql_tool=True`` never went through :func:`open_backend`, so the
    refusal has to happen here as well: the gateway would otherwise be built on a
    ``db.connection`` the client does not have.
    """
    if config.sql_tool and not isinstance(db, Anatid):
        raise BackendConfigError(
            "the SQL escape hatch needs an embedded anatid.Anatid handle: it runs statements on "
            "the file's own DuckDB connection, which an AnatidClient does not have because the "
            "file is in the server process. Build the server with sql_tool=False, or give it "
            "the embedded handle."
        )


def check_ingest_backend(db: Anatid | AnatidClient, config: ServerConfig) -> None:
    """The ingest half of :func:`check_backend_config`, for a ``build_server`` called directly.

    A program that connects its own :class:`~anatid.server.client.AnatidClient` and passes it to
    ``build_server`` with an extractor never went through :func:`open_backend`, so the refusal
    has to happen here as well.
    """
    if not isinstance(db, Anatid):
        raise BackendConfigError(
            "the ingest tools need an embedded anatid.Anatid handle: MemoryPatch.apply commits "
            "a whole patch in one transaction on the file's own connection, and an AnatidClient "
            "has neither, because the file is in the server process. Build the server without "
            "an extractor, or give it the embedded handle."
        )


def embedder_for(config: ServerConfig) -> Any:
    """The embedder ``ANATID_EMBED_*`` / ``--embed-hash`` ask for, or None.

    :class:`~anatid.integrations.mcp.embedding.EmbedderConfigError` (half an endpoint, or the
    hash stand-in together with an endpoint) becomes a :class:`BackendConfigError`, so ``main``
    reports it the way it reports every other configuration refusal.
    """
    try:
        return embedder_from_config(
            embedding_dim=config.embedding_dim, env=config.env, embed_hash=config.embed_hash
        )
    except EmbedderConfigError as exc:
        raise BackendConfigError(str(exc)) from exc


def resolve_token(config: ServerConfig) -> str | None:
    """The bearer token for ``--http-url``: ``--token`` first, else the first token in the file.

    The file format is ``anatid-server``'s: one token per line, ``# comments`` and blank lines
    ignored, the token being the first whitespace-separated field so that a server's own token
    file (``<token> [name=NAME] [tenants=1,2] [read-only]``) can be reused as is.
    """
    if config.token:
        return config.token
    if not config.token_file:
        return None
    path = Path(config.token_file).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BackendConfigError(f"cannot read the token file {path}: {exc}") from exc
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped.split()[0]
    raise BackendConfigError(f"the token file {path} holds no token")


# --------------------------------------------------------------------------- the one decision


def open_backend(config: ServerConfig) -> Anatid | AnatidClient:
    """Open what the configuration names: an embedded handle for a file, a client for a server.

    ``--socket`` / ``ANATID_SOCKET``
        Connect to that Unix socket and return a :class:`ServerHandle`.  The connect sends one
        ``health`` probe, so a server that is not running is reported here as
        :class:`~anatid.server.client.ServerUnavailable`, with the socket path and what to do
        about it, rather than on the first tool call.
    ``--http-url`` plus a token
        The same over the server's HTTP listener, for a server on another host.
    otherwise
        ``Anatid.open`` on ``--db`` / ``ANATID_DB`` (default ``~/.anatid/memory.anatid``), with
        the embedder ``ANATID_EMBED_*`` or ``--embed-hash`` names, if any, so ``remember``
        stores a vector and ``recall`` runs the vector arm without any tool changing.  A file
        another process already holds raises :class:`DatabaseLocked`, whose message is the
        two-command recipe for sharing it instead.

    :func:`check_backend_config` runs first, so a configuration that names both a file and a
    server, or asks a server for the SQL escape hatch, is refused before anything is opened.
    """
    check_backend_config(config)
    if config.socket is not None:
        return ServerHandle.connect(
            Path(config.socket).expanduser(),
            tenant=config.tenant,
            read_only=config.read_only,
        )
    if config.http_url is not None:
        return ServerHandle.connect_http(
            config.http_url,
            tenant=config.tenant,
            token=resolve_token(config),
            read_only=config.read_only,
        )
    path = config.resolved_db()
    embedder = embedder_for(config)
    try:
        return Anatid.open(
            path,
            tenant=config.tenant,
            embedding_dim=config.embedding_dim,
            read_only=config.read_only,
            embedder=embedder,
        )
    except duckdb.IOException as exc:
        if LOCK_MARKER in str(exc):
            raise DatabaseLocked(path, tenant=config.tenant, detail=str(exc).strip()) from exc
        raise
