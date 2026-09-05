"""The operator surface for the anatid server: start, stop, status, backup, restore, doctor.

This is the command an init system runs and the command a human runs at three in the morning.
Everything here is a thin, testable shell over :mod:`anatid.server.server`; no policy is decided
in this file that is not visible in ``--help``.

Why a CLI at all, given that ``serve()`` is four lines of Python: because the parts an operator
gets wrong are not the four lines.  They are the socket that ends up world-writable, the HTTP
port that ends up on 0.0.0.0 with no token, the ``TimeoutStopSec`` that is shorter than the
drain timeout so systemd kills the process in the middle of a write, and the backup that was
taken with ``cp`` while the server held the file.  Each of those has an answer here.

The subcommands
---------------
``start``
    Run the server until SIGTERM.  ``--check`` validates the configuration and exits without
    binding anything or opening a file, so a deployment fails at deploy time rather than at
    the first request.
``stop``
    Ask a running server on this machine to drain and exit, and wait until it has.
``status``
    Health, readiness and queue counters, for a human or, with ``--json``, for a monitor.
``backup``
    A consistent copy of one tenant's file.  Online through a running server (which is the only
    way, while it holds the file), or offline against the files directly.
``restore``
    Put a backup back, offline, with the previous file kept aside rather than overwritten.
``doctor``
    :meth:`anatid.Anatid.doctor` over the wire or against the files directly.

Exit codes, because a script reads them
---------------------------------------
==  =========================================================================================
0   the command did what it was asked
1   it failed at run time: could not connect, the server returned an error, a copy failed
2   the configuration or the arguments are wrong; nothing was started and nothing was written
3   the server answered, and the answer is bad news: not ready, or doctor found errors
==  =========================================================================================

``start`` returns 0 when its shutdown drained cleanly and 1 when the drain timed out with
writes still queued, because writes that were accepted and then abandoned are a failure an
operator has to see.

What this file deliberately does not do
---------------------------------------
It does not decide anything about tenancy that the server does not already enforce, it does not
open a database in ``--check`` mode (a check that creates a file is not a check), and it does
not print a token, ever.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as _dt
import json
import logging
import os
import shutil
import signal
import socket as _socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .. import __version__ as _anatid_version
from ..database import Anatid, DatabasePool
from ..errors import AnatidError
from ..schema import SCHEMA_VERSION, current_version, quote_ident
from . import protocol
from .auth import (
    AllowAllAuthenticator,
    BearerTokenAuthenticator,
    ConnectionContext,
    Principal,
    UnixPeerAuthenticator,
    check_bind_address,
)
from .protocol import (
    AuthenticationError,
    AuthorizationError,
    ProtocolError,
    Request,
    Response,
)
from .backup import BackupCoordinator
from .queue import DrainReport, QueueStats
from .server import AnatidServer, ServerConfig, connect_unix

__all__ = [
    "main",
    "TransportAuthenticator",
    "OperatorServer",
    "BACKUP_VERB",
    "EXIT_OK",
    "EXIT_FAILED",
    "EXIT_CONFIG",
    "EXIT_UNHEALTHY",
]

log = logging.getLogger("anatid.server.cli")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_UNHEALTHY = 3

#: The verb an operator client sends to ask a running server for a backup.  It is NOT in
#: :data:`anatid.server.server.VERBS`, and that is deliberate: a server that was not started with
#: ``--backup-dir`` must not have it at all, and a plain :class:`AnatidServer` must refuse it as
#: an unknown verb.  :class:`OperatorServer` answers it and nothing else does.
BACKUP_VERB = "backup"

#: ``sun_path`` in ``sockaddr_un`` is a fixed-size array: 104 bytes on the BSDs and macOS, 108 on
#: Linux, including the terminating NUL.  A longer path fails at ``bind()`` with a message that
#: does not mention length, so it is checked here instead, where the fix is obvious.
SOCKADDR_UN_MAX = 103 if sys.platform in ("darwin", "freebsd") else 107


# --------------------------------------------------------------------------- small output


def _out(text: str = "") -> None:
    """Write one line to stdout.  A function rather than ``print`` so the library keeps its
    no-``print`` rule and so a test can capture the same call the operator sees."""
    sys.stdout.write(f"{text}\n")


def _err(text: str = "") -> None:
    """Write one line to stderr."""
    sys.stderr.write(f"{text}\n")


class ConfigProblem(Exception):
    """The arguments describe a server that should not be started.  Exit code 2."""


# --------------------------------------------------------------------------- authentication


class TransportAuthenticator:
    """One authenticator per transport, because the two transports authenticate differently.

    A Unix socket is authenticated by the permissions on the socket and its directory, which the
    server sets to 0600 in a 0700 directory.  An HTTP listener has no such thing and needs a
    bearer token.  :class:`AnatidServer` holds a single authenticator for both, so a server given
    a :class:`~anatid.server.auth.BearerTokenAuthenticator` would refuse every Unix connection
    for want of an ``Authorization`` header.  That is the trap this class exists to remove:
    ``anatid-server start --socket S --http 127.0.0.1:8787 --token-file F`` has to serve both.

    ``requires_token`` reports the HTTP half, since that is the question
    :func:`~anatid.server.auth.check_bind_address` is asking: whether the listener that faces the
    network needs a secret.
    """

    def __init__(self, *, unix: Any = None, http: Any = None) -> None:
        self.unix = unix
        self.http = http

    @property
    def requires_token(self) -> bool:
        return bool(self.http is not None and getattr(self.http, "requires_token", False))

    def authenticate(self, ctx: ConnectionContext) -> Principal:
        chosen = self.http if ctx.transport == "http" else self.unix
        if chosen is None:
            raise AuthenticationError(
                f"this server has no authenticator for the {ctx.transport} transport, so it "
                f"cannot attribute this connection to anyone"
            )
        return chosen.authenticate(ctx)

    def __repr__(self) -> str:
        return (
            f"TransportAuthenticator(unix={type(self.unix).__name__}, "
            f"http={type(self.http).__name__})"
        )


def _parse_token_line(line: str, lineno: int) -> tuple[str, Principal]:
    """One line of a token file into ``(token, principal)``.

    The grammar is one line per token::

        <token> [name=<name>] [tenants=1,2,3] [read-only]

    A token with no ``tenants=`` may name every tenant the server holds.  Blank lines and lines
    beginning with ``#`` are ignored.  The token itself is the first field and may not contain
    whitespace, which is the only restriction on it.
    """
    fields = line.split()
    token = fields[0]
    name = f"token{lineno}"
    tenants: frozenset[int] | None = None
    read_only = False
    for field in fields[1:]:
        if field == "read-only":
            read_only = True
            continue
        key, sep, value = field.partition("=")
        if not sep:
            raise ConfigProblem(
                f"line {lineno} of the token file: {field!r} is not 'read-only' and not "
                f"'key=value'. The fields after a token are name=, tenants= and read-only."
            )
        if key == "name":
            name = value
        elif key == "tenants":
            try:
                tenants = frozenset(int(t) for t in value.split(",") if t)
            except ValueError as exc:
                raise ConfigProblem(
                    f"line {lineno} of the token file: tenants={value!r} is not a comma "
                    f"separated list of integers"
                ) from exc
        else:
            raise ConfigProblem(
                f"line {lineno} of the token file: unknown field {key!r}. The fields after a "
                f"token are name=, tenants= and read-only."
            )
    return token, Principal(name=name, tenants=tenants, transport="http", read_only=read_only)


def read_token_file(path: str | os.PathLike[str]) -> dict[str, Principal]:
    """Parse a token file into the mapping :class:`BearerTokenAuthenticator` takes.

    Warns on stderr when the file is readable by anyone but its owner.  It warns rather than
    refuses because a secret mounted into a container is routinely 0444 and owned by root, and a
    server that will not start in that situation is a worse outcome than one that says so.
    """
    p = Path(path).expanduser()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigProblem(f"cannot read the token file {p}: {exc}") from exc
    with contextlib.suppress(OSError):
        mode = p.stat().st_mode & 0o777
        if mode & 0o077:
            _err(
                f"anatid-server: warning: {p} is mode {mode:04o}, so users other than its owner "
                f"can read the tokens in it. chmod 600 {p}"
            )
    tokens: dict[str, Principal] = {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        token, principal = _parse_token_line(text, lineno)
        if token in tokens:
            raise ConfigProblem(f"line {lineno} of the token file repeats an earlier token")
        tokens[token] = principal
    if not tokens:
        raise ConfigProblem(
            f"{p} holds no tokens. A token file with nothing in it would refuse every HTTP "
            f"request, which is a configuration error worth failing on at startup."
        )
    return tokens


def _build_authenticator(args: argparse.Namespace) -> TransportAuthenticator:
    """The authenticator a ``start`` invocation implies, with no side effects."""
    unix: Any = None
    if args.socket:
        if args.allow_uid:
            unix = UnixPeerAuthenticator(allow_uids=[int(u) for u in args.allow_uid])
        else:
            unix = AllowAllAuthenticator(name="local")
    http: Any = None
    if args.http:
        if args.token_file:
            http = BearerTokenAuthenticator(read_token_file(args.token_file))
        elif args.http_no_auth:
            http = AllowAllAuthenticator(name="anonymous-http")
        else:
            raise ConfigProblem(
                "--http needs --token-file. Without a token, anyone who can reach the port "
                "can read and write every tenant this server holds, and on a shared machine "
                "that includes every other local user even when the bind is 127.0.0.1. Pass "
                "--http-no-auth if this port is genuinely unreachable by anyone else."
            )
    return TransportAuthenticator(unix=unix, http=http)


# --------------------------------------------------------------------------- the server


class OperatorServer(AnatidServer):
    """An :class:`AnatidServer` that also answers the ``backup`` verb, when configured to.

    Backup has to go through the server process.  Measured on duckdb 1.5.5, a process holding a
    file read-write excludes every other process from it, read-only attempts included, so
    nothing outside this process can copy a tenant's file while the server runs.
    :meth:`AnatidServer.backup_tenant` does the work and is reachable from inside the process;
    this makes it reachable from the socket, which is what an operator has.

    Two restrictions, both because a backup writes a file and a file is not something a client
    should get to place:

    * the destination is a single file name inside ``--backup-dir``, never a path.  A client
      cannot name a directory, cannot escape with ``..``, and cannot write a dotfile.
    * the caller must be an administrator, meaning a principal with no tenant restriction and
      not read-only.  Over a Unix socket that is the default principal.  Over HTTP it is a token
      with no ``tenants=`` field.  A token scoped to one tenant can read and write that tenant's
      memories and cannot make the server write files.

    Without ``--backup-dir`` the verb is refused, and a plain :class:`AnatidServer` does not have
    it at all: the verb is not in :data:`~anatid.server.server.VERBS`, so an unconfigured server
    answers "unknown verb" exactly as it does for anything else it does not implement.
    """

    def __init__(self, *, backup_dir: str | os.PathLike[str] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.backup_dir = Path(backup_dir).expanduser() if backup_dir is not None else None

    def run_read(self, request: Request, principal: Principal) -> Any:
        # Intercepted before the verb table rather than added to it, so that this verb exists
        # only on a server that was started with somewhere to put the file.  There is still no
        # attribute lookup on client text here: one literal name, one method.
        if request.verb == BACKUP_VERB:
            return self._run_backup(request, principal)
        return super().run_read(request, principal)

    def _destination(self, name: Any) -> Path:
        base = self.backup_dir
        if base is None:  # pragma: no cover - _run_backup checks first
            raise ProtocolError("this server has no backup directory")
        if not isinstance(name, str) or not name:
            raise ProtocolError("backup needs a 'name' argument: the file name to write")
        if name != os.path.basename(name) or name in (".", "..") or name.startswith("."):
            raise ProtocolError(
                f"backup name {name!r} must be a plain file name with no directory part and no "
                f"leading dot. The server decides which directory backups go in; a client "
                f"chooses the name inside it."
            )
        return base / name

    def _run_backup(self, request: Request, principal: Principal) -> dict[str, Any]:
        if self.backup_dir is None:
            raise ProtocolError(
                "this server was started without --backup-dir, so it does not take backup "
                "requests. Restart it with --backup-dir DIR, or stop it and run "
                "'anatid-server backup' against the files."
            )
        if not principal.unrestricted:
            raise AuthorizationError(
                f"{principal.name} is scoped to particular tenants; backup is an administrative "
                f"verb and needs a principal with no tenant restriction, because it makes the "
                f"server write a file to its own disk"
            )
        principal.require_write(BACKUP_VERB)
        if "tenant" in request.args:
            raise ProtocolError(
                "'tenant' belongs in the request envelope, not in args, for backup as for "
                "every other verb"
            )
        tenant_id = principal.require(request.tenant)
        if self.pool is None:
            raise AnatidError(
                "this server holds one shared file, so it has no per-tenant file to copy. "
                "Back it up by stopping the server and copying the file, or run the server on "
                "a DatabasePool."
            )
        dest = self._destination(request.args.get("name"))
        overwrite = bool(request.args.get("overwrite", False))
        # Before the drain, not after.  A backup that is going to be refused should not first
        # pause the tenant it names.  ProtocolError rather than the ConfigProblem underneath,
        # because a destination that already exists is a bad argument like a bad name, and the
        # HTTP mapping answers 400 for those and 500 for anything it does not recognise.
        try:
            _refuse_existing(dest, overwrite=overwrite)
        except ConfigProblem as exc:
            raise ProtocolError(str(exc)) from exc
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.backup_dir, 0o700)
        # A barrier, not a drain.  This used to call WriteQueue.drain_tenant, which waits for the
        # tenant's queue to empty and then RETURNS, stopping nothing: a write submitted between
        # the drain returning and the copy starting commits into the copy, so the boundary was
        # the weaker "snapshot" one however long the drain waited.  BackupCoordinator takes the
        # tenant's single serving slot for the length of the copy instead, so what the report
        # claims is what the copy has.  Other tenants are never paused either way.
        result = BackupCoordinator(self, acquire_timeout=self.config.shutdown_timeout).backup(
            dest, tenant=tenant_id, quiesce=True, overwrite=overwrite, verify=False
        )
        report = result.one
        log.info(
            "backed tenant %s up to %s (%d bytes, %s, tenant paused %.3fs)",
            tenant_id,
            report.path,
            report.bytes,
            report.guarantee.value,
            report.quiesced_for,
        )
        return {
            "tenant": tenant_id,
            "path": str(report.path),
            "bytes": report.bytes,
            "guarantee": report.guarantee.value,
            "quiesced": report.quiesced,
            "paused_s": round(report.quiesced_for, 3),
            "waited_s": round(report.waited_for, 3),
        }


# --------------------------------------------------------------------------- the client


class OperatorClient:
    """A blocking client for the subcommands that talk to a running server.

    One connection, reused across the two or three calls a subcommand makes, closed by
    :meth:`close` or by the context manager.  It exists so ``status`` over a Unix socket and
    ``status`` over HTTP are the same three calls and print the same thing.
    """

    def __init__(
        self,
        *,
        socket_path: str | os.PathLike[str] | None = None,
        http: tuple[str, int] | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if (socket_path is None) == (http is None):
            raise ConfigProblem("pass exactly one of --socket and --http")
        self.socket_path = socket_path
        self.http = http
        self.token = token
        self.timeout = timeout
        self._sock: _socket.socket | None = None
        self._conn: Any = None

    def __enter__(self) -> "OperatorClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    @property
    def address(self) -> str:
        if self.socket_path is not None:
            return f"unix:{self.socket_path}"
        host, port = self.http or ("", 0)
        return f"http://{host}:{port}"

    def call(self, verb: str, *, tenant: int = 0, **args: Any) -> Any:
        """Send one request and return its result, raising the server's error if there is one."""
        request = Request(verb=verb, tenant=int(tenant), args=args)
        response = self.send(request)
        return response.raise_for_status()

    def send(self, request: Request) -> Response:
        if self.socket_path is not None:
            return self._send_unix(request)
        return self._send_http(request)

    def _send_unix(self, request: Request) -> Response:
        if self._sock is None:
            path = os.fspath(self.socket_path or "")
            try:
                self._sock = connect_unix(path, timeout=self.timeout)
            except FileNotFoundError as exc:
                raise ConnectionError(f"no socket at {path}: nothing is listening there") from exc
            except ConnectionRefusedError as exc:
                raise ConnectionError(
                    f"{path} exists but refused the connection: the server that made it is gone "
                    f"and left the socket behind"
                ) from exc
        protocol.write_frame(self._sock, request)
        body = protocol.read_frame(self._sock)
        if body is None:
            raise ConnectionError("the server closed the connection without answering")
        return Response.decode(body)

    def _send_http(self, request: Request) -> Response:
        import http.client

        host, port = self.http or ("127.0.0.1", 8787)
        if self._conn is None:
            self._conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
        headers = {"content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        try:
            self._conn.request("POST", "/rpc", body=request.encode()[4:], headers=headers)
            reply = self._conn.getresponse()
            body = reply.read()
        except OSError as exc:
            self.close()
            raise ConnectionError(f"cannot reach http://{host}:{port}/rpc: {exc}") from exc
        return Response.decode(body)


def _client_from(args: argparse.Namespace) -> OperatorClient:
    token = args.token
    if not token and getattr(args, "token_file", None):
        tokens = read_token_file(args.token_file)
        token = next(iter(tokens))
    return OperatorClient(
        socket_path=args.socket,
        http=_parse_host_port(args.http) if args.http else None,
        token=token,
        timeout=args.timeout,
    )


# --------------------------------------------------------------------------- parsing helpers


def _parse_host_port(spec: str, *, default_port: int = 8787) -> tuple[str, int]:
    """``HOST:PORT``, ``HOST``, ``:PORT`` or ``[::1]:PORT`` into ``(host, port)``."""
    text = spec.strip()
    if not text:
        raise ConfigProblem("an empty --http address")
    if text.startswith("["):
        host, sep, rest = text[1:].partition("]")
        if not sep:
            raise ConfigProblem(f"--http {spec!r}: an IPv6 address needs a closing bracket")
        port_text = rest[1:] if rest.startswith(":") else ""
    elif text.count(":") == 1:
        host, _, port_text = text.partition(":")
    elif ":" in text:
        raise ConfigProblem(
            f"--http {spec!r}: an IPv6 address has to be bracketed, as in [::1]:8787"
        )
    else:
        host, port_text = text, ""
    host = host or "127.0.0.1"
    if not port_text:
        return host, default_port
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ConfigProblem(f"--http {spec!r}: {port_text!r} is not a port number") from exc
    if not 1 <= port <= 65535:
        raise ConfigProblem(f"--http {spec!r}: port {port} is outside 1..65535")
    return host, port


def _pool_from(args: argparse.Namespace) -> DatabasePool:
    kw: dict[str, Any] = {"max_open": args.max_open}
    if args.embedding_dim:
        kw["embedding_dim"] = int(args.embedding_dim)
    return DatabasePool(args.pool, **kw)


def _timestamp_name(tenant_id: int) -> str:
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"tenant-{tenant_id}-{stamp}.anatid"


def _humanize(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# --------------------------------------------------------------------------- start


def _server_config(args: argparse.Namespace) -> ServerConfig:
    """The :class:`ServerConfig` these arguments describe.  Raises :class:`ConfigProblem`."""
    http = _parse_host_port(args.http) if args.http else None
    try:
        return ServerConfig(
            socket_path=args.socket,
            http_host=http[0] if http else None,
            http_port=http[1] if http else 8787,
            max_depth=args.max_queue,
            batch_max=args.batch_max,
            workers=args.workers,
            read_workers=args.read_workers,
            idempotency=not args.no_idempotency,
            idempotency_ttl=args.idempotency_ttl,
            default_deadline=args.deadline,
            shutdown_timeout=args.drain_timeout,
            embeddings=args.embeddings,
            create_tenants=not args.no_create_tenants,
            tenants=tuple(int(t) for t in (args.tenant or ())),
        )
    except ValueError as exc:
        raise ConfigProblem(str(exc)) from exc


def _check_start(args: argparse.Namespace) -> tuple[ServerConfig, TransportAuthenticator]:
    """Validate a ``start`` invocation without opening a file or binding anything.

    Everything that can be known before the first byte of I/O is checked here: the transports,
    the socket path length and its directory, the bind address against the authenticator, the
    token file, the storage template, and every number that has a range.  A ``--check`` run
    prints what it resolved and exits; a real run does the same checks and then serves.
    """
    problems: list[str] = []
    if bool(args.db) == bool(args.pool):
        problems.append(
            "pass exactly one of --db PATH (one file, tenants are namespaces in it) and "
            "--pool TEMPLATE (one file per tenant, real isolation)"
        )
    if not args.socket and not args.http:
        problems.append("a server needs at least one transport: pass --socket, --http, or both")
    for name, value, low in (
        ("--max-queue", args.max_queue, 1),
        ("--batch-max", args.batch_max, 1),
        ("--workers", args.workers, 1),
        ("--read-workers", args.read_workers, 1),
        ("--max-open", args.max_open, 1),
    ):
        if value < low:
            problems.append(f"{name} must be at least {low}, got {value}")
    if args.drain_timeout < 0:
        problems.append(f"--drain-timeout must not be negative, got {args.drain_timeout}")
    if args.deadline is not None and args.deadline <= 0:
        problems.append(f"--deadline must be positive, got {args.deadline}")
    if args.embedding_dim is not None and args.embedding_dim < 1:
        problems.append(f"--embedding-dim must be positive, got {args.embedding_dim}")

    if args.socket:
        sock = Path(args.socket).expanduser()
        if len(str(sock)) > SOCKADDR_UN_MAX:
            problems.append(
                f"--socket {sock} is {len(str(sock))} characters; this platform's sockaddr_un "
                f"holds {SOCKADDR_UN_MAX}. Use a shorter path, /run/anatid/anatid.sock say."
            )
        parent = sock.parent
        existing = parent
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not os.access(existing, os.W_OK):
            problems.append(f"--socket {sock}: {existing} is not writable by this user")
    if args.pool:
        if "{tenant}" not in args.pool and "{label}" not in args.pool:
            problems.append(
                f"--pool {args.pool!r} must contain {{tenant}} (or {{label}}) so that each "
                f"tenant gets its own file"
            )
        else:
            root = Path(args.pool.split("{", 1)[0]).expanduser().parent
            existing = root
            while not existing.exists() and existing != existing.parent:
                existing = existing.parent
            if not os.access(existing, os.W_OK):
                problems.append(f"--pool: {existing} is not writable by this user")
    if args.db and args.db != ":memory:":
        parent = Path(args.db).expanduser().parent
        existing = parent
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not os.access(existing, os.W_OK):
            problems.append(f"--db: {existing} is not writable by this user")
    if args.backup_dir:
        base = Path(args.backup_dir).expanduser()
        existing = base
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not os.access(existing, os.W_OK):
            problems.append(f"--backup-dir: {existing} is not writable by this user")
    if args.backup_dir and args.db:
        problems.append(
            "--backup-dir needs --pool: a single shared file has no per-tenant file to copy"
        )

    authenticator = TransportAuthenticator()
    config: ServerConfig | None = None
    if not problems:
        try:
            authenticator = _build_authenticator(args)
            config = _server_config(args)
            if config.http_host is not None:
                check_bind_address(config.http_host, authenticator)
        except (ConfigProblem, ValueError) as exc:
            problems.append(str(exc))
    if problems or config is None:
        raise ConfigProblem("\n  ".join(["the configuration is not usable:", *problems]))
    return config, authenticator


def _describe(config: ServerConfig, args: argparse.Namespace) -> list[str]:
    storage = f"--pool {args.pool}" if args.pool else f"--db {args.db}"
    queue = f"max_depth={config.max_depth} batch_max={config.batch_max} workers={config.workers}"
    keys = "on" if config.idempotency else "off"
    backup = args.backup_dir or "not configured; the backup verb is refused"
    listed = ", ".join(str(t) for t in config.tenants) or "none configured"
    unlisted = "" if config.create_tenants else " (plus existing files; new ones refused)"
    return [
        f"storage        {storage}",
        f"transports     {', '.join(_endpoints(config)) or 'none'}",
        f"tenants        {listed}{unlisted}",
        f"write queue    {queue}",
        f"read threads   {config.read_workers}",
        f"idempotency    {keys} (ttl {config.idempotency_ttl:g}s)",
        f"deadline       {config.default_deadline}s default per request",
        f"drain timeout  {config.shutdown_timeout:g}s",
        f"embeddings     {config.embeddings}",
        f"backup dir     {backup}",
    ]


def _endpoints(config: ServerConfig) -> list[str]:
    out: list[str] = []
    if config.socket_path is not None:
        out.append(f"unix:{config.socket_path}")
    if config.http_host is not None:
        out.append(f"http://{config.http_host}:{config.http_port}")
    return out


def _write_pid_file(path: str | None) -> None:
    if not path:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{os.getpid()}\n", encoding="utf-8")


def _remove_pid_file(path: str | None) -> None:
    if not path:
        return
    with contextlib.suppress(OSError):
        Path(path).expanduser().unlink()


async def _serve(server: OperatorServer, args: argparse.Namespace) -> DrainReport:
    await server.start()
    _write_pid_file(args.pid_file)
    try:
        return await server.serve_forever()
    finally:
        _remove_pid_file(args.pid_file)


def cmd_start(args: argparse.Namespace) -> int:
    config, authenticator = _check_start(args)
    if args.check:
        _out("anatid-server: the configuration is usable.")
        for line in _describe(config, args):
            _out(f"  {line}")
        return EXIT_OK

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    pool = _pool_from(args) if args.pool else None
    database = None
    if args.db:
        kw: dict[str, Any] = {}
        if args.embedding_dim:
            kw["embedding_dim"] = int(args.embedding_dim)
        tenants = tuple(int(t) for t in (args.tenant or ()))
        database = Anatid.open(args.db, tenant=tenants[0] if tenants else 0, **kw)
    server = OperatorServer(
        pool=pool,
        database=database,
        config=config,
        authenticator=authenticator,
        backup_dir=args.backup_dir,
    )
    try:
        report = asyncio.run(_serve(server, args))
    finally:
        if database is not None:
            database.close()
        if pool is not None:
            pool.close_all()
    if report.abandoned or report.timed_out:
        _err(
            f"anatid-server: the drain did not finish: {report.abandoned} queued writes were "
            f"abandoned after {report.duration_s:.1f}s ({report.abandoned_by_tenant}). Those "
            f"clients were told the server was shutting down; the writes did not happen."
        )
        return EXIT_FAILED
    _out(f"anatid-server: drained {report.drained} writes in {report.duration_s:.1f}s and exited.")
    return EXIT_OK


# --------------------------------------------------------------------------- stop


def _pid_of(args: argparse.Namespace) -> int:
    """The pid to signal: from ``--pid-file`` if there is one, otherwise from the server.

    Asking the server over its own socket is the more reliable of the two, because a pid file
    outlives the process that wrote it.  The pid comes from the process at the other end of a
    socket this user can open, which on a 0600 socket in a 0700 directory is a process this user
    already controls.
    """
    if args.pid_file:
        text = Path(args.pid_file).expanduser().read_text(encoding="utf-8").strip()
        try:
            return int(text)
        except ValueError as exc:
            raise ConfigProblem(f"{args.pid_file} does not hold a pid: {text!r}") from exc
    with _client_from(args) as client:
        health = client.call("health")
    return int(getattr(health, "pid", 0) or 0)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stopped(args: argparse.Namespace, pid: int) -> bool:
    """Whether the server has finished stopping.

    Not simply "the pid has gone", because that is not a signal a waiting process can rely on: a
    process that has exited but whose parent has not reaped it yet still answers
    ``os.kill(pid, 0)``, so a stop that watched only the pid would sit out its whole timeout
    after a perfectly clean shutdown and then report failure.

    The server's own marks are better, and they are what an operator is actually waiting for.
    :meth:`AnatidServer.shutdown` releases every open file and THEN unlinks the socket, so a
    socket that is gone means the drain finished and the files are free for the next process.
    ``start`` removes its pid file at the same point, for a ``stop`` that was given one and no
    transport.  The pid check stays as the last of the three, for a server that had neither.
    """
    if args.socket and not Path(args.socket).expanduser().exists():
        return True
    if args.pid_file and not Path(args.pid_file).expanduser().exists():
        return True
    return not _alive(pid)


def cmd_stop(args: argparse.Namespace) -> int:
    try:
        pid = _pid_of(args)
    except (ConnectionError, FileNotFoundError) as exc:
        _out(f"anatid-server: nothing to stop ({exc}).")
        return EXIT_OK
    if pid <= 0:
        _err("anatid-server: the server did not report a pid, so there is nothing to signal")
        return EXIT_FAILED
    if not _alive(pid):
        _out(f"anatid-server: pid {pid} is not running; nothing to stop.")
        _remove_pid_file(args.pid_file)
        return EXIT_OK
    os.kill(pid, signal.SIGTERM)
    _out(f"anatid-server: sent SIGTERM to pid {pid}; waiting up to {args.timeout:g}s to drain.")
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if _stopped(args, pid):
            _out(f"anatid-server: pid {pid} stopped; its files are released.")
            _remove_pid_file(args.pid_file)
            return EXIT_OK
        time.sleep(0.05)
    if args.force:
        os.kill(pid, signal.SIGKILL)
        _err(
            f"anatid-server: pid {pid} did not exit within {args.timeout:g}s and was killed. "
            f"Writes that were still queued are lost."
        )
        return EXIT_FAILED
    _err(
        f"anatid-server: pid {pid} is still draining after {args.timeout:g}s. It is finishing "
        f"writes it accepted; wait, raise --timeout, or pass --force to kill it and lose them."
    )
    return EXIT_FAILED


# --------------------------------------------------------------------------- status


def _queue_stats(client: OperatorClient) -> QueueStats | None:
    """The queue counters, or None from a server that does not answer the verb.

    ``anatid.server.queue`` registers ``QueueStats`` with the codec, so every build that has a
    write queue can encode this answer.  The None branch is for the server that is older than
    the verb, or newer and has renamed it: a status command that died on either would be
    reporting a version difference as an outage.
    """
    try:
        stats = client.call("queue_stats")
    except (ProtocolError, AnatidError):
        return None
    return stats if isinstance(stats, QueueStats) else None


def cmd_status(args: argparse.Namespace) -> int:
    with _client_from(args) as client:
        health = client.call("health")
        ready = client.call("ready")
        stats = _queue_stats(client)

    if args.json:
        payload = {
            "address": client.address,
            "health": _as_mapping(health),
            "ready": _as_mapping(ready),
            "queue": _as_mapping(stats) if stats is not None else None,
        }
        _out(json.dumps(payload, indent=2, default=str))
        return EXIT_OK if getattr(ready, "ready", False) else EXIT_UNHEALTHY

    _out(f"anatid {health.anatid_version}  protocol {health.protocol_version}  pid {health.pid}")
    _out(f"address        {client.address}")
    _out(f"status         {health.status}{'' if health.ok else '  (not healthy)'}")
    _out(f"uptime         {_humanize(health.uptime_s)}")
    _out(f"ready          {'yes' if ready.ready else 'no'}")
    _out(f"accepting      {'yes' if ready.accepting else 'no'}")
    _out(f"open files     {ready.open_files}")
    versions = ", ".join(f"{t}=v{v}" for t, v in sorted(ready.schema_versions.items()))
    _out(
        f"schema         expected v{ready.expected_schema_version}  {versions or 'no tenant open'}"
    )
    _out(f"queue depth    {ready.max_tenant_depth} deepest tenant (high water {ready.high_water})")
    if stats is not None:
        _out(
            f"queue totals   submitted={stats.submitted} completed={stats.completed} "
            f"failed={stats.failed} rejected={stats.rejected} expired={stats.expired} "
            f"replayed={stats.replayed}"
        )
        _out(
            f"batching       {stats.mean_batch:.1f} writes per transaction over "
            f"{stats.batches} transactions"
        )
    if not ready.ready:
        _out(f"why not ready  {ready.detail}")
        return EXIT_UNHEALTHY
    return EXIT_OK


def _as_mapping(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is None:
        return value
    return {name: getattr(value, name) for name in fields}


# --------------------------------------------------------------------------- backup


def _quote_catalog(name: str) -> str:
    """Double-quote a catalog name DuckDB derived from a file name.

    Not :func:`anatid.schema.quote_ident`, and the difference matters here.  ``quote_ident``
    guards identifiers anatid itself chooses, so it accepts only ``[A-Za-z_][A-Za-z0-9_]*``.  A
    catalog name is not chosen by anatid: DuckDB takes it from the stem of the file it opened, so
    ``--pool /var/lib/anatid/tenant-{tenant}.anatid`` produces the catalog ``tenant-1`` and
    ``--pool .../{tenant}.anatid`` produces ``1``.  Both are legal DuckDB catalogs, both are
    refused by ``quote_ident``, and both are ordinary ways to name a pool.  Verified on duckdb
    1.5.5: ``COPY FROM DATABASE "tenant-1" TO "anatid_backup"`` runs and the copy opens.

    The name still comes from a path the operator wrote, so it is quoted rather than trusted: a
    literal double quote is doubled, and a name that could not be quoted safely is refused.
    """
    if not name:
        raise ConfigProblem("this database has no catalog name, so it cannot be copied")
    if "\x00" in name:
        raise ConfigProblem(f"catalog name {name!r} contains a NUL and cannot be quoted")
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _refuse_existing(dest: Path, *, overwrite: bool) -> None:
    """Refuse to write over an existing backup, in words the operator can act on.

    Every backup path in this file ends in a copy, and two of them reach
    :meth:`anatid.DatabasePool.backup`, whose own refusal names its ``overwrite=True`` keyword.
    That keyword is right for a Python caller and useless to an operator, whose only way to say
    it is ``--overwrite``.  Checking here gives the offline pool path, the offline single-file
    path and the server's backup verb one sentence and one exit code, instead of three that
    disagree: before this, ``backup --pool`` exited 1 quoting a keyword argument while
    ``backup --db`` exited 2 quoting the flag, for the same situation.
    """
    if dest.exists() and not overwrite:
        raise ConfigProblem(
            f"{dest} exists; pass --overwrite to replace it (a backup that overwrites "
            f"silently can leave you with neither copy)"
        )


def _copy_database(db: Anatid, dest: Path, *, overwrite: bool) -> Path:
    """``COPY FROM DATABASE`` one open handle into a new file at ``dest``.

    The same mechanism :meth:`anatid.DatabasePool.backup` uses, for the ``--db`` case that has no
    pool.  It is DuckDB copying committed data out of the handle that owns the file, so it is a
    consistent snapshot; copying the file with ``cp`` while anything is writing it is not, and
    copying it without its write-ahead log is not either.
    """
    _refuse_existing(dest, overwrite=overwrite)
    if dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    alias = "anatid_backup"
    literal = str(dest).replace("'", "''")
    row = db.execute("SELECT current_database()").fetchone()
    catalog = _quote_catalog(str(row[0]) if row else "memory")
    db.execute(f"ATTACH '{literal}' AS {quote_ident(alias)}")
    try:
        db.execute(f"COPY FROM DATABASE {catalog} TO {quote_ident(alias)}")
    finally:
        db.execute(f"DETACH {quote_ident(alias)}")
    with contextlib.suppress(OSError):
        os.chmod(dest, 0o600)
    return dest


def _pool_backup(pool: DatabasePool, tenant_id: int, dest: Path, *, overwrite: bool) -> Path:
    """One tenant's file copied out of the pool, whatever the pool template looks like.

    :meth:`anatid.DatabasePool.backup` does the work: it records a pool audit event, applies the
    pool's file mode, and quotes the source catalog with :func:`anatid.schema.quote_name` so
    every pool template works.  It used to validate that catalog with ``quote_ident`` instead,
    which refused ``tenant-{tenant}.anatid`` and ``{tenant}.anatid``, the two templates the
    documentation itself shows, and this function carried a fallback copy for them.  That is
    fixed at the root and the fallback is gone.  :func:`_copy_database` stays, for ``--db``,
    where there is no pool to ask.

    The destination is checked here rather than left to the copy, because the operator's word for
    it is ``--overwrite`` and nobody at a command line can pass ``overwrite=True``.
    """
    _refuse_existing(dest, overwrite=overwrite)
    return pool.backup(tenant_id, dest, overwrite=overwrite)


def cmd_backup(args: argparse.Namespace) -> int:
    if args.socket or args.http:
        name = args.name or _timestamp_name(args.tenant)
        with _client_from(args) as client:
            result = client.call(
                BACKUP_VERB, tenant=args.tenant, name=name, overwrite=bool(args.overwrite)
            )
        _out(
            f"anatid-server: tenant {result['tenant']} backed up to {result['path']} "
            f"({result['bytes']} bytes), written by the server that holds the file."
        )
        # An older server does not send these, so their absence is not a failure to report.
        if result.get("quiesced") is True:
            _out(
                f"anatid-server: guarantee {result.get('guarantee', 'quiesced')}. Every write "
                f"acknowledged before the call is in it and nothing that committed after is. "
                f"Tenant {result['tenant']} was paused {result.get('paused_s', 0)}s for it; no "
                f"other tenant was."
            )
        elif "guarantee" in result:
            _out(
                f"anatid-server: guarantee {result['guarantee']}. The barrier could not be held, "
                f"so this is a consistent copy of committed data rather than the point where "
                f"everything acknowledged so far had landed. A write acknowledged during the "
                f"copy may or may not be in it. Retry when the tenant is quieter, or raise "
                f"--drain-timeout, which is the budget the barrier waits within."
            )
        return EXIT_OK

    if not args.to:
        raise ConfigProblem("an offline backup needs --to PATH")
    dest = Path(args.to).expanduser()
    if args.pool:
        pool = _pool_from(args)
        try:
            path = _pool_backup(pool, args.tenant, dest, overwrite=bool(args.overwrite))
        finally:
            pool.close_all()
    elif args.db:
        with Anatid.open(args.db, tenant=args.tenant) as db:
            path = _copy_database(db, dest, overwrite=bool(args.overwrite))
    else:
        raise ConfigProblem(
            "backup needs somewhere to read from: --socket or --http to ask a running server, "
            "or --pool or --db to read the files directly while no server holds them"
        )
    _out(f"anatid-server: tenant {args.tenant} backed up to {path} ({path.stat().st_size} bytes).")
    return EXIT_OK


# --------------------------------------------------------------------------- restore


def _target_path(args: argparse.Namespace) -> Path:
    if args.pool:
        pool = _pool_from(args)
        try:
            return pool.path_for(args.tenant)
        finally:
            pool.close_all()
    if args.db:
        return Path(args.db).expanduser()
    raise ConfigProblem("restore needs --pool TEMPLATE --tenant N, or --db PATH")


def _read_schema_version(path: Path) -> int | None:
    """The schema version recorded in ``path``, read without changing it.

    Read-only first, because opening a database read-write can migrate it, and a restore that
    silently migrated the backup it was checking would be checking something else.  A file with
    a write-ahead log left by a crash cannot be opened read-only, so that case falls back and
    says so.
    """
    import duckdb

    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        raise ConfigProblem(
            f"{path} cannot be opened as a DuckDB database: {str(exc).splitlines()[0]}"
        ) from exc
    try:
        return current_version(con)
    finally:
        con.close()


def _is_locked(path: Path) -> bool:
    """Whether another process holds ``path``.  A missing file is not locked."""
    import duckdb

    if not path.exists():
        return False
    try:
        duckdb.connect(str(path), read_only=True).close()
    except Exception as exc:
        return "Could not set lock on file" in str(exc)
    return False


def cmd_restore(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser()
    if not source.exists():
        raise ConfigProblem(f"{source} does not exist")
    target = _target_path(args)

    if args.socket and Path(args.socket).expanduser().exists():
        raise ConfigProblem(
            f"{args.socket} exists, so a server is probably still running. Restore is offline: "
            f"stop the server first ('anatid-server stop --socket {args.socket}')."
        )
    if _is_locked(target):
        raise ConfigProblem(
            f"another process holds {target}. DuckDB gives one process exclusive use of a file, "
            f"so a restore has to wait for the server to exit; 'anatid-server stop' does that."
        )
    version = _read_schema_version(source)
    if version is None:
        raise ConfigProblem(
            f"{source} is a DuckDB database with no anatid schema in it. Restoring it would "
            f"replace a tenant's memory with an unrelated file."
        )
    if version > SCHEMA_VERSION:
        raise ConfigProblem(
            f"{source} is at anatid schema v{version} and this build understands v"
            f"{SCHEMA_VERSION}. Upgrade anatid before restoring it."
        )

    kept: list[Path] = []
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for existing in (target, Path(f"{target}.wal")):
        if not existing.exists():
            continue
        aside = Path(f"{existing}.replaced-{stamp}")
        # The write-ahead log goes with the file it belongs to.  A restored file left beside the
        # previous file's WAL is a database DuckDB will try to finish writing on next open.
        existing.rename(aside)
        kept.append(aside)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, target)
        with contextlib.suppress(OSError):
            os.chmod(target, 0o600)
        restored = _read_schema_version(target)
    except Exception:
        for aside in kept:
            aside.rename(str(aside).rsplit(".replaced-", 1)[0])
        raise
    _out(f"anatid-server: restored {source} to {target} (anatid schema v{restored}).")
    if kept:
        _out(f"  the previous file is at {kept[0]}; delete it once the restore is verified.")
    if restored is not None and restored < SCHEMA_VERSION:
        _out(
            f"  it is at v{restored} and this build expects v{SCHEMA_VERSION}: the next open "
            f"migrates it, and readiness stays false until that finishes."
        )
    return EXIT_OK


# --------------------------------------------------------------------------- doctor


def _print_report(report: Any, tenant: int) -> int:
    _out(
        f"tenant {tenant}: schema v{report.schema_version} (expected v{report.expected_schema_version})"
    )
    counts = ", ".join(f"{k}={v}" for k, v in sorted(report.counts.items()))
    if counts:
        _out(f"  rows      {counts}")
    _out(f"  checks    {len(report.checks_run)} ran, {len(report.checks_skipped)} skipped")
    for finding in report.findings:
        _out(f"  {finding.severity.value.upper():7} {finding.check}: {finding.detail}")
    if report.clean:
        _out("  clean: nothing to report.")
        return EXIT_OK
    if report.ok:
        _out(f"  ok, with {len(report.warnings)} warning(s).")
        return EXIT_OK
    _out(f"  NOT ok: {len(report.errors)} error(s). doctor never repairs; see docs/server.md.")
    return EXIT_UNHEALTHY


def cmd_doctor(args: argparse.Namespace) -> int:
    if args.socket or args.http:
        with _client_from(args) as client:
            report = client.call("doctor", tenant=args.tenant, deep=not args.shallow)
        return _print_report(report, args.tenant)
    if args.pool:
        pool = _pool_from(args)
        try:
            report = pool.get(args.tenant).doctor(deep=not args.shallow)
        finally:
            pool.close_all()
        return _print_report(report, args.tenant)
    if args.db:
        with Anatid.open(args.db, tenant=args.tenant) as db:
            report = db.doctor(deep=not args.shallow)
        return _print_report(report, args.tenant)
    raise ConfigProblem(
        "doctor needs --socket or --http to ask a running server, or --pool or --db to read "
        "the files directly while no server holds them"
    )


# --------------------------------------------------------------------------- the parser


def _add_storage(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--db",
        metavar="PATH",
        help="one anatid file holding every tenant as a namespace. Namespaces are scoping, "
        "not a security boundary",
    )
    p.add_argument(
        "--pool",
        metavar="TEMPLATE",
        help="a path template with {tenant} in it, so each tenant gets its own file. This is "
        "the isolated arrangement",
    )
    p.add_argument("--max-open", type=int, default=16, metavar="N", help="pool file limit")
    p.add_argument(
        "--embedding-dim",
        type=int,
        metavar="N",
        help="FLOAT[N] width for files this server CREATES. An existing file keeps the width "
        "recorded in it",
    )


def _add_client(p: argparse.ArgumentParser) -> None:
    p.add_argument("--socket", metavar="PATH", help="a running server's Unix socket")
    p.add_argument("--http", metavar="HOST:PORT", help="a running server's HTTP address")
    p.add_argument(
        "--token",
        metavar="TOKEN",
        help="bearer token for --http. It is visible in the process list; --token-file is not",
    )
    p.add_argument(
        "--token-file", metavar="PATH", help="read the bearer token from the first line of a file"
    )
    p.add_argument(
        "--timeout", type=float, default=30.0, metavar="S", help="seconds to wait for an answer"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anatid-server",
        description=(
            "Run and operate an anatid server: one process owns the DuckDB files and everyone "
            "else reads and writes them over a socket."
        ),
        epilog=(
            "DuckDB gives one process exclusive use of a database file, read-only openers "
            "included, so every read comes back over the wire too. See docs/server.md."
        ),
    )
    parser.add_argument("--version", action="version", version=f"anatid-server {_anatid_version}")
    subs = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    start = subs.add_parser(
        "start",
        help="run the server until SIGTERM",
        description="Run the server until SIGTERM, then drain the write queues and exit.",
    )
    start.add_argument("--socket", metavar="PATH", help="Unix socket to listen on, mode 0600")
    start.add_argument(
        "--http",
        metavar="HOST:PORT",
        help="also listen for HTTP on this address. Needs --token-file unless --http-no-auth",
    )
    start.add_argument(
        "--token-file",
        metavar="PATH",
        help="bearer tokens for the HTTP listener: one per line, "
        "'<token> [name=NAME] [tenants=1,2] [read-only]'",
    )
    start.add_argument(
        "--http-no-auth",
        action="store_true",
        help="serve HTTP with no token. Only for a port no other user or host can reach",
    )
    start.add_argument(
        "--allow-uid",
        action="append",
        type=int,
        metavar="UID",
        help="restrict the Unix socket to these uids as well as to its permissions (repeatable)",
    )
    _add_storage(start)
    start.add_argument(
        "--tenant",
        action="append",
        type=int,
        metavar="N",
        help="open this tenant at startup and hold readiness false until it is migrated "
        "(repeatable)",
    )
    start.add_argument(
        "--max-queue",
        type=int,
        default=256,
        metavar="N",
        help="queued writes per tenant before the server answers busy (default 256)",
    )
    start.add_argument(
        "--batch-max",
        type=int,
        default=32,
        metavar="N",
        help="writes that may share one transaction (default 32)",
    )
    start.add_argument(
        "--workers", type=int, default=4, metavar="N", help="tenants written in parallel"
    )
    start.add_argument(
        "--read-workers", type=int, default=8, metavar="N", help="threads answering reads"
    )
    start.add_argument(
        "--drain-timeout",
        type=float,
        default=30.0,
        metavar="S",
        help="seconds to finish queued writes on shutdown, and the bound on the whole exit "
        "(default 30). It is also the budget an online backup spends draining its tenant",
    )
    start.add_argument(
        "--deadline",
        type=float,
        default=30.0,
        metavar="S",
        help="default per-request budget in seconds (default 30)",
    )
    start.add_argument("--no-idempotency", action="store_true", help="do not keep idempotency keys")
    start.add_argument(
        "--no-create-tenants",
        action="store_true",
        help="serve only --tenant plus the files that already exist. Without it, a client that "
        "may name any tenant can make this server create a file for every integer it sends",
    )
    start.add_argument(
        "--idempotency-ttl",
        type=float,
        default=86400.0,
        metavar="S",
        help="how long an idempotency key is remembered (default 86400)",
    )
    start.add_argument(
        "--embeddings",
        choices=("list", "f32"),
        default="list",
        help="how embeddings cross the wire. f32 is base64 float32: 3x smaller and 4x faster to "
        "encode, and rounds a float64 the caller computed",
    )
    start.add_argument(
        "--backup-dir",
        metavar="DIR",
        help="where the backup verb writes. Without it the server refuses backup requests",
    )
    start.add_argument("--pid-file", metavar="PATH", help="write this process's pid here")
    start.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="default info",
    )
    start.add_argument(
        "--check",
        action="store_true",
        help="validate the configuration, print it and exit without binding or opening anything",
    )
    start.set_defaults(func=cmd_start)

    stop = subs.add_parser(
        "stop",
        help="drain and stop a running server",
        description="Send SIGTERM to a server on this machine and wait for it to drain and exit.",
    )
    _add_client(stop)
    stop.add_argument("--pid-file", metavar="PATH", help="take the pid from here, not the server")
    stop.add_argument(
        "--force", action="store_true", help="SIGKILL if the drain outlasts --timeout"
    )
    stop.set_defaults(func=cmd_stop)

    status = subs.add_parser(
        "status",
        help="health, readiness and queue counters",
        description="Ask a running server how it is. Exit 0 ready, 3 not ready, 1 unreachable.",
    )
    _add_client(status)
    status.add_argument("--json", action="store_true", help="machine-readable output")
    status.set_defaults(func=cmd_status)

    backup = subs.add_parser(
        "backup",
        help="copy one tenant's database",
        description=(
            "A consistent copy of one tenant's file. With --socket or --http the running server "
            "makes it, which is the only way while the server holds the file; with --pool or "
            "--db this process makes it, which needs the server to be stopped."
        ),
    )
    _add_client(backup)
    _add_storage(backup)
    backup.add_argument("--tenant", type=int, required=True, metavar="N")
    backup.add_argument(
        "--name", metavar="FILENAME", help="online: the file name inside the server's --backup-dir"
    )
    backup.add_argument("--to", metavar="PATH", help="offline: where to write the copy")
    backup.add_argument("--overwrite", action="store_true", help="replace an existing destination")
    backup.set_defaults(func=cmd_backup)

    restore = subs.add_parser(
        "restore",
        help="put a backup back, offline",
        description=(
            "Replace a tenant's file with a backup. Offline only: the server has to be stopped, "
            "because DuckDB gives one process exclusive use of the file. The previous file and "
            "its write-ahead log are renamed aside rather than deleted."
        ),
    )
    _add_storage(restore)
    restore.add_argument("--tenant", type=int, default=0, metavar="N")
    restore.add_argument(
        "--socket", metavar="PATH", help="refuse if a server is still listening here"
    )
    restore.add_argument("source", metavar="BACKUP", help="the file to restore from")
    restore.set_defaults(func=cmd_restore)

    doctor = subs.add_parser(
        "doctor",
        help="integrity report for one tenant",
        description=(
            "Run anatid's integrity checks. doctor reads and never repairs. Exit 0 when nothing "
            "of severity ERROR was found, 3 when something was."
        ),
    )
    _add_client(doctor)
    _add_storage(doctor)
    doctor.add_argument("--tenant", type=int, default=0, metavar="N")
    doctor.add_argument("--shallow", action="store_true", help="skip the row-scanning checks")
    doctor.set_defaults(func=cmd_doctor)
    return parser


# --------------------------------------------------------------------------- entry point


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point (``anatid-server``), and ``python -m anatid.server``.

    Returns a process exit code and raises nothing: an operator gets a sentence on stderr, not a
    traceback, for every failure this command can have.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except ConfigProblem as exc:
        _err(f"anatid-server: {exc}")
        return EXIT_CONFIG
    except KeyboardInterrupt:
        _err("anatid-server: interrupted")
        return EXIT_FAILED
    except (ConnectionError, OSError) as exc:
        _err(f"anatid-server: {exc}")
        return EXIT_FAILED
    except AnatidError as exc:
        _err(f"anatid-server: {type(exc).__name__}: {exc}")
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
