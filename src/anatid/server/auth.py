"""Who a connection is, and which tenants it may name.

The boundary this defends
-------------------------
anatid's tenant isolation is file per tenant, enforced by the wrapper and the filesystem.  A
shared file gives namespaces, not a security boundary, and that is written down wherever tenancy
is documented.  The server must not weaken it, and there is exactly one way it could: by letting
a connection authenticated for one tenant name another in a request.  So a :class:`Principal`
carries the set of tenants it may name, the dispatcher asks before it resolves a handle, and a
request for a tenant outside the set is refused with
:class:`~anatid.server.protocol.AuthorizationError` before anything opens a file.  The refusal
says nothing about whether the tenant exists, so a client cannot enumerate tenants by watching
which errors change.

The two transports authenticate differently, on purpose
-------------------------------------------------------
A Unix domain socket is a filesystem object.  Who may connect is decided by the directory and
file permissions on it -- 0700 on the directory, 0600 on the socket, which is what the server
sets -- and that is a real check the kernel makes, not one this module could improve on.  On top
of it, where the platform provides peer credentials, the server learns the connecting process's
uid and gid and can map them to principals.  Peer credentials are available on Linux
(``SO_PEERCRED``) and on macOS and the BSDs (``LOCAL_PEERCRED``); :func:`peer_credentials`
returns None where they are not, and a policy that requires them says so rather than silently
accepting everyone.

A TCP socket is not a filesystem object and has no equivalent.  So HTTP requires a bearer token,
and :func:`check_bind_address` refuses to bind anything but a loopback address without one.  The
refusal is not advice printed at startup: it is an exception, because a server that binds
0.0.0.0 with no authentication is not a misconfiguration a log line fixes.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import socket
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from .protocol import AuthenticationError, AuthorizationError

log = logging.getLogger("anatid.server")

__all__ = [
    "PeerCredentials",
    "peer_credentials",
    "Principal",
    "ConnectionContext",
    "Authenticator",
    "AllowAllAuthenticator",
    "UnixPeerAuthenticator",
    "BearerTokenAuthenticator",
    "is_loopback",
    "check_bind_address",
    "LOOPBACK_HOSTS",
    "SOCKET_MODE",
    "SOCKET_DIR_MODE",
]

#: Mode the server sets on its Unix socket: owner read and write, nobody else.  On Linux the
#: permission bits of a socket file are enforced on connect; on some BSDs they historically were
#: not, which is why the directory mode below matters as much as this one.
SOCKET_MODE = 0o600

#: Mode for the directory holding the socket.  0700 is the check that holds everywhere: a
#: process that cannot traverse the directory cannot reach the socket whatever its own mode says.
SOCKET_DIR_MODE = 0o700


# --------------------------------------------------------------------------- peer credentials


@dataclass(frozen=True, slots=True)
class PeerCredentials:
    """The uid, gid and (where the platform gives it) pid of the process at the other end."""

    uid: int
    gid: int
    pid: int | None = None
    source: str = ""

    def __str__(self) -> str:
        pid = "" if self.pid is None else f" pid={self.pid}"
        return f"uid={self.uid} gid={self.gid}{pid}"


def peer_credentials(sock: socket.socket) -> PeerCredentials | None:
    """The credentials of the process connected to ``sock``, or None where unavailable.

    Linux answers with ``SO_PEERCRED`` (pid, uid, gid).  macOS and the BSDs answer with
    ``LOCAL_PEERCRED``, a ``struct xucred`` that carries the uid and the group list but no pid,
    so ``pid`` is None there.  Anywhere else, and on a socket that is not a Unix socket, this
    returns None: a caller that requires credentials has to treat None as "no", which
    :class:`UnixPeerAuthenticator` does.
    """
    if sock.family != getattr(socket, "AF_UNIX", object()):
        return None
    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is not None:
        try:
            blob = sock.getsockopt(socket.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
        except OSError:
            return None
        pid, uid, gid = struct.unpack("3i", blob)
        return PeerCredentials(uid=uid, gid=gid, pid=pid, source="SO_PEERCRED")
    # macOS / BSD: struct xucred { u_int cr_version; uid_t cr_uid; short cr_ngroups;
    #                              gid_t cr_groups[NGROUPS]; }
    sol_local = getattr(socket, "SOL_LOCAL", 0)
    local_peercred = getattr(socket, "LOCAL_PEERCRED", 0x001)
    try:
        blob = sock.getsockopt(sol_local, local_peercred, 4 + 4 + 2 + 2 + 16 * 4)
    except OSError:
        return None
    if len(blob) < 12:
        return None
    _version, uid, ngroups = struct.unpack_from("<IIh", blob, 0)
    gid = uid
    if ngroups > 0 and len(blob) >= 16:
        (gid,) = struct.unpack_from("<I", blob, 12)
    return PeerCredentials(uid=int(uid), gid=int(gid), pid=None, source="LOCAL_PEERCRED")


# --------------------------------------------------------------------------- principals


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated identity and the tenants it may name.

    ``tenants`` is None for a principal allowed to reach every tenant the server holds -- the
    single-user embedded case, and the administrator.  Anything else is an explicit set, and the
    set is the whole authorization model: there are no roles and no per-verb permissions,
    because the boundary anatid actually enforces is the tenant, and inventing a second one that
    the storage layer does not back would be a claim this design cannot keep.
    """

    name: str
    tenants: frozenset[int] | None = None
    peer: PeerCredentials | None = None
    transport: str = "unix"
    read_only: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_tenants(cls, name: str, tenants: Iterable[int], **kw: Any) -> "Principal":
        return cls(name=name, tenants=frozenset(int(t) for t in tenants), **kw)

    @property
    def unrestricted(self) -> bool:
        """True when this principal may name any tenant."""
        return self.tenants is None

    def may(self, tenant_id: int) -> bool:
        """Whether this principal may name ``tenant_id``."""
        return self.tenants is None or int(tenant_id) in self.tenants

    def require(self, tenant_id: int) -> int:
        """``tenant_id``, or :class:`~anatid.server.protocol.AuthorizationError`.

        The message names the principal and the tenant it asked for, and says nothing about
        whether that tenant exists on this server.  A different message for "no such tenant"
        would let a client enumerate tenants it is not allowed to see.
        """
        if not self.may(tenant_id):
            raise AuthorizationError(f"{self.name} is not authorized for tenant {int(tenant_id)}")
        return int(tenant_id)

    def require_write(self, verb: str) -> None:
        """Refuse a write for a read-only principal."""
        if self.read_only:
            raise AuthorizationError(f"{self.name} is read-only and cannot call {verb}")


@dataclass(frozen=True, slots=True)
class ConnectionContext:
    """What an authenticator is given about one connection.

    ``socket`` is the accepted socket for a Unix connection and None for HTTP, where there is
    nothing to ask the kernel about.  ``headers`` is the HTTP request's headers, lower-cased, and
    empty for a Unix connection.  ``peer`` is the remote address as the transport reports it.
    """

    transport: str
    socket: socket.socket | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    peer: Any = None

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


@runtime_checkable
class Authenticator(Protocol):
    """Turns a connection into a :class:`Principal`, or refuses it."""

    def authenticate(self, ctx: ConnectionContext) -> Principal:
        """The principal for ``ctx``.

        Raises :class:`~anatid.server.protocol.AuthenticationError` when the connection cannot
        be attributed.  Never returns None: a server that treats None as "anonymous" is one
        refactor away from treating a bug as a login.
        """
        ...


class AllowAllAuthenticator:
    """Every connection is the same principal, with access to every tenant.

    Honest about what it is.  It is the right policy for a Unix socket in a 0700 directory owned
    by the user running the server, where the filesystem has already made the decision and a
    second one here would be theatre.  It is the wrong policy for anything reachable over TCP,
    and :func:`check_bind_address` refuses that combination.
    """

    def __init__(
        self, name: str = "local", tenants: Iterable[int] | None = None, read_only: bool = False
    ) -> None:
        self.name = name
        self.tenants = None if tenants is None else frozenset(int(t) for t in tenants)
        self.read_only = bool(read_only)

    @property
    def requires_token(self) -> bool:
        return False

    def authenticate(self, ctx: ConnectionContext) -> Principal:
        peer = peer_credentials(ctx.socket) if ctx.socket is not None else None
        return Principal(
            name=self.name,
            tenants=self.tenants,
            peer=peer,
            transport=ctx.transport,
            read_only=self.read_only,
        )


class UnixPeerAuthenticator:
    """Authenticates a Unix connection by the connecting process's uid.

    ``allow_uids`` is the set of uids that may connect; None means "the uid running the server",
    which is the default because it is the one that matches the 0600 socket the server creates.
    ``tenants_by_uid`` maps a uid to the tenants it may name; a uid not in the mapping gets
    ``default_tenants``.

    ``require_credentials`` decides what happens where the platform has no peer credentials to
    give.  True refuses the connection, which is the safe answer and the right one when the
    socket may be reachable by another user.  False falls back to the filesystem permission
    check alone and says so in the principal's attributes, which is the right answer for a
    socket in a 0700 directory that only the server's own user can traverse.
    """

    def __init__(
        self,
        *,
        allow_uids: Iterable[int] | None = None,
        tenants_by_uid: Mapping[int, Iterable[int]] | None = None,
        default_tenants: Iterable[int] | None = None,
        require_credentials: bool = False,
        read_only_uids: Iterable[int] = (),
    ) -> None:
        self.allow_uids = (
            frozenset({os.getuid()})
            if allow_uids is None
            else frozenset(int(u) for u in allow_uids)
        )
        self.tenants_by_uid = {
            int(u): frozenset(int(t) for t in ts) for u, ts in (tenants_by_uid or {}).items()
        }
        self.default_tenants = (
            None if default_tenants is None else frozenset(int(t) for t in default_tenants)
        )
        self.require_credentials = bool(require_credentials)
        self.read_only_uids = frozenset(int(u) for u in read_only_uids)

    @property
    def requires_token(self) -> bool:
        return False

    def authenticate(self, ctx: ConnectionContext) -> Principal:
        if ctx.transport != "unix":
            raise AuthenticationError(
                "UnixPeerAuthenticator only authenticates Unix socket connections; a TCP "
                "listener needs BearerTokenAuthenticator"
            )
        creds = peer_credentials(ctx.socket) if ctx.socket is not None else None
        if creds is None:
            if self.require_credentials:
                raise AuthenticationError(
                    "this platform does not report Unix peer credentials and this server was "
                    "configured to require them; connect over a socket whose directory "
                    "permissions restrict it, and set require_credentials=False, or use a "
                    "bearer token"
                )
            return Principal(
                name="peer",
                tenants=self.default_tenants,
                peer=None,
                transport="unix",
                attributes={"authenticated_by": "filesystem permissions"},
            )
        if creds.uid not in self.allow_uids:
            raise AuthenticationError(f"uid {creds.uid} is not allowed to connect to this server")
        return Principal(
            name=f"uid:{creds.uid}",
            tenants=self.tenants_by_uid.get(creds.uid, self.default_tenants),
            peer=creds,
            transport="unix",
            read_only=creds.uid in self.read_only_uids,
            attributes={"authenticated_by": creds.source},
        )


class BearerTokenAuthenticator:
    """Authenticates by ``Authorization: Bearer <token>``.

    Tokens map to principals, so one server can hold a token per tenant and the tenant check
    downstream is then the same check for both transports.  Comparison is
    :func:`hmac.compare_digest` against every candidate, which keeps the time taken independent
    of how much of a wrong token was right.

    A token is a secret in this process's memory.  It is never logged, never echoed in an error,
    and the failure message says only that the token was not accepted.

    A token that could never be sent is refused at construction rather than at the first
    request.  :meth:`authenticate` strips the value it takes off the header, because a client
    library that appends a newline is common and a trailing space is invisible in a config file;
    a configured token with leading or trailing whitespace therefore could never match anything,
    and a server that started anyway would refuse every request with "the bearer token was not
    accepted" and give the operator nothing to go on.  Stripping the configured token instead
    would be worse: two entries that differ only in whitespace would silently become one.
    """

    def __init__(
        self,
        tokens: Mapping[str, Principal | Iterable[int] | None],
        *,
        header: str = "authorization",
    ) -> None:
        if not tokens:
            raise ValueError(
                "BearerTokenAuthenticator needs at least one token; an empty mapping would "
                "refuse every request, which is a configuration error worth failing on at "
                "startup rather than at the first request"
            )
        self.header = header.lower()
        self._tokens: dict[str, Principal] = {}
        for token, value in tokens.items():
            if not token:
                raise ValueError("a token cannot be the empty string")
            if token != token.strip():
                raise ValueError(
                    "a token cannot start or end with whitespace: the header value is stripped "
                    "before it is compared, so this one could never be sent and the server "
                    "would refuse every request that offered it"
                )
            if any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in token):
                raise ValueError(
                    "a token cannot contain whitespace or control characters: it travels in an "
                    "HTTP header, where a newline ends the header and a space ends the value"
                )
            if isinstance(value, Principal):
                principal = value
            elif value is None:
                principal = Principal(name=f"token:{_fingerprint(token)}", transport="http")
            else:
                principal = Principal.for_tenants(
                    f"token:{_fingerprint(token)}", value, transport="http"
                )
            self._tokens[token] = principal

    @property
    def requires_token(self) -> bool:
        return True

    def authenticate(self, ctx: ConnectionContext) -> Principal:
        raw = ctx.header(self.header)
        if not raw:
            raise AuthenticationError(
                f"this listener requires an {self.header} header of the form 'Bearer <token>'"
            )
        scheme, _, token = raw.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthenticationError(f"the {self.header} header must be 'Bearer <token>'")
        # Bytes, not str: hmac.compare_digest refuses a non-ASCII str, and a client sending
        # one should get "not accepted", not a TypeError out of the authenticator.
        offered = token.strip().encode("utf-8")
        matched: Principal | None = None
        for candidate, principal in self._tokens.items():
            if hmac.compare_digest(candidate.encode("utf-8"), offered):
                matched = principal
        if matched is None:
            raise AuthenticationError("the bearer token was not accepted")
        return Principal(
            name=matched.name,
            tenants=matched.tenants,
            peer=None,
            transport=ctx.transport,
            read_only=matched.read_only,
            attributes=dict(matched.attributes),
        )


def _fingerprint(token: str) -> str:
    """Eight hex characters of the token's digest, for a principal name that is not the token."""
    import hashlib

    return hashlib.sha256(token.encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- binding


#: Host names that mean "this machine only" without a DNS lookup.
LOOPBACK_HOSTS = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})


def is_loopback(host: str) -> bool:
    """True when binding ``host`` reaches only this machine.

    An empty host, ``0.0.0.0`` and ``::`` are every interface, so they are not loopback.  A name
    is only accepted from :data:`LOOPBACK_HOSTS`: resolving an arbitrary name here would make the
    safety of a bind depend on what DNS says at that moment, which is not a property a check
    should have.
    """
    name = (host or "").strip().lower()
    if not name:
        return False
    if name in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(name.strip("[]")).is_loopback
    except ValueError:
        return False


def check_bind_address(host: str, authenticator: Any) -> None:
    """Refuse to bind a non-loopback interface without authentication.

    Raises ``ValueError`` when ``host`` is reachable from outside this machine and
    ``authenticator`` does not require a token.  This is a hard failure at configuration time,
    not a warning at runtime: an anatid file holds an agent's memory, and a server that puts it
    on an unauthenticated port is not something to log about and continue from.

    A loopback bind without a token is allowed and warned about.  It is correct on a single-user
    machine and in a container with one process in it, and it is wrong on a shared host, where
    ``127.0.0.1`` is reachable by every local user and no filesystem permission stands between
    them and the port.  Only the operator knows which of those this is, so this warns rather than
    refuses; ``anatid-server`` is stricter and refuses ``--http`` outright without
    ``--token-file`` unless ``--http-no-auth`` says the choice was deliberate.
    """
    if is_loopback(host):
        if not getattr(authenticator, "requires_token", False):
            log.warning(
                "binding HTTP to %r with %s, which requires no token. Every local user on this "
                "machine can reach a loopback port and would be able to read and write every "
                "tenant this server holds. Pass a BearerTokenAuthenticator unless this machine "
                "has one user.",
                host,
                type(authenticator).__name__,
            )
        return
    if getattr(authenticator, "requires_token", False):
        return
    raise ValueError(
        f"refusing to bind HTTP to {host!r} without authentication. {host!r} is reachable from "
        f"outside this machine, and the configured authenticator "
        f"({type(authenticator).__name__}) does not require a token, so anyone who can reach "
        f"the port could read and write every tenant this server holds. Either bind 127.0.0.1, "
        f"or pass a BearerTokenAuthenticator."
    )
