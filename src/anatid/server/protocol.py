"""The wire contract between an anatid client and :class:`anatid.server.AnatidServer`.

Why there is a wire at all
--------------------------
DuckDB takes an exclusive lock on a database file for the whole time a process holds it
read-write.  Measured on this build (duckdb 1.5.5, macOS arm64), a second process asking for the
same file gets ``duckdb.IOException`` whether it asks for read-write **or read-only** access:

    holder=read-write  second=read-write  -> IOException: Could not set lock on file ...
    holder=read-write  second=read-only   -> IOException: Could not set lock on file ...
    holder=read-only   second=read-write  -> IOException: Could not set lock on file ...
    holder=read-only   second=read-only   -> ok

So "the server writes and everybody else reads the file directly" is not a design that exists.
While the server holds a tenant's file, that file is unreadable to every other process, and every
read as well as every write has to come back over this protocol.  That is the single fact the
whole server profile is built on.

The format
----------
Length-prefixed JSON, one frame per message:

    +---------------------------+--------------------------------+
    | 4 bytes, big-endian uint  | that many bytes of UTF-8 JSON  |
    +---------------------------+--------------------------------+

JSON rather than a binary format, and the choice is not free.  Measured, p50: a bare Unix-socket
echo of a 128-byte frame is 7.5 us, but a whole server round trip for ``get`` on a memory with a
384-dimension embedding is 589 us more than the same call in process, and 260 us of that 589 is
this codec.  The socket is noise; the JSON is not.

The reason the format stays JSON anyway is that the 260 us has one cause, embeddings, and one
remedy that does not require a binary wire.  A ``Memory`` carrying a 384-float embedding is 7,725
bytes and 274 us to encode and decode as a float array; the same value as base64 little-endian
float32 is 2,408 bytes and 69 us, 4.0x faster and 3.2x smaller.  At 1536 dimensions it is 29,934
bytes and 1,050 us against 8,552 bytes and 211 us, 5.0x and 3.5x.  So the codec accepts, and on
request emits, that form (``embeddings="f32"``), and everything else on the wire stays readable
with ``nc``.  The default is the float array, because the default should be the one you can read
and because f32 rounds: lossless for a value that came out of anatid's ``FLOAT[N]`` column,
lossy for a float64 a caller computed (measured maximum absolute error 3.0e-08).

Types on the wire
-----------------
JSON has objects, arrays, strings, numbers, booleans and null.  anatid's verbs return datetimes,
tuples, enums and frozen dataclasses.  Every value that is not natively JSON is therefore wrapped
in a tagged object::

    {"__anatid__": "datetime", "v": "2026-09-04T11:22:33.000456"}
    {"__anatid__": "Memory",   "v": {"memory_id": 7, "tenant_id": 1, ...}}

Tagging is structural: the encoder marks what it emits and the decoder reads the mark, so nothing
has to know a verb's return type to decode its result.  A plain JSON object that happens to
contain the key ``__anatid__`` is encoded as a tagged map instead, so a user dictionary can never
be mistaken for a tagged value.

Ids
---
anatid mints 63-bit ids, and a JSON *number* is a double in JavaScript, whose largest exact
integer is ``2**53 - 1``.  An id sent as a number comes back changed and nothing raises::

    $ node -e 'console.log(JSON.parse(String.raw`{"memory_id": 883768514279557120}`).memory_id)'
    883768514279557100

Measured on this build, 1,984 of 2,000 freshly minted ids change value under a plain
``JSON.parse``.  So an integer this codec cannot fit in a JSON number is tagged like any other
value JSON has no shape for, and its ``v`` is the decimal string::

    {"__anatid__": "id", "v": "883768514279557120"}

The rule is :func:`anatid.integrations.wire.is_wire_safe_int`, the same definition the MCP server
and the Agents SDK tools already use, so there is one answer to "which integers cannot travel as
numbers" rather than two that can drift.  The test is on magnitude and not on a field name: the
codec is structural and does not know which field is an id, and a sweep over every reply is what
catches the id nobody thought of.  A ``count(*)`` of 12 is still the number 12.

Decoding is exact: :func:`decode_value` turns the tag back into a Python ``int``, so a Python
client sees ``int`` on both sides and nothing in this package deals in id strings.  A client in
another language reads the string and keeps the digits.  Requests are encoded by the same
function, so an id in ``args`` travels the same way in the other direction.

Errors
------
A response carries either a result or a :class:`WireError`.  ``WireError`` keeps the error class
name, the message, the retryable flag, and for a conflict the resource, the expected version and
the current version, so :class:`anatid.errors.ConflictError` survives the wire with the fields a
caller reasons about.  :meth:`WireError.to_exception` rebuilds the original class when this build
knows it and :class:`RemoteError` when it does not, so a client is never left with a string.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import datetime as _dt
import json
import math
import pathlib as _pathlib
import struct
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from .. import errors as _errors
from .. import types as _types
from ..integrations.wire import is_wire_safe_int

__all__ = [
    "PROTOCOL_VERSION",
    "MAX_FRAME_BYTES",
    "LENGTH_PREFIX_BYTES",
    "TAG",
    "ID_TAG",
    "ProtocolError",
    "FrameError",
    "RemoteError",
    "BusyError",
    "DeadlineExceeded",
    "AuthenticationError",
    "AuthorizationError",
    "IdempotencyConflict",
    "ShuttingDown",
    "Status",
    "Request",
    "Response",
    "WireError",
    "encode_value",
    "decode_value",
    "dumps",
    "loads",
    "pack_frame",
    "frame_length",
    "read_frame",
    "write_frame",
    "read_frame_async",
    "write_frame_async",
    "register_dataclass",
    "register_enum",
    "register_codec",
    "register_error",
    "new_request_id",
]

#: Bumped when a change to this file would make an older client read a message wrongly.  A
#: request naming a different version is refused with :class:`ProtocolError` rather than
#: decoded on the hope that the shape did not move.
PROTOCOL_VERSION = 1

#: Bytes in the length prefix.  Four, big-endian, unsigned.
LENGTH_PREFIX_BYTES = 4

#: The largest frame the reader will assemble.  A length prefix is four attacker-controlled
#: bytes; without a cap a single 4-byte write asks the server for 4 GiB.
MAX_FRAME_BYTES = 64 * 1024 * 1024

#: The key that marks a tagged value.  Chosen to be unlikely in a user dictionary, and handled
#: even so: see :func:`encode_value`.
TAG = "__anatid__"

#: The tag an integer too large for a JSON number travels under.  Its ``v`` is the decimal
#: string, so a client that parses numbers as doubles reads the digits rather than a rounded
#: double.  See the "Ids" section of this module's docstring.
ID_TAG = "id"

_MAX_UINT32 = (1 << 32) - 1


# --------------------------------------------------------------------------- errors
#
# Every error class the wire can name that :mod:`anatid.errors` does not already define lives
# here, in one module, so the class registry below is complete and nothing has to import
# sideways to build it.


class ProtocolError(_errors.AnatidError):
    """A message could not be understood: wrong protocol version, missing field, bad shape.

    Never retryable.  The same bytes decode the same way on every attempt.
    """

    retryable = False


class FrameError(ProtocolError):
    """The framing is wrong: a short read, a length prefix past :data:`MAX_FRAME_BYTES`, or
    a body that is not UTF-8 JSON.  The stream is not resynchronisable after one, because the
    reader no longer knows where the next frame starts, so a connection that raises this is
    closed."""


class RemoteError(_errors.AnatidError):
    """An error the server raised whose class this client does not have.

    ``error_class`` is the name the server used and ``retryable`` is what the server said about
    it.  Raised by :meth:`WireError.to_exception` instead of guessing at a superclass, so a
    client can log the real name rather than a string that lost it.
    """

    def __init__(
        self,
        message: str,
        *,
        error_class: str = "AnatidError",
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.retryable = bool(retryable)
        self.details = dict(details or {})


class BusyError(_errors.AnatidError):
    """A tenant's write queue is full and the server refused the write instead of blocking.

    This is backpressure said out loud.  The alternatives are worse: blocking forever hides a
    server that is not keeping up until the client's own timeout fires, and dropping the write
    loses it silently.  ``retry_after`` is the server's estimate, in seconds, of how long the
    queue takes to fall below its high-water mark at the rate it is currently draining, and
    ``depth`` and ``max_depth`` say how full it was, so a client can decide to shed load rather
    than retry.  Always retryable: the queue drains.
    """

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after: float = 0.05,
        tenant_id: int | None = None,
        depth: int | None = None,
        max_depth: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = float(retry_after)
        self.tenant_id = tenant_id
        self.depth = depth
        self.max_depth = max_depth


class UnsupportedTransport(_errors.AnatidError):
    """The requested transport does not exist on this platform (for example a Unix domain socket on Windows)."""


class DeadlineExceeded(_errors.AnatidError):
    """The request's deadline passed before the server ran it.

    Retryable, because nothing was written: a write whose deadline expires while it is queued is
    dropped before the transaction opens, never half-done.  A deadline that expires while the
    write is already running is not enforced -- the transaction finishes or it aborts, and this
    error is not raised for it.  See :attr:`Request.deadline`.
    """

    retryable = True

    def __init__(
        self, message: str, *, waited: float | None = None, deadline: float | None = None
    ) -> None:
        super().__init__(message)
        self.waited = waited
        self.deadline = deadline


class AuthenticationError(_errors.AnatidError):
    """The connection could not be attributed to a principal: no token, an unknown token, or a
    Unix peer whose credentials the policy does not accept.  Never retryable."""

    retryable = False


class AuthorizationError(_errors.AnatidError):
    """A principal asked for a tenant it is not allowed to touch.

    The server checks this before it resolves a handle, so a client authenticated for tenant 1
    cannot learn whether tenant 2 exists from the error it gets.  Never retryable.
    """

    retryable = False


class IdempotencyConflict(_errors.AnatidError):
    """An idempotency key was reused for a different request.

    A key identifies one write.  Sending it again with different arguments is a client bug, and
    returning the first write's result for the second write's arguments would hide it, so the
    server refuses.  ``key`` is the key and ``recorded_verb`` the verb it was first used for.
    Never retryable.
    """

    retryable = False

    def __init__(
        self, message: str, *, key: str | None = None, recorded_verb: str | None = None
    ) -> None:
        super().__init__(message)
        self.key = key
        self.recorded_verb = recorded_verb


class ShuttingDown(_errors.AnatidError):
    """The server has stopped accepting work and is draining.  Retryable against another
    instance, not against this one."""

    retryable = True


#: ``name -> class`` for every error the wire can name.  :func:`register_error` adds to it.
_ERROR_CLASSES: dict[str, type[BaseException]] = {}


def _claim(registry: Mapping[str, Any], name: str, owner: Any) -> None:
    """Refuse to give one wire tag to two different types.

    Found by measurement, not by reasoning: :class:`anatid.types.PruneReport` (what ``prune()``
    removed) and ``anatid.server.backup.PruneReport`` (what backup retention removed) share a
    class name, the second registration silently replaced the first, and the failure showed up
    three modules away as ``cannot rebuild PruneReport from the wire``.  A tag is the only thing
    the decoder has, so two claims on one tag is a bug wherever it happens, and it costs nothing
    to say so at import time instead.  Pass an explicit ``name`` to register both.
    """
    held = registry.get(name)
    if held is not None and held is not owner:
        raise ValueError(
            f"the wire tag {name!r} is already registered to "
            f"{getattr(held, '__module__', '?')}.{getattr(held, '__qualname__', held)}; "
            f"give one of the two an explicit name, because the decoder has only the tag to go on"
        )


def register_error(cls: type[BaseException], name: str | None = None) -> type[BaseException]:
    """Make ``cls`` reconstructable by :meth:`WireError.to_exception` under ``name``."""
    tag = name or cls.__name__
    _claim(_ERROR_CLASSES, tag, cls)
    _ERROR_CLASSES[tag] = cls
    return cls


for _cls in (
    _errors.AnatidError,
    _errors.SchemaVersionError,
    _errors.ConflictError,
    _errors.TenantIsolationError,
    _errors.BackupDestinationExists,
    _errors.ExtensionUnavailable,
    _errors.NotFoundError,
    _errors.ValidationError,
    _errors.RangeError,
    _errors.EmbeddingDimensionError,
    _errors.EmbeddingValueError,
    _errors.DuplicateIdError,
    _errors.IntegrityError,
    _errors.StaleIndexError,
    _errors.BruteForceCeilingError,
    _errors.IndexGenerationError,
    _errors.IndexValidationError,
    ProtocolError,
    FrameError,
    RemoteError,
    BusyError,
    DeadlineExceeded,
    AuthenticationError,
    AuthorizationError,
    IdempotencyConflict,
    ShuttingDown,
    ValueError,
    TypeError,
    KeyError,
    LookupError,
    NotImplementedError,
    PermissionError,
    TimeoutError,
    RuntimeError,
):
    register_error(_cls)
del _cls


# --------------------------------------------------------------------------- value codec

#: ``tag -> (encoder, decoder)`` for the value shapes that are not native JSON.
_CODECS: dict[str, tuple[Callable[[Any, Any], Any], Callable[[Any], Any]]] = {}

#: ``tag -> class`` for generically encoded frozen dataclasses.
_DATACLASSES: dict[str, type] = {}

#: ``tag -> class`` for enums.
_ENUMS: dict[str, type[Enum]] = {}

#: ``class -> tag``, a name index and nothing more.  The encoder used to tag by
#: ``type(value).__name__``, which meant the ``name`` argument of :func:`register_dataclass` moved
#: only the DECODER: a type registered under another name was written under its own and could not
#: be read back.  This is the other half.
#:
#: It is an index, not the registry: :func:`_tag_for` confirms the tag against ``_DATACLASSES`` or
#: ``_ENUMS`` before using it, so those two stay the single answer to "can this cross the wire"
#: and removing an entry from one of them still un-registers the type.
_TAGS: dict[type, str] = {}


def _tag_for(kind: type, registry: Mapping[str, Any]) -> str | None:
    """The wire tag ``kind`` is registered under in ``registry``, or None if it is not in it."""
    tag = _TAGS.get(kind)
    return tag if tag is not None and registry.get(tag) is kind else None


def register_dataclass(cls: type, name: str | None = None) -> type:
    """Let ``cls`` cross the wire as ``{"__anatid__": name, "v": {field: value}}``.

    The generic path encodes every field by name and rebuilds with ``cls(**fields)``, so it
    works for any dataclass whose ``__init__`` takes its fields as keywords -- which is every
    row type anatid returns.  A type whose constructor does not (``RecallHits``) gets an
    explicit codec instead.

    Registering a second class under a tag another class already holds raises: see :func:`_claim`.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    tag = name or cls.__name__
    _claim(_DATACLASSES, tag, cls)
    _DATACLASSES[tag] = cls
    _TAGS[cls] = tag
    return cls


def register_enum(cls: type[Enum], name: str | None = None) -> type[Enum]:
    """Let ``cls`` cross the wire as ``{"__anatid__": name, "v": <member value>}``."""
    tag = name or cls.__name__
    _claim(_ENUMS, tag, cls)
    _ENUMS[tag] = cls
    _TAGS[cls] = tag
    return cls


def register_codec(
    name: str,
    encode: Callable[[Any, Any], Any],
    decode: Callable[[Any], Any],
) -> None:
    """Give one type an explicit pair of functions instead of the generic dataclass path.

    ``name`` has to be the class's ``__name__``, because :func:`encode_value` looks the codec up
    by ``type(value).__name__``.  ``encode`` takes the value and the encoder's options and
    returns something JSON can hold; ``decode`` takes that back and rebuilds the value.  This is
    what a type needs when it is not a dataclass whose ``__init__`` takes its fields as keywords:
    :class:`anatid.types.RecallHits` is a list subclass, :class:`anatid.csr.ExpandPath` is a str
    subclass with extra attributes, and neither can be rebuilt from a field mapping.

    Registering the same name twice is allowed only when it is the same pair of functions, for
    the same reason :func:`_claim` gives: the decoder has only the tag.
    """
    held = _CODECS.get(name)
    if held is not None and held != (encode, decode):
        raise ValueError(
            f"the wire tag {name!r} already has a codec; give one of the two an explicit name, "
            f"because the decoder has only the tag to go on"
        )
    _CODECS[name] = (encode, decode)


@dataclass(frozen=True, slots=True)
class _EncodeOptions:
    """How :func:`encode_value` should render the shapes that have more than one rendering."""

    embeddings: str = "list"
    f32_min_length: int = 64


_DEFAULT_OPTIONS = _EncodeOptions()


def _encode_datetime(value: _dt.datetime) -> str:
    """Naive UTC ISO 8601.  An aware datetime is converted first, by anatid's own rule: the
    columns behind these values are ``TIMESTAMP`` with no zone, so a zone on the wire would be
    information the database cannot store."""
    naive = _types.to_utc_naive(value)
    if naive is None:  # to_utc_naive only returns None for None, which never reaches here
        raise ProtocolError("cannot encode a null datetime as a datetime value")
    return naive.isoformat()


def _decode_id(raw: Any) -> int:
    """Rebuild the integer a large id travelled as.

    A float is refused rather than rounded: ``8.8377183962144358e+17`` arriving here means some
    client already parsed the digits into a double, which is the corruption the tag exists to
    prevent, and accepting it would hide the one case worth reporting.
    """
    if isinstance(raw, (bool, float)):
        raise ProtocolError(f"an id travels as a decimal string, got {raw!r}")
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw), 10)
    except ValueError as exc:
        raise ProtocolError(f"{raw!r} is not an anatid id as a decimal string") from exc


def _decode_datetime(raw: Any) -> _dt.datetime:
    try:
        return _dt.datetime.fromisoformat(str(raw))
    except ValueError as exc:
        raise ProtocolError(f"{raw!r} is not an ISO 8601 datetime") from exc


def _float_array(seq: Iterable[Any]) -> list[float] | None:
    """``seq`` as a list of finite floats, or None when any element is not one.

    ``bool`` is a subclass of ``int``, so it is rejected explicitly: packing True as 1.0 would
    turn a list of flags into an embedding that decodes as floats.
    """
    out: list[float] = []
    for item in seq:
        if item is True or item is False or not isinstance(item, (int, float)):
            return None
        f = float(item)
        if not math.isfinite(f):
            return None
        out.append(f)
    return out


def _pack_f32(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def _unpack_f32(raw: Any) -> list[float]:
    try:
        blob = base64.b64decode(str(raw), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"f32 value is not valid base64: {exc}") from exc
    if len(blob) % 4:
        raise ProtocolError(f"f32 value is {len(blob)} bytes, not a whole number of float32s")
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _tag(name: str, value: Any) -> dict[str, Any]:
    return {TAG: name, "v": value}


def encode_value(value: Any, *, embeddings: str = "list", f32_min_length: int = 64) -> Any:
    """Render ``value`` as something :func:`json.dumps` accepts, tagging what JSON has no shape for.

    ``embeddings``
        ``"list"`` (the default) sends a float sequence as a JSON array, which is what you want
        when someone may read the frame.  ``"f32"`` sends any all-float sequence of at least
        ``f32_min_length`` elements as base64-encoded little-endian float32, measured at 46 us
        and 8,192 bytes for 1536 dimensions against 458 us and 20,606 bytes for the array.  It
        rounds to float32, which is lossless for a value that came out of anatid's ``FLOAT[N]``
        column and lossy for a float64 a caller computed, so it is opt in and the default is not.
    """
    opts = (
        _DEFAULT_OPTIONS
        if embeddings == "list" and f32_min_length == 64
        else _EncodeOptions(embeddings=embeddings, f32_min_length=f32_min_length)
    )
    return _encode(value, opts)


def _encode(value: Any, opts: _EncodeOptions) -> Any:
    if value is None or value is True or value is False:
        return value
    kind = type(value)
    if kind is str:
        return value
    if kind is int:
        # A 63-bit anatid id is not a JSON number any double-parsing client can read back.  The
        # magnitude decides, not the field name, because this codec does not know which field is
        # an id and the one nobody thought of is the one that breaks.
        return value if is_wire_safe_int(value) else _tag(ID_TAG, str(value))
    if kind is float:
        return value if math.isfinite(value) else _tag("float", repr(value))
    # Enum before str and int: EdgeType, Isolation, Severity and HealthReason are str subclasses,
    # so an isinstance(str) test would swallow them and lose the class.
    if isinstance(value, Enum):
        tag = _tag_for(type(value), _ENUMS)
        if tag is not None:
            return _tag(tag, value.value)
        return _encode(value.value, opts)
    if isinstance(value, _dt.datetime):
        return _tag("datetime", _encode_datetime(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _tag("bytes", base64.b64encode(bytes(value)).decode("ascii"))
    codec = _CODECS.get(kind.__name__)
    if codec is not None:
        return _tag(kind.__name__, codec[0](value, opts))
    if is_dataclass(value) and not isinstance(value, type):
        tag = _tag_for(kind, _DATACLASSES)
        if tag is not None:
            return _tag(tag, {f.name: _encode(getattr(value, f.name), opts) for f in fields(value)})
        raise ProtocolError(
            f"{kind.__name__} is a dataclass this protocol does not carry. Register it with "
            f"anatid.server.protocol.register_dataclass({kind.__name__}) before sending it."
        )
    if isinstance(value, (list, tuple)):
        if opts.embeddings == "f32" and len(value) >= opts.f32_min_length:
            floats = _float_array(value)
            if floats is not None:
                packed = _tag("f32", _pack_f32(floats))
                return packed if kind is list else _tag("tuple", packed)
        body = [_encode(v, opts) for v in value]
        return body if kind is list else _tag("tuple", body)
    if isinstance(value, (set, frozenset)):
        return _tag("set", [_encode(v, opts) for v in value])
    if isinstance(value, Mapping):
        if all(type(k) is str for k in value) and TAG not in value:
            return {k: _encode(v, opts) for k, v in value.items()}
        # A non-string key, or a user dictionary that happens to hold the tag key.  Entry pairs
        # keep both, and keep a user dictionary from ever being read as a tagged value.
        return _tag("map", [[_encode(k, opts), _encode(v, opts)] for k, v in value.items()])
    raise ProtocolError(
        f"{kind.__name__} has no wire representation. Verbs return JSON-native values, "
        f"datetimes, tuples, enums and the dataclasses in anatid.types; anything else has to "
        f"be registered."
    )


def decode_value(value: Any) -> Any:
    """Rebuild what :func:`encode_value` produced.  Structural: no type hint is consulted."""
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    if not isinstance(value, dict):
        return value
    name = value.get(TAG)
    if name is None:
        return {k: decode_value(v) for k, v in value.items()}
    if "v" not in value:
        raise ProtocolError(f"tagged value {name!r} has no 'v' member")
    raw = value["v"]
    if name == ID_TAG:
        return _decode_id(raw)
    if name == "datetime":
        return _decode_datetime(raw)
    if name == "float":
        return float(raw)
    if name == "bytes":
        try:
            return base64.b64decode(str(raw), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProtocolError(f"bytes value is not valid base64: {exc}") from exc
    if name == "f32":
        return _unpack_f32(raw)
    if name == "tuple":
        decoded = decode_value(raw)
        return tuple(decoded) if isinstance(decoded, (list, tuple)) else (decoded,)
    if name == "set":
        return {decode_value(v) for v in raw}
    if name == "map":
        return {_hashable(decode_value(k)): decode_value(v) for k, v in raw}
    codec = _CODECS.get(name)
    if codec is not None:
        return codec[1](raw)
    enum_cls = _ENUMS.get(name)
    if enum_cls is not None:
        return enum_cls(raw)
    cls = _DATACLASSES.get(name)
    if cls is not None:
        if not isinstance(raw, dict):
            raise ProtocolError(
                f"{name} should carry an object of fields, got {type(raw).__name__}"
            )
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            # A newer server sending a field this build does not have.  Dropping it is the
            # forward-compatible choice and the only one that does not raise on a rolling
            # upgrade; the field is named in the exception only when the type cannot be built
            # at all, which the TypeError below reports.
            raw = {k: v for k, v in raw.items() if k in known}
        try:
            return cls(**{k: decode_value(v) for k, v in raw.items()})
        except TypeError as exc:
            raise ProtocolError(f"cannot rebuild {name} from the wire: {exc}") from exc
    raise ProtocolError(
        f"unknown wire tag {name!r}. This build has no codec for it, which means the message "
        f"came from a newer anatid than the one reading it."
    )


def _hashable(key: Any) -> Any:
    return tuple(key) if isinstance(key, list) else key


# --- the value types the verbs actually return -------------------------------------------

for _dc in (
    _types.Namespace,
    _types.AsOf,
    _types.Memory,
    _types.Entity,
    _types.Edge,
    _types.Episode,
    _types.RecallHit,
    _types.Provenance,
    _types.ForgetReceipt,
    _types.PruneReport,
    _types.CorrectionReceipt,
    _types.FtsStatus,
    _types.SchemaInfo,
    _types.DoctorFinding,
    _types.DoctorReport,
):
    register_dataclass(_dc)
del _dc

for _en in (_types.Isolation, _types.EdgeType, _types.Severity):
    register_enum(_en)
del _en


def _encode_recall_hits(value: _types.RecallHits, opts: _EncodeOptions) -> Any:
    return {
        "hits": [_encode(h, opts) for h in value],
        "bm25_available": bool(value.bm25_available),
        "bm25_stale": bool(value.bm25_stale),
        "pending_fts_rows": int(value.pending_fts_rows),
        "arms": list(value.arms),
        "as_of": _encode(value.as_of, opts),
        "notes": list(value.notes),
        "seeds": list(value.seeds),
        "weights": dict(getattr(value, "weights", {}) or {}),
    }


def _decode_recall_hits(raw: Any) -> _types.RecallHits:
    if not isinstance(raw, dict):
        raise ProtocolError(f"RecallHits should carry an object, got {type(raw).__name__}")
    return _types.RecallHits(
        [decode_value(h) for h in raw.get("hits", ())],
        bm25_available=bool(raw.get("bm25_available", False)),
        bm25_stale=bool(raw.get("bm25_stale", False)),
        pending_fts_rows=int(raw.get("pending_fts_rows", 0)),
        arms=tuple(raw.get("arms", ())),
        as_of=decode_value(raw.get("as_of")) or _types.CURRENT,
        notes=tuple(raw.get("notes", ())),
        # A 0.3.0 server sends no seeds; the attribute defaults to () on the embedded handle too.
        seeds=tuple(raw.get("seeds", ())),
        # A server built before 0.4.2 sends no weights; the attribute is then empty.
        weights={str(k): float(v) for k, v in (raw.get("weights") or {}).items()},
    )


#: ``RecallHits`` is a list subclass with keyword-only extras, not a dataclass, so the generic
#: path cannot rebuild it.  It gets an explicit pair.
register_codec("RecallHits", _encode_recall_hits, _decode_recall_hits)


def _decode_path(raw: Any) -> _pathlib.Path:
    if not isinstance(raw, str):
        raise ProtocolError(f"Path should carry a string, got {type(raw).__name__}")
    return _pathlib.Path(raw)


#: A path is the string the sending side would print, and the receiving side rebuilds it with its
#: own flavour.  That is right for what crosses this wire: a backup report names a file on the
#: SERVER's disk, and a client that reads it is reading a name to show a person or to pass back in
#: another request, not a path it can open.  ``PosixPath`` and ``WindowsPath`` are both registered
#: under their own names so a report encoded on either kind of host decodes on the other.
for _path_cls in (_pathlib.Path, _pathlib.PosixPath, _pathlib.WindowsPath):
    register_codec(_path_cls.__name__, lambda v, _o: str(v), _decode_path)
del _path_cls


def dumps(value: Any, *, embeddings: str = "list", f32_min_length: int = 64) -> bytes:
    """Encode and serialise in one step.  ``allow_nan=False``: the output is always valid JSON,
    because a non-finite float is tagged by :func:`encode_value` before json sees it."""
    return json.dumps(
        encode_value(value, embeddings=embeddings, f32_min_length=f32_min_length),
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def loads(data: bytes | str) -> Any:
    """Parse and decode in one step.  Raises :class:`FrameError` on anything that is not JSON."""
    try:
        raw = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError(f"frame body is not UTF-8 JSON: {exc}") from exc
    return decode_value(raw)


# --------------------------------------------------------------------------- framing


def pack_frame(payload: bytes) -> bytes:
    """``len(payload)`` as a 4-byte big-endian prefix, then ``payload``."""
    n = len(payload)
    if n > MAX_FRAME_BYTES:
        raise FrameError(
            f"frame is {n} bytes, over the {MAX_FRAME_BYTES} byte limit; send it in pieces or "
            f"raise MAX_FRAME_BYTES on both ends"
        )
    return n.to_bytes(LENGTH_PREFIX_BYTES, "big") + payload


def frame_length(header: bytes, *, max_bytes: int = MAX_FRAME_BYTES) -> int:
    """The body length a 4-byte header announces, checked against ``max_bytes``.

    The check is the point.  Four bytes a client controls can ask for 4 GiB, and a server that
    allocates first and validates later is a one-packet denial of service.
    """
    if len(header) != LENGTH_PREFIX_BYTES:
        raise FrameError(f"frame header is {len(header)} bytes, expected {LENGTH_PREFIX_BYTES}")
    n = int.from_bytes(header, "big")
    if n > max_bytes:
        raise FrameError(
            f"frame announces {n} bytes, over the {max_bytes} byte limit for this connection"
        )
    return n


def read_frame(sock: Any, *, max_bytes: int = MAX_FRAME_BYTES) -> bytes | None:
    """Read one frame from a blocking socket.  None at a clean end of stream.

    A stream that ends in the middle of a frame raises :class:`FrameError`: half a message is
    not the same thing as no message, and treating it as end-of-stream would silently drop a
    write a client believes it sent.
    """
    header = _recv_exactly(sock, LENGTH_PREFIX_BYTES, allow_empty=True)
    if header is None:
        return None
    n = frame_length(header, max_bytes=max_bytes)
    if n == 0:
        return b""
    body = _recv_exactly(sock, n, allow_empty=False)
    # allow_empty=False raises rather than returning None, so the body is always bytes here.
    return b"" if body is None else body


def _recv_exactly(sock: Any, n: int, *, allow_empty: bool) -> bytes | None:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            if got == 0 and allow_empty:
                return None
            raise FrameError(f"stream ended after {got} of {n} bytes")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def write_frame(sock: Any, message: Any, **kw: Any) -> None:
    """Encode ``message`` and send it as one frame on a blocking socket."""
    sock.sendall(
        pack_frame(dumps(message.to_wire() if hasattr(message, "to_wire") else message, **kw))
    )


async def read_frame_async(reader: Any, *, max_bytes: int = MAX_FRAME_BYTES) -> bytes | None:
    """Read one frame from an ``asyncio.StreamReader``.  None at a clean end of stream.

    A stream that ends part-way through a frame raises :class:`FrameError`, for the reason
    :func:`read_frame` gives: half a message is not no message.
    """
    try:
        header = await reader.readexactly(LENGTH_PREFIX_BYTES)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise FrameError(
            f"stream ended after {len(exc.partial)} of {LENGTH_PREFIX_BYTES} header bytes"
        ) from exc
    n = frame_length(header, max_bytes=max_bytes)
    if n == 0:
        return b""
    try:
        return await reader.readexactly(n)
    except asyncio.IncompleteReadError as exc:
        raise FrameError(f"stream ended after {len(exc.partial)} of {n} body bytes") from exc


async def write_frame_async(writer: Any, message: Any, **kw: Any) -> None:
    """Encode ``message`` and send it as one frame on an ``asyncio.StreamWriter``."""
    body = dumps(message.to_wire() if hasattr(message, "to_wire") else message, **kw)
    writer.write(pack_frame(body))
    await writer.drain()


# --------------------------------------------------------------------------- messages


def new_request_id() -> str:
    """A fresh request id.  Opaque, 32 hex characters, generated client-side."""
    return uuid.uuid4().hex


class Status(str, Enum):
    """What a response is.

    ``OK``
        ``result`` holds the verb's return value.
    ``ERROR``
        ``error`` holds a :class:`WireError`.  Whether to retry is ``error.retryable``.
    ``BUSY``
        A queue was full and the write was not performed.  Its own status rather than an error
        class so a client can shed load on it without parsing a class name; ``error`` is filled
        in as well, with ``error_class == "BusyError"`` and a ``retry_after``.
    """

    OK = "ok"
    ERROR = "error"
    BUSY = "busy"


register_enum(Status)


@dataclass(frozen=True, slots=True)
class Request:
    """One verb call on its way to the server.

    ``verb``
        The name in the server's dispatch table (``"remember"``, ``"recall"``, ...).  Not a
        Python attribute lookup: the server matches it against an explicit table, so a request
        cannot reach a method the table does not name.
    ``tenant``
        The tenant this call is for.  Checked against the connection's principal before the
        server resolves a handle, so a client authenticated for one tenant cannot name another.
    ``args``
        Keyword arguments for the verb, decoded by this module's codec.
    ``idempotency_key``
        Optional, and only meaningful for a write.  A retry carrying the key of a write that
        already succeeded gets the original result back instead of writing twice.  See
        :class:`anatid.server.queue.IdempotencyStore`.
    ``deadline``
        Seconds of budget, measured by the SERVER from the moment it decodes the frame, not an
        absolute timestamp.  Absolute would be the more precise thing to send and the wrong
        thing to trust: a client clock a minute fast expires every request on arrival and one a
        minute slow never expires any.  A relative budget cannot be wrong in either direction.
        None means no deadline.
    ``protocol``
        :data:`PROTOCOL_VERSION`.  A mismatch is refused, not guessed at.
    """

    verb: str
    tenant: int
    args: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    deadline: float | None = None
    request_id: str = field(default_factory=new_request_id)
    protocol: int = PROTOCOL_VERSION

    def to_wire(self) -> dict[str, Any]:
        """The JSON-ready mapping for this request (values still need :func:`encode_value`)."""
        out: dict[str, Any] = {
            "protocol": int(self.protocol),
            "id": self.request_id,
            "tenant": int(self.tenant),
            "verb": self.verb,
            "args": dict(self.args),
        }
        if self.idempotency_key is not None:
            out["idempotency_key"] = self.idempotency_key
        if self.deadline is not None:
            out["deadline"] = float(self.deadline)
        return out

    @classmethod
    def from_wire(cls, raw: Any) -> "Request":
        """Rebuild from a decoded mapping, refusing anything malformed."""
        if not isinstance(raw, dict):
            raise ProtocolError(f"a request is an object, got {type(raw).__name__}")
        protocol = raw.get("protocol", PROTOCOL_VERSION)
        if not isinstance(protocol, int) or isinstance(protocol, bool):
            raise ProtocolError(f"protocol must be an integer, got {protocol!r}")
        if protocol != PROTOCOL_VERSION:
            raise ProtocolError(
                f"request speaks protocol version {protocol}; this server speaks {PROTOCOL_VERSION}"
            )
        verb = raw.get("verb")
        if not isinstance(verb, str) or not verb:
            raise ProtocolError("a request needs a non-empty 'verb'")
        tenant = raw.get("tenant")
        if not isinstance(tenant, int) or isinstance(tenant, bool):
            raise ProtocolError(f"a request needs an integer 'tenant', got {tenant!r}")
        args = raw.get("args", {})
        if not isinstance(args, dict):
            raise ProtocolError(
                f"'args' is an object of keyword arguments, got {type(args).__name__}"
            )
        if any(not isinstance(k, str) for k in args):
            raise ProtocolError("every key in 'args' has to be a string, they become keywords")
        key = raw.get("idempotency_key")
        if key is not None and not isinstance(key, str):
            raise ProtocolError(
                f"'idempotency_key' is a string or absent, got {type(key).__name__}"
            )
        deadline = raw.get("deadline")
        if deadline is not None:
            if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
                raise ProtocolError(
                    f"'deadline' is a number of seconds or absent, got {deadline!r}"
                )
            deadline = float(deadline)
            if deadline <= 0 or not math.isfinite(deadline):
                raise ProtocolError(
                    f"'deadline' must be a finite positive number, got {deadline!r}"
                )
        rid = raw.get("id", "")
        return cls(
            verb=verb,
            tenant=int(tenant),
            args=args,
            idempotency_key=key,
            deadline=deadline,
            request_id=rid if isinstance(rid, str) else "",
            protocol=protocol,
        )

    def encode(self, **kw: Any) -> bytes:
        """This request as one framed message, ready to write to a stream."""
        return pack_frame(dumps(self.to_wire(), **kw))

    @classmethod
    def decode(cls, body: bytes | str) -> "Request":
        """Parse one frame BODY (no length prefix) into a request."""
        return cls.from_wire(loads(body))


#: Scalar attributes anatid's errors hang extra context on.  Copied into ``WireError.details``
#: when present, so ``EmbeddingDimensionError.expected`` or ``BruteForceCeilingError.rows``
#: survives the wire without this module knowing every error class by hand.
_DETAIL_FIELDS = (
    "found",
    "expected",
    "got",
    "index",
    "value",
    "field",
    "low",
    "high",
    "table",
    "id",
    "tenant_id",
    "rows",
    "ceiling",
    "generation",
    "key",
    "recorded_verb",
    "waited",
    "deadline",
    "depth",
    "max_depth",
)


@dataclass(frozen=True, slots=True)
class WireError:
    """A structured error, as it crosses the wire.

    ``error_class`` is the class name the server raised, not a category, so a client that has
    the class gets the class back and one that does not still knows what it was.  ``retryable``
    is decided by the server, because only the server knows whether anything was committed.
    The conflict triple (``resource``, ``expected_version``, ``current_version``) is carried
    explicitly so :class:`anatid.errors.ConflictError` arrives with the fields a compare-and-swap
    caller reasons about rather than as a sentence.
    """

    error_class: str
    message: str
    retryable: bool = False
    resource: str | None = None
    expected_version: int | None = None
    current_version: int | None = None
    attempt: int | None = None
    retry_after: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(cls, exc: BaseException) -> "WireError":
        """Describe ``exc`` for the wire, keeping the fields anatid's errors carry."""
        retryable = bool(getattr(exc, "retryable", False))
        details: dict[str, Any] = {}
        for name in _DETAIL_FIELDS:
            got = getattr(exc, name, None)
            if got is not None and isinstance(got, (str, int, float, bool)):
                details[name] = got
        return cls(
            error_class=type(exc).__name__,
            message=str(exc) or type(exc).__name__,
            retryable=retryable,
            resource=getattr(exc, "resource", None),
            expected_version=getattr(exc, "expected_version", None),
            current_version=getattr(exc, "current_version", None),
            attempt=getattr(exc, "attempt", None),
            retry_after=getattr(exc, "retry_after", None),
            details=details,
        )

    def to_exception(self) -> BaseException:
        """Rebuild the exception, as the original class where this build has it.

        A :class:`anatid.errors.ConflictError` comes back with its resource and both versions,
        so a client's ``except ConflictError as e: e.current_version`` keeps working across the
        wire.  A class this build does not know becomes :class:`RemoteError`, which keeps the
        name rather than guessing at a superclass.
        """
        cls = _ERROR_CLASSES.get(self.error_class)
        if cls is None:
            return RemoteError(
                self.message,
                error_class=self.error_class,
                retryable=self.retryable,
                details=self.details,
            )
        if cls is _errors.ConflictError or (
            isinstance(cls, type) and issubclass(cls, _errors.ConflictError)
        ):
            return cls(
                self.message,
                resource=self.resource,
                expected_version=self.expected_version,
                current_version=self.current_version,
                retryable=self.retryable,
                attempt=self.attempt,
            )
        if cls is BusyError:
            return BusyError(
                self.message,
                retry_after=self.retry_after if self.retry_after is not None else 0.05,
                tenant_id=self.details.get("tenant_id"),
                depth=self.details.get("depth"),
                max_depth=self.details.get("max_depth"),
            )
        try:
            exc = cls(self.message)
        except TypeError:
            return RemoteError(
                self.message,
                error_class=self.error_class,
                retryable=self.retryable,
                details=self.details,
            )
        for name, got in self.details.items():
            # Set what the one-argument constructor left as None.  Overwriting a value the
            # constructor did set would let the wire contradict the class; leaving a None in
            # place would drop the context the class exists to carry.
            if getattr(exc, name, None) is None:
                try:
                    setattr(exc, name, got)
                except AttributeError:  # slots, or a read-only property
                    pass
        return exc

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "class": self.error_class,
            "message": self.message,
            "retryable": bool(self.retryable),
        }
        optional = {
            "resource": self.resource,
            "expected_version": self.expected_version,
            "current_version": self.current_version,
            "attempt": self.attempt,
            "retry_after": self.retry_after,
        }
        out.update({k: v for k, v in optional.items() if v is not None})
        if self.details:
            out["details"] = dict(self.details)
        return out

    @classmethod
    def from_wire(cls, raw: Any) -> "WireError":
        if not isinstance(raw, dict):
            raise ProtocolError(f"an error is an object, got {type(raw).__name__}")
        name = raw.get("class")
        if not isinstance(name, str) or not name:
            raise ProtocolError("an error needs a non-empty 'class'")
        message = raw.get("message", "")
        details = raw.get("details", {})
        return cls(
            error_class=name,
            message=message if isinstance(message, str) else str(message),
            retryable=bool(raw.get("retryable", False)),
            resource=raw.get("resource"),
            expected_version=raw.get("expected_version"),
            current_version=raw.get("current_version"),
            attempt=raw.get("attempt"),
            retry_after=raw.get("retry_after"),
            details=dict(details) if isinstance(details, dict) else {},
        )


@dataclass(frozen=True, slots=True)
class Response:
    """What the server sends back for one request.

    ``request_id`` echoes the request's, so a client that pipelines several requests down one
    connection can match them up.
    """

    status: Status
    request_id: str = ""
    result: Any = None
    error: WireError | None = None
    protocol: int = PROTOCOL_VERSION

    @classmethod
    def ok(cls, result: Any = None, *, request_id: str = "") -> "Response":
        return cls(status=Status.OK, request_id=request_id, result=result)

    @classmethod
    def failure(cls, exc: BaseException, *, request_id: str = "") -> "Response":
        """An error response for ``exc``, BUSY when it is a :class:`BusyError`."""
        wire = WireError.from_exception(exc)
        status = Status.BUSY if isinstance(exc, BusyError) else Status.ERROR
        return cls(status=status, request_id=request_id, error=wire)

    @classmethod
    def busy(
        cls,
        message: str,
        *,
        retry_after: float,
        request_id: str = "",
        tenant_id: int | None = None,
        depth: int | None = None,
        max_depth: int | None = None,
    ) -> "Response":
        return cls.failure(
            BusyError(
                message,
                retry_after=retry_after,
                tenant_id=tenant_id,
                depth=depth,
                max_depth=max_depth,
            ),
            request_id=request_id,
        )

    @property
    def retryable(self) -> bool:
        """True when the server said this failure is worth sending again."""
        return bool(self.error is not None and self.error.retryable)

    @property
    def retry_after(self) -> float | None:
        """The wait the server suggested, in seconds, or None when it suggested none."""
        return None if self.error is None else self.error.retry_after

    def raise_for_status(self) -> Any:
        """The result, or the server's error raised as an exception in this process."""
        if self.status is Status.OK:
            return self.result
        if self.error is None:
            raise ProtocolError(f"response status is {self.status.value} but it carries no error")
        raise self.error.to_exception()

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "protocol": int(self.protocol),
            "id": self.request_id,
            "status": self.status.value,
        }
        if self.status is Status.OK:
            out["result"] = self.result
        if self.error is not None:
            out["error"] = self.error.to_wire()
        return out

    @classmethod
    def from_wire(cls, raw: Any) -> "Response":
        if not isinstance(raw, dict):
            raise ProtocolError(f"a response is an object, got {type(raw).__name__}")
        protocol = raw.get("protocol", PROTOCOL_VERSION)
        if protocol != PROTOCOL_VERSION:
            raise ProtocolError(
                f"response speaks protocol version {protocol}; this client speaks "
                f"{PROTOCOL_VERSION}"
            )
        status_raw = raw.get("status")
        try:
            status = Status(status_raw)
        except ValueError as exc:
            raise ProtocolError(f"unknown response status {status_raw!r}") from exc
        error = raw.get("error")
        rid = raw.get("id", "")
        return cls(
            status=status,
            request_id=rid if isinstance(rid, str) else "",
            result=raw.get("result"),
            error=None if error is None else WireError.from_wire(error),
            protocol=PROTOCOL_VERSION,
        )

    def encode(self, **kw: Any) -> bytes:
        """This response as one framed message."""
        return pack_frame(dumps(self.to_wire(), **kw))

    @classmethod
    def decode(cls, body: bytes | str) -> "Response":
        """Parse one frame BODY (no length prefix) into a response."""
        return cls.from_wire(loads(body))
