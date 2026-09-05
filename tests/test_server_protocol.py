"""The wire contract: framing, the value codec, errors, and every dataclass in anatid.types.

The measurement these tests are designed around is in the module docstring of
``anatid.server.protocol``: while one process holds an anatid file read-write, no other process
can open it at all, read-only included.  So every read as well as every write crosses this wire,
and a value that does not survive the round trip is a verb that does not work over the server.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path

import pytest

from anatid import errors as anatid_errors
from anatid import types as anatid_types
from anatid.ids import new_id
from anatid.integrations.wire import JS_MAX_SAFE_INTEGER as P_JS_MAX
from anatid.integrations.wire import is_wire_safe_int
from anatid.server import protocol as P

T0 = _dt.datetime(2026, 1, 1, 12, 30, 45, 123456)


# --------------------------------------------------------------------------- framing


def test_frame_is_a_four_byte_big_endian_length_then_the_body():
    frame = P.pack_frame(b'{"a":1}')
    assert frame[:4] == (7).to_bytes(4, "big")
    assert frame[4:] == b'{"a":1}'
    assert P.frame_length(frame[:4]) == 7


def test_frame_length_refuses_a_header_that_asks_for_more_than_the_cap():
    header = (P.MAX_FRAME_BYTES + 1).to_bytes(4, "big")
    with pytest.raises(P.FrameError) as excinfo:
        P.frame_length(header)
    assert str(P.MAX_FRAME_BYTES) in str(excinfo.value)


def test_pack_frame_refuses_a_body_over_the_cap():
    with pytest.raises(P.FrameError):
        P.pack_frame(b"x" * (P.MAX_FRAME_BYTES + 1))


def test_read_frame_returns_none_at_a_clean_end_of_stream():
    a, b = socket.socketpair()
    a.close()
    try:
        assert P.read_frame(b) is None
    finally:
        b.close()


def test_read_frame_raises_on_a_stream_that_ends_mid_frame():
    a, b = socket.socketpair()
    try:
        a.sendall((100).to_bytes(4, "big") + b"only ten..")
        a.close()
        with pytest.raises(P.FrameError) as excinfo:
            P.read_frame(b)
        assert "10 of 100" in str(excinfo.value)
    finally:
        b.close()


def test_frames_round_trip_over_a_socket_pair():
    a, b = socket.socketpair()
    try:
        request = P.Request(verb="get", tenant=3, args={"memory_id": 9})
        a.sendall(request.encode())
        body = P.read_frame(b)
        assert body is not None
        back = P.Request.decode(body)
        assert (back.verb, back.tenant, back.args) == ("get", 3, {"memory_id": 9})
        assert back.request_id == request.request_id
    finally:
        a.close()
        b.close()


def test_two_frames_on_one_stream_stay_separate():
    a, b = socket.socketpair()
    try:
        a.sendall(P.Request(verb="get", tenant=1, args={"memory_id": 1}).encode())
        a.sendall(P.Request(verb="get", tenant=1, args={"memory_id": 2}).encode())
        first = P.Request.decode(P.read_frame(b) or b"")
        second = P.Request.decode(P.read_frame(b) or b"")
        assert first.args["memory_id"] == 1
        assert second.args["memory_id"] == 2
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------------------- scalars


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -17,
        2**63,
        0.0,
        -1.5,
        "",
        "a string with ü and \U0001f600",
        [],
        {},
        [1, "two", 3.0, None],
        {"a": 1, "b": [2, 3]},
    ],
)
def test_json_native_values_round_trip_unchanged(value):
    assert P.loads(P.dumps(value)) == value


def test_non_finite_floats_survive_and_the_json_stays_valid():
    for value in (float("nan"), float("inf"), float("-inf")):
        raw = P.dumps(value)
        json.loads(raw)  # would raise if the encoder had emitted bare NaN / Infinity
        back = P.loads(raw)
        assert math.isnan(back) if math.isnan(value) else back == value


def test_tuples_come_back_as_tuples_not_lists():
    assert P.loads(P.dumps((1, 2, 3))) == (1, 2, 3)
    assert isinstance(P.loads(P.dumps((1, 2))), tuple)
    assert isinstance(P.loads(P.dumps([1, 2])), list)


def test_nested_tuples_round_trip():
    value = ((1, 2), (3, (4, 5)))
    assert P.loads(P.dumps(value)) == value


def test_sets_and_bytes_round_trip():
    assert P.loads(P.dumps({1, 2, 3})) == {1, 2, 3}
    assert P.loads(P.dumps(b"\x00\xff binary")) == b"\x00\xff binary"


def test_datetimes_round_trip_as_naive_utc():
    assert P.loads(P.dumps(T0)) == T0
    aware = _dt.datetime(2026, 1, 1, 13, 30, 45, tzinfo=_dt.timezone(_dt.timedelta(hours=1)))
    assert P.loads(P.dumps(aware)) == _dt.datetime(2026, 1, 1, 12, 30, 45)


def test_a_dict_with_non_string_keys_round_trips():
    value = {1: "one", 2: "two"}
    assert P.loads(P.dumps(value)) == value


def test_a_user_dict_holding_the_tag_key_is_not_mistaken_for_a_tagged_value():
    value = {P.TAG: "Memory", "v": {"memory_id": 1}}
    back = P.loads(P.dumps(value))
    assert back == value
    assert isinstance(back, dict)


def test_an_unencodable_type_says_so_rather_than_dropping_it():
    class Opaque:
        pass

    with pytest.raises(P.ProtocolError) as excinfo:
        P.dumps(Opaque())
    assert "Opaque" in str(excinfo.value)


def test_an_unknown_tag_is_refused_rather_than_guessed_at():
    with pytest.raises(P.ProtocolError) as excinfo:
        P.decode_value({P.TAG: "SomethingFromTheFuture", "v": 1})
    assert "SomethingFromTheFuture" in str(excinfo.value)


# --------------------------------------------------------------------------- embeddings


def test_embeddings_default_to_a_readable_float_list():
    vector = [0.25, -0.5, 0.75]
    raw = P.dumps({"embedding": vector})
    assert b"[0.25,-0.5,0.75]" in raw
    assert P.loads(raw)["embedding"] == vector


def test_f32_encoding_is_accepted_on_decode_and_is_smaller_on_the_wire():
    vector = [i / 1024.0 for i in range(1536)]
    as_list = P.dumps(vector)
    as_f32 = P.dumps(vector, embeddings="f32")
    assert len(as_f32) < len(as_list) / 2
    back = P.loads(as_f32)
    assert len(back) == 1536
    # float32 rounding, which is what the FLOAT[N] column already holds.
    assert all(abs(a - b) < 1e-6 for a, b in zip(back, vector, strict=True))


def test_f32_leaves_short_sequences_and_non_float_sequences_alone():
    assert P.loads(P.dumps([1.0, 2.0], embeddings="f32")) == [1.0, 2.0]
    assert P.loads(P.dumps(["a"] * 100, embeddings="f32")) == ["a"] * 100
    # bool is an int, and packing True as 1.0 would turn flags into an embedding
    flags = [True, False] * 64
    assert P.loads(P.dumps(flags, embeddings="f32")) == flags


def test_a_float_tuple_stays_a_tuple_under_f32():
    vector = tuple(i / 1024.0 for i in range(128))
    back = P.loads(P.dumps(vector, embeddings="f32"))
    assert isinstance(back, tuple)
    assert len(back) == 128


def test_a_corrupt_f32_payload_is_refused():
    with pytest.raises(P.ProtocolError):
        P.decode_value({P.TAG: "f32", "v": "not base64 !!"})
    with pytest.raises(P.ProtocolError):
        P.decode_value({P.TAG: "f32", "v": "AAA="})  # 2 bytes, not a whole float32


# --------------------------------------------------------------------------- enums


@pytest.mark.parametrize(
    "member",
    [
        anatid_types.Isolation.FILE_PER_TENANT,
        anatid_types.Isolation.SCOPED,
        anatid_types.EdgeType.ABOUT,
        anatid_types.EdgeType.RELATES_TO,
        anatid_types.EdgeType.SUPERSEDES,
        anatid_types.Severity.ERROR,
        anatid_types.Severity.WARNING,
        P.Status.OK,
        P.Status.BUSY,
    ],
)
def test_enums_keep_their_class_across_the_wire(member):
    back = P.loads(P.dumps(member))
    assert back is member
    assert type(back) is type(member)


def test_a_str_enum_is_not_flattened_into_a_plain_string():
    # EdgeType subclasses str; an isinstance(str) test in the encoder would lose the class.
    back = P.loads(P.dumps({"edge_type": anatid_types.EdgeType.ABOUT}))
    assert back["edge_type"] is anatid_types.EdgeType.ABOUT


# --------------------------------------------------------------------------- the value types
#
# One populated instance of every dataclass in anatid.types, round-tripped field by field.


def _memory(**kw):
    base = {
        "memory_id": 7,
        "tenant_id": 3,
        "content": "Ada prefers dark roast",
        "kind": "fact",
        "embedding": (0.1, -0.2, 0.3, 0.4),
        "created_at": T0,
        "valid_from": T0,
        "valid_to": T0 + _dt.timedelta(days=1),
        "tx_from": T0,
        "tx_to": None,
        "writer": "agent-1",
        "episode_id": 11,
        "confidence": 0.75,
        "access_count": 4,
        "last_access_at": T0 + _dt.timedelta(hours=2),
        "version": 3,
    }
    base.update(kw)
    return anatid_types.Memory(**base)


def _entity():
    return anatid_types.Entity(
        entity_id=21,
        tenant_id=3,
        name="Ada",
        kind="person",
        valid_from=T0,
        valid_to=None,
        tx_from=T0,
        tx_to=None,
        writer="agent-1",
        episode_id=11,
        confidence=0.9,
    )


def _edge():
    return anatid_types.Edge(
        edge_id=31,
        edge_type=anatid_types.EdgeType.RELATES_TO,
        src=21,
        dst=22,
        tenant_id=3,
        weight=0.5,
        rel_kind="likes",
        valid_from=T0,
        valid_to=None,
        tx_from=T0,
        tx_to=None,
        writer="agent-1",
        episode_id=11,
        confidence=0.8,
        version=2,
    )


def _episode():
    return anatid_types.Episode(
        episode_id=11,
        tenant_id=3,
        content="the raw transcript",
        source="slack",
        kind="chat",
        created_at=T0,
        valid_from=T0,
        valid_to=None,
        tx_from=T0,
        tx_to=None,
        writer="agent-1",
    )


def _recall_hit():
    return anatid_types.RecallHit(
        memory=_memory(),
        score=0.0328,
        rank=1,
        vector_rank=2,
        text_rank=1,
        graph_rank=None,
        vector_score=0.87,
        text_score=3.21,
        about=("Ada", "coffee"),
    )


def _provenance():
    return anatid_types.Provenance(
        memory_id=7,
        chain=(_memory(), _memory(memory_id=6, version=1)),
        episodes=(_episode(),),
        edges=(_edge(),),
        writers=("agent-1", "agent-0"),
        versions=(_memory(version=1), _memory(version=3)),
    )


def _forget_receipt():
    return anatid_types.ForgetReceipt(
        memory_id=7,
        tenant_id=3,
        hard=True,
        at=T0,
        memories_deleted=1,
        about_edges_deleted=2,
        supersedes_edges_deleted=1,
        episodes_deleted=1,
        audit_rows_deleted=3,
        audit_rows_written=1,
        fts_rows_deleted=1,
        extra_rows_deleted=4,
        memory_versions_deleted=3,
        about_edge_versions_deleted=4,
        derived_rows_deleted=5,
        invalidated_generations=1,
        reason="gdpr erasure",
    )


def _prune_report():
    return anatid_types.PruneReport(
        dry_run=False,
        hard=True,
        at=T0,
        memory_ids=(7, 8, 9),
        receipts=(_forget_receipt(),),
        older_than=T0 - _dt.timedelta(days=30),
        max_access_count=2,
    )


def _fts_status():
    return anatid_types.FtsStatus(
        available=True,
        stale=True,
        indexed_rows=1000,
        current_rows=1200,
        pending_rows=200,
        indexed_at=T0,
        newest_row_at=T0 + _dt.timedelta(minutes=5),
        policy="rows>=10000 or ratio>=0.05 or age>=900s",
        indexed_max_id=999,
        current_max_id=1199,
    )


def _schema_info():
    return anatid_types.SchemaInfo(
        schema_version=4,
        created_at=T0,
        embedding_dim=1536,
        anatid_version="0.2.0",
        duckdb_version="1.5.5",
        fts_indexed_at=T0,
        fts_indexed_rows=1000,
        contract="one line\nanother line",
        extras={"a": 1, "b": [1, 2], "c": None},
    )


def _doctor_finding():
    return anatid_types.DoctorFinding(
        check="duplicate_memory_ids",
        severity=anatid_types.Severity.ERROR,
        count=2,
        detail="two rows share one (tenant_id, memory_id)",
        table="memories",
        samples=((3, 7), (3, 8)),
    )


def _doctor_report():
    return anatid_types.DoctorReport(
        checked_at=T0,
        schema_version=4,
        expected_schema_version=4,
        tenant_id=3,
        all_tenants=False,
        findings=(_doctor_finding(),),
        counts={"memories": 100, "entities": 20},
        checks_run=("dangling_edges", "nan_embeddings"),
        checks_skipped={"bm25_stale": "no index built"},
        duration_ms=12.5,
    )


def _namespace():
    return anatid_types.Namespace(
        tenant_id=3, label="acme", isolation=anatid_types.Isolation.FILE_PER_TENANT
    )


def _as_of():
    return anatid_types.AsOf(valid_time=T0, tx_time=T0 + _dt.timedelta(hours=1))


VALUE_TYPES = {
    "Namespace": _namespace,
    "AsOf": _as_of,
    "Memory": _memory,
    "Entity": _entity,
    "Edge": _edge,
    "Episode": _episode,
    "RecallHit": _recall_hit,
    "Provenance": _provenance,
    "ForgetReceipt": _forget_receipt,
    "PruneReport": _prune_report,
    "FtsStatus": _fts_status,
    "SchemaInfo": _schema_info,
    "DoctorFinding": _doctor_finding,
    "DoctorReport": _doctor_report,
}


def test_every_dataclass_in_anatid_types_is_covered_by_this_file():
    """A new row type must not be able to ship without a wire test.

    ``anatid.types`` is the return surface of every verb.  A dataclass added there and not
    registered in the codec would raise the first time a verb returned it over the server, in
    production rather than here.
    """
    declared = {
        name
        for name, obj in vars(anatid_types).items()
        if isinstance(obj, type) and is_dataclass(obj) and obj.__module__ == anatid_types.__name__
    }
    assert declared == set(VALUE_TYPES), (
        f"anatid.types dataclasses not round-tripped here: {sorted(declared - set(VALUE_TYPES))}; "
        f"names here that are not dataclasses in anatid.types: "
        f"{sorted(set(VALUE_TYPES) - declared)}"
    )


def test_two_classes_cannot_claim_one_wire_tag():
    """A tag is all the decoder has, so a second claim on one is a bug wherever it happens.

    Found by measurement, not by reasoning.  ``anatid.types.PruneReport`` (what ``prune()``
    removed from a tenant) and ``anatid.server.backup.PruneReport`` (what backup retention
    removed from a directory) share a class name.  The second registration silently replaced the
    first, and the failure surfaced three modules away as ``cannot rebuild PruneReport from the
    wire: missing 1 required positional argument: 'directory'``.
    """

    @dataclass(frozen=True)
    class Memory:  # deliberately the name of a type anatid already sends
        field_one: int = 1

    with pytest.raises(ValueError, match="already registered"):
        P.register_dataclass(Memory)

    with pytest.raises(ValueError, match="already has a codec"):
        P.register_codec("RecallHits", lambda v, o: None, lambda r: None)

    # Registering the same class twice under the same tag is not a collision; import order and a
    # reload should not be able to break a process.
    P.register_dataclass(anatid_types.Memory)


def test_an_explicit_name_moves_the_encoder_as_well_as_the_decoder():
    """Otherwise the escape hatch for a name collision does not actually work.

    ``register_dataclass(cls, name)`` set the DECODER's key while the encoder still tagged by
    ``type(value).__name__``, so a type registered under another name was written under its own
    and could not be read back at all.  Both PruneReports are the live case.
    """
    from anatid.server import backup as backup_module

    theirs = backup_module.PruneReport(directory=Path("/var/backups"), kept=(), pruned=())
    raw = P.encode_value(theirs)
    assert raw[P.TAG] == "BackupPruneReport"
    assert P.decode_value(P.loads(P.dumps(raw))) == theirs

    ours = anatid_types.PruneReport(dry_run=True, hard=False, at=_dt.datetime(2026, 1, 1))
    mine = P.encode_value(ours)
    assert mine[P.TAG] == "PruneReport"
    assert P.decode_value(P.loads(P.dumps(mine))) == ours


@pytest.mark.parametrize("name", sorted(VALUE_TYPES))
def test_each_value_type_round_trips_field_for_field(name):
    original = VALUE_TYPES[name]()
    back = P.loads(P.dumps(original))
    assert type(back) is type(original)
    for f in fields(original):
        got, want = getattr(back, f.name), getattr(original, f.name)
        assert got == want, f"{name}.{f.name}: {got!r} != {want!r}"
        assert type(got) is type(want), f"{name}.{f.name}: {type(got)} is not {type(want)}"
    assert back == original


def test_recall_hits_carries_its_extra_attributes_across_the_wire():
    hits = anatid_types.RecallHits(
        [_recall_hit()],
        bm25_available=True,
        bm25_stale=True,
        pending_fts_rows=42,
        arms=("vector", "text", "graph"),
        as_of=_as_of(),
        notes=("fts index is stale",),
    )
    back = P.loads(P.dumps(hits))
    assert isinstance(back, anatid_types.RecallHits)
    assert list(back) == list(hits)
    assert back.bm25_available is True
    assert back.bm25_stale is True
    assert back.pending_fts_rows == 42
    assert back.arms == ("vector", "text", "graph")
    assert back.as_of == _as_of()
    assert back.notes == ("fts index is stale",)
    assert back.memory_ids == hits.memory_ids


def test_an_empty_recall_hits_round_trips():
    back = P.loads(P.dumps(anatid_types.RecallHits()))
    assert isinstance(back, anatid_types.RecallHits)
    assert list(back) == []
    assert back.as_of is anatid_types.CURRENT or back.as_of == anatid_types.CURRENT


def test_a_field_a_newer_server_added_is_dropped_rather_than_raising():
    raw = {P.TAG: "Entity", "v": {"entity_id": 1, "tenant_id": 1, "name": "Ada", "colour": "red"}}
    back = P.decode_value(raw)
    assert isinstance(back, anatid_types.Entity)
    assert back.name == "Ada"


def test_a_memory_with_a_full_size_embedding_round_trips_both_ways():
    memory = _memory(embedding=tuple(i / 2048.0 for i in range(1536)))
    for encoding in ("list", "f32"):
        back = P.loads(P.dumps(memory, embeddings=encoding))
        assert back.embedding is not None
        assert len(back.embedding) == 1536
        assert isinstance(back.embedding, tuple)
        assert all(
            abs(a - b) < 1e-6 for a, b in zip(back.embedding, memory.embedding or (), strict=True)
        )


# --------------------------------------------------------------------------- requests


def test_a_request_round_trips_every_envelope_field():
    request = P.Request(
        verb="remember",
        tenant=5,
        args={"content": "hello", "embedding": [0.1, 0.2], "now": T0},
        idempotency_key="k-1",
        deadline=2.5,
        request_id="abc123",
    )
    back = P.Request.decode(request.encode()[4:])
    assert back == request
    assert back.args["now"] == T0


def test_a_request_defaults_to_the_current_protocol_and_a_fresh_id():
    a = P.Request(verb="get", tenant=1)
    b = P.Request(verb="get", tenant=1)
    assert a.protocol == P.PROTOCOL_VERSION
    assert a.request_id != b.request_id
    assert len(a.request_id) == 32


def test_a_request_from_another_protocol_version_is_refused():
    raw = {"protocol": P.PROTOCOL_VERSION + 1, "verb": "get", "tenant": 1, "args": {}}
    with pytest.raises(P.ProtocolError) as excinfo:
        P.Request.from_wire(raw)
    assert str(P.PROTOCOL_VERSION) in str(excinfo.value)


@pytest.mark.parametrize(
    "raw",
    [
        {"verb": "", "tenant": 1},
        {"verb": "get"},
        {"verb": "get", "tenant": "1"},
        {"verb": "get", "tenant": True},
        {"verb": "get", "tenant": 1, "args": []},
        {"verb": "get", "tenant": 1, "args": {"1": 2}, "idempotency_key": 7},
        {"verb": "get", "tenant": 1, "deadline": -1},
        {"verb": "get", "tenant": 1, "deadline": "soon"},
        [],
        "not an object",
    ],
)
def test_malformed_requests_are_refused_with_a_protocol_error(raw):
    with pytest.raises(P.ProtocolError):
        P.Request.from_wire(raw)


def test_a_deadline_is_relative_seconds_not_a_timestamp():
    """A clock a minute fast must not expire every request on arrival."""
    request = P.Request(verb="get", tenant=1, deadline=1.5)
    assert P.Request.decode(request.encode()[4:]).deadline == 1.5
    assert isinstance(request.deadline, float)


# --------------------------------------------------------------------------- responses


def test_an_ok_response_carries_the_result_and_the_request_id():
    response = P.Response.ok(_memory(), request_id="r1")
    back = P.Response.decode(response.encode()[4:])
    assert back.status is P.Status.OK
    assert back.request_id == "r1"
    assert back.result == _memory()
    assert back.raise_for_status() == _memory()


def test_a_conflict_error_keeps_its_versions_across_the_wire():
    original = anatid_errors.ConflictError(
        "memories(7) is at version 9 but the update expected version 7",
        resource="memories(7)",
        expected_version=7,
        current_version=9,
        retryable=False,
        attempt=2,
    )
    back = P.Response.decode(P.Response.failure(original, request_id="r2").encode()[4:])
    assert back.status is P.Status.ERROR
    assert back.retryable is False
    with pytest.raises(anatid_errors.ConflictError) as excinfo:
        back.raise_for_status()
    rebuilt = excinfo.value
    assert rebuilt.resource == "memories(7)"
    assert rebuilt.expected_version == 7
    assert rebuilt.current_version == 9
    assert rebuilt.retryable is False
    assert rebuilt.attempt == 2
    assert str(rebuilt) == str(original)


def test_a_retryable_conflict_stays_retryable():
    original = anatid_errors.ConflictError("write-write abort", retryable=True)
    back = P.Response.decode(P.Response.failure(original).encode()[4:])
    assert back.retryable is True
    assert back.error is not None and back.error.retryable is True


@pytest.mark.parametrize(
    "exc",
    [
        anatid_errors.NotFoundError("no memory 7 in tenant 3"),
        anatid_errors.TenantIsolationError("handle is for tenant 1"),
        anatid_errors.ValidationError("k must be at least 1"),
        anatid_errors.StaleIndexError("the bm25 index is stale"),
        anatid_errors.ExtensionUnavailable("no csr extension"),
        P.ProtocolError("unknown verb"),
        P.AuthorizationError("not authorized for tenant 2"),
        P.ShuttingDown("draining"),
        ValueError("a plain builtin"),
        NotImplementedError("owned_hnsw"),
    ],
)
def test_known_error_classes_arrive_as_themselves(exc):
    back = P.Response.decode(P.Response.failure(exc).encode()[4:])
    with pytest.raises(type(exc)) as excinfo:
        back.raise_for_status()
    assert str(excinfo.value) == str(exc)
    assert back.error is not None
    assert back.error.retryable == bool(getattr(exc, "retryable", False))


def test_structured_error_fields_survive():
    exc = anatid_errors.EmbeddingDimensionError("wrong dimension", expected=1536, got=8)
    back = P.Response.decode(P.Response.failure(exc).encode()[4:])
    assert back.error is not None
    assert back.error.details["expected"] == 1536
    assert back.error.details["got"] == 8
    rebuilt = back.error.to_exception()
    assert isinstance(rebuilt, anatid_errors.EmbeddingDimensionError)
    assert rebuilt.expected == 1536


def test_an_error_class_this_build_does_not_have_becomes_a_remote_error_that_keeps_the_name():
    wire = P.WireError(error_class="SomeFutureError", message="from a newer server", retryable=True)
    rebuilt = wire.to_exception()
    assert isinstance(rebuilt, P.RemoteError)
    assert rebuilt.error_class == "SomeFutureError"
    assert rebuilt.retryable is True
    assert str(rebuilt) == "from a newer server"


def test_a_busy_response_is_its_own_status_and_carries_the_suggested_wait():
    response = P.Response.busy(
        "tenant 3 has 256 writes queued", retry_after=0.25, tenant_id=3, depth=256, max_depth=256
    )
    back = P.Response.decode(response.encode()[4:])
    assert back.status is P.Status.BUSY
    assert back.retryable is True
    assert back.retry_after == 0.25
    with pytest.raises(P.BusyError) as excinfo:
        back.raise_for_status()
    assert excinfo.value.retry_after == 0.25
    assert excinfo.value.tenant_id == 3
    assert excinfo.value.depth == 256


def test_raise_for_status_returns_the_result_on_success():
    assert P.Response.ok([1, 2, 3]).raise_for_status() == [1, 2, 3]


def test_a_response_claiming_an_error_with_no_error_body_is_refused():
    with pytest.raises(P.ProtocolError):
        P.Response(status=P.Status.ERROR).raise_for_status()


def test_an_unknown_response_status_is_refused():
    with pytest.raises(P.ProtocolError):
        P.Response.from_wire({"protocol": P.PROTOCOL_VERSION, "status": "maybe", "id": ""})


def test_a_response_from_another_protocol_version_is_refused():
    with pytest.raises(P.ProtocolError):
        P.Response.from_wire({"protocol": P.PROTOCOL_VERSION + 1, "status": "ok", "id": ""})


# --------------------------------------------------------------------------- registration


def test_an_unregistered_dataclass_names_itself_and_says_how_to_fix_it():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class NotRegistered:
        x: int

    with pytest.raises(P.ProtocolError) as excinfo:
        P.dumps(NotRegistered(1))
    assert "NotRegistered" in str(excinfo.value)
    assert "register_dataclass" in str(excinfo.value)


def test_registering_a_dataclass_makes_it_round_trip():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Extra:
        a: int
        b: tuple[str, ...] = ()

    P.register_dataclass(Extra)
    try:
        assert P.loads(P.dumps(Extra(1, ("x", "y")))) == Extra(1, ("x", "y"))
    finally:
        P._DATACLASSES.pop("Extra", None)


def test_the_server_registers_the_derived_index_types_it_dispatches():
    """``index_health`` and ``maintain_indexes`` return these, so they have to cross the wire."""
    import anatid.server  # noqa: F401 - importing is what registers them
    from anatid.derived import Generation, HealthReason, HealthReport

    report = HealthReport(
        index_name="fts",
        tenant_id=None,
        reason=HealthReason.STALE_GENERATION,
        usable=True,
        generation=Generation(
            index_name="fts",
            generation=3,
            tenant_id=None,
            built_at=T0,
            watermark_id=99,
            watermark_ts=T0,
            validated=True,
            published=True,
            published_unvalidated=False,
            stats={"rows": 100},
            notes=None,
        ),
        pending_rows=12,
        tombstone_rows=1,
        pending_ratio=0.12,
        base_rows=100,
        age_seconds=42.0,
        detail="12 journal rows since the base",
    )
    back = P.loads(P.dumps(report))
    assert back == report
    assert back.reason is HealthReason.STALE_GENERATION


def test_expand_path_keeps_the_reason_it_carries_across_the_wire():
    """``stats`` and ``info`` both return one of these, so the wire has to hold it.

    ``ExpandPath`` is a ``str`` subclass, not a dataclass, so the generic path cannot rebuild it.
    Encoding it as a bare string would look like it worked -- the value still compares equal to
    ``"sql"`` -- and would drop the reason, which is the only part that tells an operator why the
    CSR index was not used.
    """
    import anatid.server  # noqa: F401 - importing is what registers the codec
    from anatid.csr import ExpandPath
    from anatid.derived import HealthReason

    path = ExpandPath(
        "sql",
        reason=HealthReason.STALE_GENERATION,
        detail="12 journal rows since the base",
        generation=3,
        tenant_id=1,
    )
    back = P.loads(P.dumps(path))
    assert isinstance(back, ExpandPath)
    assert back == "sql"
    assert back.reason is HealthReason.STALE_GENERATION
    assert back.detail == "12 journal rows since the base"
    assert back.generation == 3
    assert back.tenant_id == 1
    assert back.explain() == path.explain()


def test_an_expand_path_inside_a_stats_mapping_survives():
    """The shape the verb actually returns: a dict of counts with one of these in it."""
    import anatid.server  # noqa: F401
    from anatid.csr import ExpandPath
    from anatid.derived import HealthReason

    stats = {
        "memories": 7,
        "entities": 3,
        "expand_path": ExpandPath("csr", reason=HealthReason.FRESH, detail="", generation=2),
    }
    back = P.loads(P.dumps(stats))
    assert back["memories"] == 7
    assert back["expand_path"] == "csr"
    assert back["expand_path"].reason is HealthReason.FRESH


def test_register_codec_carries_a_type_the_generic_dataclass_path_cannot():
    class Weird(str):
        __slots__ = ("note",)

        def __new__(cls, value, *, note=""):
            self = super().__new__(cls, value)
            self.note = note
            return self

    P.register_codec(
        "Weird",
        lambda v, _opts: {"v": str(v), "note": v.note},
        lambda raw: Weird(raw["v"], note=raw["note"]),
    )
    try:
        back = P.loads(P.dumps(Weird("x", note="why")))
        assert isinstance(back, Weird)
        assert back == "x"
        assert back.note == "why"
    finally:
        P._CODECS.pop("Weird", None)


# --------------------------------------------------------------------------- ids on the wire
#
# anatid mints 63-bit ids and a JSON number is a double in JavaScript, so an id sent as a number
# comes back changed and nothing raises.  ``anatid.integrations.wire`` settled this for the MCP
# server and the Agents SDK; the server protocol is a third external boundary and the same rule
# applies here.  ``tests/test_wire_ids.py`` uses this same methodology at the other two.

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(
    NODE is None,
    reason=(
        "node is not on PATH; the round trip that proves a 63-bit id survives needs a real "
        "JavaScript parser, and a Python one would prove nothing (Python has no 2**53 limit)"
    ),
)

NODE_ROUND_TRIP = """
let raw = "";
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify(JSON.parse(raw)));
});
"""

#: Fields that carry an anatid id.  ``src`` and ``dst`` are the two that do not say so in their
#: names, which is the reason this list exists rather than a suffix test.
_ID_FIELDS = ("src", "dst")


def through_node(payload: str) -> str:
    """Hand ``payload`` to a real node, and return what its JSON parser gives back."""
    done = subprocess.run(
        [str(NODE), "-e", NODE_ROUND_TRIP],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def unsafe_ints(value, path: str = "$") -> list[tuple[str, int]]:
    """Every ``(json path, value)`` in ``value`` that a JSON number could not carry exactly."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [] if is_wire_safe_int(value) else [(path, value)]
    if isinstance(value, dict):
        found: list[tuple[str, int]] = []
        for key, item in value.items():
            found.extend(unsafe_ints(item, f"{path}.{key}"))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for index, item in enumerate(value):
            found.extend(unsafe_ints(item, f"{path}[{index}]"))
        return found
    return []


def assert_wire_safe(label: str, payload) -> None:
    """Fail with the path of every id that went out as a JSON number."""
    offenders = unsafe_ints(payload, label)
    assert not offenders, (
        f"{label} put {len(offenders)} integer(s) on the wire that a JSON number cannot carry "
        f"exactly; each of these is rounded by any JavaScript client: "
        + ", ".join(f"{path} = {value}" for path, value in offenders)
    )


def with_real_ids(value):
    """``value`` rebuilt with a freshly minted 63-bit id in every field that carries one.

    The fixtures above use 7 and 21 and 31, which fit in a JSON number and would make the sweep
    below pass without proving anything.  This is what makes it mean something, and
    ``test_the_sweep_would_notice_an_id_sent_as_a_number`` checks that it did its job.
    """
    if not is_dataclass(value) or isinstance(value, type):
        return value
    changes = {}
    for f in fields(value):
        got = getattr(value, f.name)
        if f.name == "samples" and isinstance(got, tuple):
            changes[f.name] = tuple(tuple(new_id() for _ in row) for row in got)
        elif isinstance(got, bool):
            continue
        elif isinstance(got, int) and (f.name.endswith("_id") or f.name in _ID_FIELDS):
            changes[f.name] = new_id()
        elif is_dataclass(got) and not isinstance(got, type):
            changes[f.name] = with_real_ids(got)
        elif isinstance(got, tuple) and got and all(is_dataclass(x) for x in got):
            changes[f.name] = tuple(with_real_ids(x) for x in got)
    return replace(value, **changes) if changes else value


def test_an_anatid_id_is_bigger_than_a_json_number_can_hold():
    """The premise.  If this ever fails, this whole section is about nothing."""
    minted = new_id()
    assert minted > P_JS_MAX, f"{minted} would have fitted in a JSON number"


@requires_node
def test_node_rounds_a_bare_63_bit_json_number():
    """The negative control.  Without it the round trips below could pass because node is
    lenient rather than because this codec is right."""
    memory_id = new_id()
    as_number = through_node(json.dumps({"memory_id": memory_id}))
    assert str(memory_id) not in as_number, "node kept the digits; the premise has changed"
    assert json.loads(as_number)["memory_id"] != memory_id
    as_string = through_node(json.dumps({"memory_id": str(memory_id)}))
    assert json.loads(as_string)["memory_id"] == str(memory_id)


def test_an_id_leaves_under_its_own_tag_as_a_decimal_string():
    memory_id = new_id()
    wire = json.loads(P.dumps({"memory_id": memory_id}))
    assert wire["memory_id"] == {P.TAG: P.ID_TAG, "v": str(memory_id)}


def test_a_number_a_json_client_can_carry_is_still_a_number():
    """Only the values that could not survive change shape.  A count is still a count."""
    wire = json.loads(P.dumps({"memories": 12, "at_the_limit": P_JS_MAX, "negative": -5}))
    assert wire == {"memories": 12, "at_the_limit": P_JS_MAX, "negative": -5}


def test_the_tag_comes_back_as_an_int_so_python_never_sees_an_id_string():
    memory_id = new_id()
    back = P.loads(P.dumps({"memory_id": memory_id, "ids": [memory_id], "by_id": {memory_id: 1}}))
    assert back["memory_id"] == memory_id
    assert isinstance(back["memory_id"], int)
    assert back["ids"] == [memory_id]
    assert back["by_id"] == {memory_id: 1}


def test_a_float_inside_an_id_tag_is_refused_rather_than_rounded():
    """A float here means some client already parsed the digits into a double.  That is the
    corruption the tag exists to prevent, so it is reported rather than accepted."""
    with pytest.raises(P.ProtocolError):
        P.loads(json.dumps({P.TAG: P.ID_TAG, "v": 8.8377183962144358e17}))
    with pytest.raises(P.ProtocolError):
        P.loads(json.dumps({P.TAG: P.ID_TAG, "v": "not an id"}))


def test_the_sweep_would_notice_an_id_sent_as_a_number():
    """The control for the sweep below: the values it sweeps really do carry 63-bit ids."""
    counted = {
        name: len(unsafe_ints(asdict(with_real_ids(build()))))
        for name, build in VALUE_TYPES.items()
    }
    assert counted["Memory"] >= 3, counted
    assert counted["DoctorFinding"] >= 4, counted
    assert sum(counted.values()) >= 25, counted


@pytest.mark.parametrize("name", sorted(VALUE_TYPES))
def test_no_value_type_puts_an_id_on_the_wire_as_a_number(name):
    original = with_real_ids(VALUE_TYPES[name]())
    assert_wire_safe(name, json.loads(P.dumps(original)))
    assert P.loads(P.dumps(original)) == original, "and it still round trips inside Python"


def test_an_id_in_an_error_detail_is_wire_safe_too():
    """``WireError.details`` carries ``id`` and ``tenant_id``, so it is the same boundary."""
    memory_id = new_id()
    failed = P.Response.failure(
        anatid_errors.DuplicateIdError("two rows", table="memories", id=memory_id, tenant_id=1)
    )
    assert_wire_safe("DuplicateIdError", json.loads(P.dumps(failed.to_wire())))
    rebuilt = P.Response.decode(P.dumps(failed.to_wire()))
    assert rebuilt.error is not None
    assert rebuilt.error.details["id"] == memory_id


def test_an_id_in_a_request_argument_is_wire_safe_in_the_other_direction_too():
    memory_id = new_id()
    request = P.Request(verb="get", tenant=1, args={"memory_id": memory_id})
    body = P.dumps(request.to_wire())
    assert_wire_safe("get request", json.loads(body))
    assert P.Request.decode(body).args["memory_id"] == memory_id


@requires_node
@pytest.mark.parametrize("name", sorted(VALUE_TYPES))
def test_every_value_type_survives_a_real_javascript_parser(name):
    sent = P.dumps(with_real_ids(VALUE_TYPES[name]())).decode("utf-8")
    assert json.loads(through_node(sent)) == json.loads(sent), (
        f"{name} changed in a JavaScript JSON parser"
    )


@requires_node
def test_an_id_through_node_comes_back_as_the_same_id():
    """End to end: encode, parse and re-serialise in real JavaScript, decode, compare."""
    ids = [new_id() for _ in range(12)]
    sent = P.dumps({"ids": ids}).decode("utf-8")
    back = P.loads(through_node(sent))
    assert back["ids"] == ids
    for value in ids:
        assert str(value) in sent, "the digits are in the frame, not a rounded double"
