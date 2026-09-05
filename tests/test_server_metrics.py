"""Server metrics: what they measure, what they refuse to say, and what they cost.

Three properties are asserted against behaviour rather than against the collector's own opinion
of itself:

* a slow recall is attributable to an ARM.  ``test_a_recall_is_charged_to_its_arms`` drives a
  real recall through the dispatcher and checks that the vector, text, hydrate and about arms
  each got a measurement, that the graph arm did not because no seed entity was given, and that
  the arms add up to less than the request they came from;
* a refusal names WHO was refused.  A tenant whose queue is full is counted against that tenant,
  and a tenant that was not refused has no series at all;
* the log line cannot carry memory content.  ``test_the_log_never_leaks_what_was_written`` puts
  a distinctive string in every field a client controls -- content, entity name, episode text,
  query text, idempotency key, verb name, bearer token -- drives every path that logs, and greps
  the whole captured log output for the marker.

The cardinality tests are the other half of that last one.  A ``verb`` label comes off the wire
and a ``tenant`` id comes off the wire, so both are bounded here: an unknown verb becomes one
fixed label value and tenants past the ceiling collapse into ``other``.  A metric label is
memory the process holds until it exits, and a client that can mint label values can exhaust it.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import logging
import shutil
import socket
import tempfile
from pathlib import Path

import pytest

from anatid import DatabasePool
from anatid.server import auth, metrics as M, protocol, queue as Q, server as S

posix_only = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="needs Unix domain sockets or POSIX file semantics; not available on this platform",
)


DIM = 8
T0 = _dt.datetime(2026, 1, 1, 0, 0, 0)

#: The marker that must never reach a log line.  One prefix so a single grep covers content,
#: entity names, episode text, query text, idempotency keys, verb names and tokens.
MARK = "zqxleak7714"

#: An operator-supplied principal name.  Not secret, and logged verbatim on purpose: the log
#: line is supposed to say who made the request.  Kept distinct from :data:`MARK` so the two
#: halves of the boundary can be asserted separately.
OPERATOR_NAME = "principal-vfmoperator"

SOCKADDR_UN_MAX = 100

ANY_TENANT = auth.Principal(name="test", tenants=None)
ONLY_TENANT_1 = auth.Principal.for_tenants("only-1", [1])


@pytest.fixture
def sock_dir():
    """A directory short enough to hold a Unix socket path (``sockaddr_un`` is 104 bytes)."""
    base = Path(tempfile.mkdtemp(prefix="anatid-m-"))
    if len(str(base)) + len("/run/xxxxxxxx.sock") > SOCKADDR_UN_MAX:
        shutil.rmtree(base, ignore_errors=True)
        base = Path(tempfile.mkdtemp(prefix="anatid-m-", dir="/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def pool(tmp_path):
    with DatabasePool(
        str(tmp_path / "tenants" / "t_{tenant}.anatid"), embedding_dim=DIM, max_open=8
    ) as p:
        yield p


def build_server(pool, sock_dir, **config_kw):
    """A dispatcher-only server: the queue runs, no socket is bound."""
    config = S.ServerConfig(
        socket_path=sock_dir / "run" / "anatid.sock",
        max_depth=config_kw.pop("max_depth", 32),
        batch_max=config_kw.pop("batch_max", 8),
        workers=config_kw.pop("workers", 2),
        read_workers=2,
        tenants=(1, 2),
        shutdown_timeout=10.0,
        **config_kw,
    )
    srv = S.AnatidServer(pool=pool, config=config)
    srv.queue.start()
    srv.open_tenants()
    srv._status = "serving"  # start() also binds sockets; these tests do not need one
    return srv


@pytest.fixture
def metered(pool, sock_dir):
    """A running dispatcher with metrics attached, torn down cleanly.

    ``detach`` matters here and not only for tidiness: the arm probes and the ``metrics`` entry
    in :data:`~anatid.server.server.VERBS` are process-wide, so a fixture that leaked them would
    change what every later test in the session sees.
    """
    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv, slow_seconds=10.0)
    try:
        yield srv, metrics
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)
        srv._status = "stopped"


def vec(*values):
    out = list(values) + [0.0] * DIM
    return [float(x) for x in out[:DIM]]


def series_of(snapshot, name):
    """``snapshot``'s series for one family, as ``{label tuple: series dict}``."""
    for family in snapshot["families"]:
        if family["name"] == name:
            return {tuple(sorted(s["labels"].items())): s for s in family["series"]}
    return {}


def counts(snapshot, name):
    """``{label tuple: observation count}`` for a histogram family."""
    return {key: s["count"] for key, s in series_of(snapshot, name).items()}


def values(snapshot, name):
    """``{label tuple: value}`` for a counter or gauge family."""
    return {key: s["value"] for key, s in series_of(snapshot, name).items()}


# --------------------------------------------------------------------------- request latency


def test_a_request_lands_in_its_verb_and_outcome_series(metered):
    srv, metrics = metered
    written = srv.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "Ada likes coffee"}),
        ANY_TENANT,
    ).raise_for_status()
    srv.call(
        protocol.Request(verb="get", tenant=1, args={"memory_id": written.memory_id}), ANY_TENANT
    ).raise_for_status()

    seen = counts(metrics.snapshot(), "anatid_request_duration_seconds")
    assert seen[(("outcome", "ok"), ("verb", "remember"))] == 1
    assert seen[(("outcome", "ok"), ("verb", "get"))] == 1


def test_an_error_is_labelled_by_outcome_and_never_by_its_message(metered):
    srv, metrics = metered
    response = srv.call(
        protocol.Request(verb="provenance", tenant=1, args={"memory_id": 999_999}), ANY_TENANT
    )
    assert response.status is protocol.Status.ERROR
    assert response.error is not None and response.error.error_class == "NotFoundError"

    snapshot = metrics.snapshot()
    seen = counts(snapshot, "anatid_request_duration_seconds")
    assert seen[(("outcome", "not_found"), ("verb", "provenance"))] == 1
    # The outcome vocabulary is closed, so nothing a verb wrote into its message can become a
    # label value.
    for key in seen:
        assert dict(key)["outcome"] in M.OUTCOMES


def test_an_unknown_verb_cannot_mint_a_label(metered):
    srv, metrics = metered
    for i in range(5):
        srv.call(protocol.Request(verb=f"invented_{i}_{MARK}", tenant=1), ANY_TENANT)

    seen = counts(metrics.snapshot(), "anatid_request_duration_seconds")
    unknown = [key for key in seen if dict(key)["verb"] == M.UNKNOWN_VERB]
    assert len(unknown) == 1, "five invented verbs must collapse into one series"
    assert seen[unknown[0]] == 5
    assert MARK not in str(seen)


def test_a_tenant_past_the_ceiling_collapses_into_one_label():
    metrics = M.Metrics(max_tenant_labels=3, log_requests=False)
    for tenant_id in range(10):
        metrics.observe_queue_wait(tenant_id, 0.001)

    seen = counts(metrics.snapshot(), "anatid_queue_wait_seconds")
    labels = sorted(dict(key)["tenant"] for key in seen)
    assert labels == ["0", "1", "2", M.OVERFLOW_TENANT]
    assert seen[(("tenant", M.OVERFLOW_TENANT),)] == 7


# --------------------------------------------------------------------------- the recall arms


def test_a_recall_is_charged_to_its_arms(metered):
    srv, metrics = metered
    srv.call(
        protocol.Request(
            verb="remember",
            tenant=1,
            args={"content": "Ada likes coffee", "embedding": vec(1, 0), "entities": ["Ada"]},
        ),
        ANY_TENANT,
    ).raise_for_status()
    srv.call(
        protocol.Request(verb="recall", tenant=1, args={"query": "coffee", "embedding": vec(1, 0)}),
        ANY_TENANT,
    ).raise_for_status()

    snapshot = metrics.snapshot()
    arms = {dict(key)["arm"] for key in counts(snapshot, "anatid_recall_arm_duration_seconds")}
    assert {"vector", "vector_scan", "text", "hydrate", "about"} <= arms
    assert "graph" not in arms, "no seed entity was given, so the graph arm did not run"

    # The arms are inside the request they came from, so their total cannot exceed it.  This is
    # what makes "the vector arm is most of a recall" a statement about the same clock.
    arm_total = sum(
        s["sum"] for s in series_of(snapshot, "anatid_recall_arm_duration_seconds").values()
    )
    request_total = series_of(snapshot, "anatid_request_duration_seconds")[
        (("outcome", "ok"), ("verb", "recall"))
    ]["sum"]
    assert 0.0 < arm_total <= request_total


def test_the_graph_arm_appears_when_a_seed_entity_is_given(metered):
    srv, metrics = metered
    entity = srv.call(
        protocol.Request(verb="upsert_entity", tenant=1, args={"name": "Ada"}), ANY_TENANT
    ).raise_for_status()
    srv.call(
        protocol.Request(
            verb="remember",
            tenant=1,
            args={"content": "Ada likes coffee", "entities": ["Ada"]},
        ),
        ANY_TENANT,
    ).raise_for_status()
    srv.call(
        protocol.Request(
            verb="recall", tenant=1, args={"query": "coffee", "seed_entity": entity.entity_id}
        ),
        ANY_TENANT,
    ).raise_for_status()

    arms = {
        dict(key)["arm"] for key in counts(metrics.snapshot(), "anatid_recall_arm_duration_seconds")
    }
    assert "graph" in arms


def test_an_arm_run_outside_a_request_is_not_charged_to_one(metered, pool):
    """The arm probes are process-global; an in-process handle must not be measured by them."""
    _srv, metrics = metered
    db = pool.get(1)
    db.remember("Ada likes coffee", embedding=vec(1, 0), tenant=1)
    db.recall(query="coffee", embedding=vec(1, 0), tenant=1)

    assert counts(metrics.snapshot(), "anatid_recall_arm_duration_seconds") == {}


# --------------------------------------------------------------------------- the write queue


def test_queue_wait_is_measured_per_tenant(pool, sock_dir):
    srv = build_server(pool, sock_dir, workers=1, batch_max=2)
    metrics = M.attach(srv, slow_seconds=10.0)
    try:
        futures = [
            srv.submit_write(
                protocol.Request(verb="remember", tenant=1, args={"content": f"m{i}"}),
                ANY_TENANT,
            )
            for i in range(16)
        ]
        futures.append(
            srv.submit_write(
                protocol.Request(verb="remember", tenant=2, args={"content": "other"}),
                ANY_TENANT,
            )
        )
        Q.wait_all(futures, timeout=30.0)

        snapshot = metrics.snapshot()
        seen = counts(snapshot, "anatid_queue_wait_seconds")
        assert seen[(("tenant", "1"),)] == 16
        assert seen[(("tenant", "2"),)] == 1

        # One worker and a batch_max of 2 means the sixteenth write waited behind seven turns.
        # If the wait were not really measured every observation would sit in the first bucket.
        buckets = series_of(snapshot, "anatid_queue_wait_seconds")[(("tenant", "1"),)]["buckets"]
        assert buckets["0.001"] < 16, "some write must have waited more than a millisecond"
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)


def test_batch_size_records_what_a_worker_turn_took(pool, sock_dir):
    srv = build_server(pool, sock_dir, workers=1, batch_max=4)
    metrics = M.attach(srv, slow_seconds=10.0)
    try:
        futures = [
            srv.submit_write(
                protocol.Request(verb="remember", tenant=1, args={"content": f"m{i}"}),
                ANY_TENANT,
            )
            for i in range(24)
        ]
        Q.wait_all(futures, timeout=30.0)

        snapshot = metrics.snapshot()
        sizes = series_of(snapshot, "anatid_write_batch_size")[()]
        assert sizes["count"] >= 6, "24 writes at batch_max=4 is at least six turns"
        assert sizes["sum"] == 24.0, "every write is in exactly one turn"
        assert sizes["buckets"]["4"] == sizes["count"], "no turn exceeded batch_max"
        assert (
            series_of(snapshot, "anatid_write_batch_duration_seconds")[()]["count"]
            == (sizes["count"])
        )
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)


def test_backpressure_is_counted_against_the_tenant_that_was_refused(pool, sock_dir):
    """A full queue names the tenant.  A rate of 429s alone cannot say which one to shard."""
    srv = build_server(pool, sock_dir, max_depth=1)
    metrics = M.attach(srv, slow_seconds=10.0)
    # The workers are deliberately not running, so nothing drains and the second write for
    # tenant 1 meets a full queue.
    srv.queue.stop(timeout=0.1, drain=False)
    srv.queue._accepting = True
    try:
        srv.submit_write(
            protocol.Request(verb="remember", tenant=1, args={"content": "first"}), ANY_TENANT
        )
        with pytest.raises(protocol.BusyError):
            srv.submit_write(
                protocol.Request(verb="remember", tenant=1, args={"content": "second"}),
                ANY_TENANT,
            )
        srv.submit_write(
            protocol.Request(verb="remember", tenant=2, args={"content": "elsewhere"}),
            ANY_TENANT,
        )

        seen = values(metrics.snapshot(), "anatid_backpressure_rejections_total")
        assert seen == {(("tenant", "1"),): 1.0}, "tenant 2 was never refused"
    finally:
        metrics.detach()


def test_queue_depth_is_read_live_rather_than_accumulated(pool, sock_dir):
    srv = build_server(pool, sock_dir, max_depth=8)
    metrics = M.attach(srv, slow_seconds=10.0)
    srv.queue.stop(timeout=0.1, drain=False)
    srv.queue._accepting = True
    try:
        for i in range(3):
            srv.submit_write(
                protocol.Request(verb="remember", tenant=1, args={"content": f"m{i}"}),
                ANY_TENANT,
            )
        assert values(metrics.snapshot(), "anatid_queue_depth") == {(("tenant", "1"),): 3.0}

        # A gauge, not a counter: after the queue drains the depth is what it is now, not the
        # sum of what it has ever been.
        srv.queue.start()
        srv.queue.drain(timeout=30.0)
        assert values(metrics.snapshot(), "anatid_queue_depth") == {}
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)


# --------------------------------------------------------------------------- idempotency


def test_a_replayed_write_is_counted_as_a_hit_not_a_second_write(metered):
    srv, metrics = metered
    request = protocol.Request(
        verb="remember", tenant=1, args={"content": "once"}, idempotency_key="k-1"
    )
    first = srv.call(request, ANY_TENANT).raise_for_status()
    second = srv.call(
        protocol.Request(
            verb="remember", tenant=1, args={"content": "once"}, idempotency_key="k-1"
        ),
        ANY_TENANT,
    ).raise_for_status()
    assert first.memory_id == second.memory_id

    snapshot = metrics.snapshot()
    assert values(snapshot, "anatid_idempotency_lookups_total") == {(("verb", "remember"),): 2.0}
    assert values(snapshot, "anatid_idempotency_hits_total") == {(("verb", "remember"),): 1.0}
    assert values(snapshot, "anatid_writes_replayed_total")[()] == 1.0


def test_a_write_with_no_key_is_not_counted_as_a_lookup(metered):
    srv, metrics = metered
    srv.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "unkeyed"}), ANY_TENANT
    ).raise_for_status()
    assert values(metrics.snapshot(), "anatid_idempotency_lookups_total") == {}


# --------------------------------------------------------------------------- conflicts


def test_a_stale_compare_and_swap_is_counted_as_a_conflict_that_must_not_be_retried(metered):
    srv, metrics = metered
    written = srv.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "Ada likes coffee"}),
        ANY_TENANT,
    ).raise_for_status()
    response = srv.call(
        protocol.Request(
            verb="update",
            tenant=1,
            args={
                "memory_id": written.memory_id,
                "content": "Ada drinks tea",
                "expected_version": written.version + 5,
            },
        ),
        ANY_TENANT,
    )
    assert response.error is not None
    assert response.error.error_class == "ConflictError"

    seen = values(metrics.snapshot(), "anatid_conflicts_total")
    # retryable=false is the whole point of the label: an identical retry fails identically,
    # so a dashboard that lumped this with an MVCC abort would advise exactly the wrong thing.
    assert seen == {(("retryable", "false"), ("verb", "update")): 1.0}


# --------------------------------------------------------------------------- exposition


def test_the_exposition_is_well_formed_prometheus_text(metered):
    srv, metrics = metered
    srv.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "Ada likes coffee"}),
        ANY_TENANT,
    ).raise_for_status()

    text = metrics.render()
    families: dict[str, str] = {}
    samples: list[str] = []
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ", 3)
            families[name] = kind
        elif line.startswith("# HELP "):
            assert len(line.split(" ", 3)) == 4, f"HELP line with no text: {line!r}"
        elif line:
            samples.append(line)
    assert families["anatid_request_duration_seconds"] == "histogram"
    assert families["anatid_up"] == "gauge"
    assert families["anatid_writes_submitted_total"] == "counter"

    # Every sample belongs to a declared family, which is what a scraper requires.
    for sample in samples:
        name = sample.split("{")[0].split(" ")[0]
        base = name.removesuffix("_bucket").removesuffix("_sum").removesuffix("_count")
        assert base in families, f"{sample!r} has no TYPE line"


def test_histogram_buckets_are_cumulative_and_end_at_the_count(metered):
    srv, metrics = metered
    for i in range(5):
        srv.call(
            protocol.Request(verb="remember", tenant=1, args={"content": f"m{i}"}), ANY_TENANT
        ).raise_for_status()

    for series in series_of(metrics.snapshot(), "anatid_request_duration_seconds").values():
        counted = list(series["buckets"].values())
        assert counted == sorted(counted), "cumulative buckets never decrease"
        assert series["buckets"]["+Inf"] == series["count"]


def test_json_and_prometheus_carry_the_same_numbers(metered):
    srv, metrics = metered
    srv.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "Ada likes coffee"}),
        ANY_TENANT,
    ).raise_for_status()

    text = metrics.render()
    snapshot = metrics.snapshot()
    for family in snapshot["families"]:
        if family["type"] != "histogram":
            continue
        for series in family["series"]:
            labels = ",".join(f'{k}="{v}"' for k, v in series["labels"].items())
            prefix = f"{family['name']}_count" + (("{" + labels + "}") if labels else "")
            assert f"{prefix} {series['count']}" in text


def test_the_snapshot_crosses_the_wire_unchanged(metered):
    srv, metrics = metered
    srv.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "Ada likes coffee"}),
        ANY_TENANT,
    ).raise_for_status()

    snapshot = metrics.snapshot()
    assert protocol.loads(protocol.dumps(snapshot)) == snapshot


# --------------------------------------------------------------------------- the surfaces


def test_the_metrics_verb_answers_over_the_dispatcher(metered):
    srv, metrics = metered
    srv.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "Ada likes coffee"}),
        ANY_TENANT,
    ).raise_for_status()

    payload = srv.call(protocol.Request(verb="metrics", tenant=1), ANY_TENANT).raise_for_status()
    assert {f["name"] for f in payload["families"]} == {
        f["name"] for f in metrics.snapshot()["families"]
    }


def test_the_metrics_verb_is_gone_after_detach(pool, sock_dir):
    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv)
    try:
        assert "metrics" in S.VERBS
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)
    assert "metrics" not in S.VERBS, "an unattached server must not advertise a verb it lacks"


def test_a_scoped_principal_sees_only_its_own_tenants(metered):
    srv, _metrics = metered
    for tenant_id in (1, 2):
        srv.call(
            protocol.Request(
                verb="remember", tenant=tenant_id, args={"content": f"in {tenant_id}"}
            ),
            ANY_TENANT,
        ).raise_for_status()

    everything = srv.call(protocol.Request(verb="metrics", tenant=1), ANY_TENANT).raise_for_status()
    scoped = srv.call(protocol.Request(verb="metrics", tenant=1), ONLY_TENANT_1).raise_for_status()

    assert {dict(k)["tenant"] for k in counts(everything, "anatid_queue_wait_seconds")} == {
        "1",
        "2",
    }
    assert {dict(k)["tenant"] for k in counts(scoped, "anatid_queue_wait_seconds")} == {"1"}
    # The process-wide numbers are not per-tenant and stay visible: they say nothing about
    # which tenants exist.
    assert values(scoped, "anatid_writes_submitted_total")[()] == 2.0


def test_the_overflow_bucket_is_hidden_from_a_scoped_principal():
    metrics = M.Metrics(max_tenant_labels=1, log_requests=False)
    metrics.observe_queue_wait(1, 0.001)
    metrics.observe_queue_wait(2, 0.001)  # past the ceiling, so it lands in "other"

    scoped = metrics.snapshot(tenants=[1])
    labels = {dict(k)["tenant"] for k in counts(scoped, "anatid_queue_wait_seconds")}
    assert labels == {"1"}, "'other' holds tenants this principal may not name"


def test_a_principal_cannot_ask_for_metrics_on_a_tenant_it_may_not_name(metered):
    srv, _metrics = metered
    response = srv.call(protocol.Request(verb="metrics", tenant=2), ONLY_TENANT_1)
    assert response.error is not None
    assert response.error.error_class == "AuthorizationError"


# --------------------------------------------------------------------------- HTTP


def _run(coro):
    return asyncio.run(coro)


async def _http(port, method, path, payload=b"", token=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        head = f"{method} {path} HTTP/1.1\r\nhost: localhost\r\ncontent-length: {len(payload)}\r\n"
        if token:
            head += f"authorization: Bearer {token}\r\n"
        head += "connection: close\r\n\r\n"
        writer.write(head.encode("latin-1") + payload)
        await writer.drain()
        raw = await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()
    header, _, body = raw.partition(b"\r\n\r\n")
    lines = header.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ")[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return status, headers, body


def test_metrics_is_served_over_http_in_the_prometheus_format(pool):
    seen: dict[str, object] = {}

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=None, http_host="127.0.0.1", http_port=0, tenants=(1,), workers=1
            ),
        )
        metrics = M.attach(server, slow_seconds=10.0)
        await server.start()
        port = server._servers[0].sockets[0].getsockname()[1]
        try:
            await _http(
                port,
                "POST",
                "/rpc",
                protocol.dumps(
                    protocol.Request(
                        verb="remember", tenant=1, args={"content": "over http"}
                    ).to_wire()
                ),
            )
            seen["metrics"] = await _http(port, "GET", "/metrics")
            seen["health"] = await _http(port, "GET", "/health")
            seen["missing"] = await _http(port, "GET", "/nope")
        finally:
            metrics.detach()
            await server.shutdown()

    _run(scenario())
    status, headers, body = seen["metrics"]
    assert status == 200
    assert headers["content-type"] == M.CONTENT_TYPE
    text = body.decode()
    assert "# TYPE anatid_request_duration_seconds histogram" in text
    assert 'anatid_request_duration_seconds_count{verb="remember",outcome="ok"} 1' in text

    # The other routes keep their own content type: the writer wrapper is matched on the body
    # it was given, not on a flag that could be read by the wrong reply.
    assert seen["health"][0] == 200
    assert seen["health"][1]["content-type"] == "application/json"
    assert seen["missing"][0] == 404
    assert seen["missing"][1]["content-type"] == "application/json"


def test_metrics_over_http_needs_the_token_when_rpc_does(pool):
    token = "test-token"
    seen: dict[str, object] = {}

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=None, http_host="127.0.0.1", http_port=0, tenants=(1,), workers=1
            ),
            authenticator=auth.BearerTokenAuthenticator({token: [1]}),
        )
        metrics = M.attach(server, slow_seconds=10.0)
        await server.start()
        port = server._servers[0].sockets[0].getsockname()[1]
        try:
            seen["anonymous"] = await _http(port, "GET", "/metrics")
            seen["authorized"] = await _http(port, "GET", "/metrics", token=token)
            seen["wrong"] = await _http(port, "GET", "/metrics", token="not-the-token")
        finally:
            metrics.detach()
            await server.shutdown()

    _run(scenario())
    assert seen["anonymous"][0] == 401
    assert seen["wrong"][0] == 401
    assert seen["authorized"][0] == 200
    assert MARK not in seen["anonymous"][2].decode()
    assert token not in seen["anonymous"][2].decode(), "the refusal never echoes the token"


# --------------------------------------------------------------------------- the log line


def parse_logfmt(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.split(" ") if "=" in part)


def request_lines(caplog) -> list[dict[str, str]]:
    return [
        parse_logfmt(record.getMessage())
        for record in caplog.records
        if record.name == "anatid.metrics" and record.getMessage().startswith("event=request")
    ]


def record_text(caplog) -> str:
    """Everything the metrics logger emitted, as one string to grep."""
    return "\n".join(r.getMessage() for r in caplog.records if r.name == "anatid.metrics")


def test_the_log_line_carries_the_shape_of_the_request(metered, caplog):
    srv, _metrics = metered
    caplog.set_level(logging.DEBUG)
    srv.call(
        protocol.Request(
            verb="remember",
            tenant=1,
            args={"content": "Ada likes coffee", "embedding": vec(1, 0)},
            request_id="req-abc-1",
        ),
        ANY_TENANT,
    ).raise_for_status()

    lines = request_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert line["verb"] == "remember"
    assert line["tenant"] == "1"
    assert line["outcome"] == "ok"
    assert float(line["duration_ms"]) > 0.0
    assert float(line["queue_ms"]) >= 0.0
    assert line["id"] == M.request_digest("req-abc-1")
    assert "req-abc-1" not in record_text(caplog), "the id itself is client-written"
    assert line["principal"] == "test"


def test_a_recall_log_line_breaks_the_duration_down_by_arm(metered, caplog):
    srv, _metrics = metered
    caplog.set_level(logging.DEBUG)
    srv.call(
        protocol.Request(
            verb="remember",
            tenant=1,
            args={"content": "Ada likes coffee", "embedding": vec(1, 0)},
        ),
        ANY_TENANT,
    ).raise_for_status()
    caplog.clear()
    srv.call(
        protocol.Request(verb="recall", tenant=1, args={"query": "coffee", "embedding": vec(1, 0)}),
        ANY_TENANT,
    ).raise_for_status()

    line = request_lines(caplog)[0]
    assert float(line["vector_ms"]) > 0.0
    assert float(line["text_ms"]) > 0.0
    assert "graph_ms" not in line, "the arm that did not run is not reported as zero"
    assert float(line["vector_ms"]) + float(line["text_ms"]) <= float(line["duration_ms"])


def test_the_json_format_carries_the_same_fields(pool, sock_dir, caplog):
    import json

    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv, log_format="json", slow_seconds=10.0)
    caplog.set_level(logging.DEBUG)
    try:
        srv.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "Ada likes coffee"}),
            ANY_TENANT,
        ).raise_for_status()
        payloads = [
            json.loads(r.getMessage())
            for r in caplog.records
            if r.name == "anatid.metrics" and r.getMessage().startswith("{")
        ]
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)
    assert len(payloads) == 1
    assert payloads[0]["event"] == "request"
    assert payloads[0]["verb"] == "remember"
    assert payloads[0]["tenant"] == 1
    assert payloads[0]["outcome"] == "ok"


def test_a_request_id_cannot_forge_a_second_log_line(metered, caplog):
    srv, _metrics = metered
    caplog.set_level(logging.DEBUG)
    srv.call(
        protocol.Request(
            verb="get",
            tenant=1,
            args={"memory_id": 1},
            request_id="a\nevent=request verb=get tenant=999 outcome=ok duration_ms=0.0",
        ),
        ANY_TENANT,
    )
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "anatid.metrics" and r.getMessage().startswith("event=request")
    ]
    assert len(lines) == 1
    assert "\n" not in lines[0]
    assert "tenant=999" not in lines[0]


def test_a_long_request_id_cannot_inflate_the_log_line(metered, caplog):
    """A four-kilobyte id logs as sixteen hex characters, whatever the client sent."""
    srv, _metrics = metered
    caplog.set_level(logging.DEBUG)
    srv.call(
        protocol.Request(verb="get", tenant=1, args={"memory_id": 1}, request_id="x" * 4096),
        ANY_TENANT,
    )
    logged = request_lines(caplog)[0]["id"]
    assert len(logged) == M.DIGEST_CHARS
    assert logged == M.request_digest("x" * 4096)
    assert "xxxx" not in record_text(caplog)


def test_the_logged_id_is_the_digest_a_client_can_compute(metered, caplog):
    """The id is greppable only if the client can reproduce it, so pin the definition."""
    srv, _metrics = metered
    caplog.set_level(logging.DEBUG)
    srv.call(
        protocol.Request(verb="get", tenant=1, args={"memory_id": 1}, request_id="req-abc-1"),
        ANY_TENANT,
    )
    expected = hashlib.sha256(b"req-abc-1").hexdigest()[: M.DIGEST_CHARS]
    assert request_lines(caplog)[0]["id"] == expected


def test_a_request_with_no_id_logs_no_digest(metered, caplog):
    """A constant in that field would read as one client sending everything."""
    srv, _metrics = metered
    caplog.set_level(logging.DEBUG)
    srv.call(
        protocol.Request(verb="get", tenant=1, args={"memory_id": 1}, request_id=""), ANY_TENANT
    )
    assert request_lines(caplog)[0]["id"] == ""


def test_logging_can_be_turned_off_without_losing_the_metrics(pool, sock_dir, caplog):
    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv, log_requests=False)
    caplog.set_level(logging.DEBUG)
    try:
        srv.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "quiet"}), ANY_TENANT
        ).raise_for_status()
        assert request_lines(caplog) == []
        assert counts(metrics.snapshot(), "anatid_request_duration_seconds")
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)


# --------------------------------------------------------------------------- redaction


@pytest.mark.parametrize("log_format", ["logfmt", "json"])
def test_the_log_never_leaks_what_was_written(pool, sock_dir, caplog, log_format):
    """The property the whole module is built around, checked by grep and not by inspection.

    Every field a client controls carries the same marker, every path that logs is driven, and
    the marker must appear nowhere in what the metrics logger emitted.  The assertion is
    scoped to ``anatid.metrics`` because that is the logger this module owns: the server's own
    failure log (``anatid.server``) prints an exception's MESSAGE, and a message is written by
    whichever verb raised it.  That is exactly why the metrics line records the exception CLASS
    instead, and the second half of this test proves the distinction rather than assuming it.

    Run against both log formats.  Today they serialise one shared ``fields`` mapping, so a
    leak could not reach one and miss the other, but that is an implementation detail and this
    guarantee is not: a later change that gives ``json`` its own field list has to keep it.
    """
    srv = build_server(pool, sock_dir, max_depth=1)
    # slow_seconds=0.0 makes every request "slow", so both log levels run.
    metrics = M.attach(srv, slow_seconds=0.0, log_format=log_format)
    caplog.set_level(logging.DEBUG)
    token = f"token-{MARK}"
    try:
        # A write, a read, a correction, a search, a graph write, raw source material.
        written = srv.call(
            protocol.Request(
                verb="remember",
                tenant=1,
                args={
                    "content": f"the secret is {MARK}",
                    "entities": [f"entity-{MARK}"],
                    "embedding": vec(0.5, 0.25),
                },
                idempotency_key=f"key-{MARK}",
            ),
            ANY_TENANT,
        ).raise_for_status()
        srv.call(
            protocol.Request(verb="get", tenant=1, args={"memory_id": written.memory_id}),
            ANY_TENANT,
        ).raise_for_status()
        srv.call(
            protocol.Request(
                verb="recall", tenant=1, args={"query": MARK, "embedding": vec(0.5, 0.25)}
            ),
            ANY_TENANT,
        ).raise_for_status()
        srv.call(
            protocol.Request(verb="episode", tenant=1, args={"content": f"raw source {MARK}"}),
            ANY_TENANT,
        ).raise_for_status()
        srv.call(
            protocol.Request(
                verb="supersede",
                tenant=1,
                args={"old_id": written.memory_id, "content": f"corrected {MARK}"},
            ),
            ANY_TENANT,
        ).raise_for_status()
        # Failure paths: an invented verb, a smuggled tenant, a stale version, a missing row,
        # a refusal, and a request whose own id carries the marker.
        srv.call(protocol.Request(verb=f"verb-{MARK}", tenant=1), ANY_TENANT)
        srv.call(
            protocol.Request(verb="get", tenant=1, args={"tenant": 2, "memory_id": 1}),
            ANY_TENANT,
        )
        srv.call(
            protocol.Request(
                verb="update",
                tenant=1,
                args={
                    "memory_id": written.memory_id,
                    "content": f"again {MARK}",
                    "expected_version": 99,
                },
            ),
            ANY_TENANT,
        )
        srv.call(
            protocol.Request(verb="provenance", tenant=1, args={"memory_id": 999_999}),
            ANY_TENANT,
        )
        srv.call(protocol.Request(verb="remember", tenant=2, args={"content": MARK}), ONLY_TENANT_1)
        srv.call(
            protocol.Request(verb="health", tenant=1, request_id=f"id-{MARK}"),
            auth.Principal(name=OPERATOR_NAME, tenants=None),
        )
        # And the metrics surfaces themselves.
        srv.call(protocol.Request(verb="metrics", tenant=1), ANY_TENANT).raise_for_status()
        rendered = metrics.render()
        snapshot = metrics.snapshot()
        traced = [entry.as_dict() for entry in metrics.slowest()]

        emitted = "\n".join(r.getMessage() for r in caplog.records if r.name == "anatid.metrics")
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)

    assert emitted, "the test proves nothing if nothing was logged"
    assert MARK not in emitted, "the metrics log carried something a client wrote"
    assert token not in emitted
    assert MARK not in rendered, "an exposition is scraped by anyone who can reach it"
    assert MARK not in str(snapshot)
    assert MARK not in str(traced), "the slow trace is not a place to put arguments"

    # The other half of the boundary.  ``principal`` is logged verbatim and is meant to be: an
    # authenticator names it, not a client.  BearerTokenAuthenticator derives "token:<digest>"
    # and never the token (the test below pins that), UnixPeerAuthenticator uses the peer uid,
    # and AllowAllAuthenticator uses a name the operator passed in.  Asserting the operator's
    # name DOES appear keeps that deliberate, so a change that starts digesting principal names
    # fails here and is decided rather than absorbed.
    assert OPERATOR_NAME in emitted, "the log line stopped saying who made the request"

    # The distinction this test rests on: the server's own failure log DOES print exception
    # messages, and a verb's message can quote what it was given.  Naming it here means a
    # future change that starts routing those messages through the metrics line fails above
    # rather than silently widening what the request log carries.
    server_log = "\n".join(r.getMessage() for r in caplog.records if r.name == "anatid.server")
    assert MARK in server_log, (
        "expected anatid.server to quote an argument in an error message; if it no longer does, "
        "this assertion is stale, not the one above"
    )


def test_a_bearer_token_never_reaches_a_principal_name(caplog):
    """The log line prints ``principal``, so the name must not be derived from the token."""
    token = f"secret-{MARK}"
    authenticator = auth.BearerTokenAuthenticator({token: [1]})
    principal = authenticator.authenticate(
        auth.ConnectionContext(transport="http", headers={"authorization": f"Bearer {token}"})
    )
    assert token not in principal.name
    assert MARK not in principal.name
    assert principal.name.startswith("token:")


# --------------------------------------------------------------------------- slow requests


def test_a_slow_request_is_logged_at_warning_and_counted(pool, sock_dir, caplog):
    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv, slow_seconds=0.0)
    caplog.set_level(logging.DEBUG)
    try:
        srv.call(
            protocol.Request(verb="remember", tenant=1, args={"content": "slow one"}), ANY_TENANT
        ).raise_for_status()
        warnings = [
            r for r in caplog.records if r.name == "anatid.metrics" and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert "slow=true" in warnings[0].getMessage()
        assert values(metrics.snapshot(), "anatid_slow_requests_total") == {
            (("verb", "remember"),): 1.0
        }
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)


def test_a_request_under_the_threshold_stays_at_the_normal_level(metered, caplog):
    srv, metrics = metered  # slow_seconds=10.0
    caplog.set_level(logging.DEBUG)
    srv.call(
        protocol.Request(verb="remember", tenant=1, args={"content": "quick"}), ANY_TENANT
    ).raise_for_status()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert values(metrics.snapshot(), "anatid_slow_requests_total") == {}


def test_the_trace_keeps_the_slowest_and_bounds_itself():
    metrics = M.Metrics(slow_seconds=0.001, slow_trace=3, log_requests=False)
    for i in range(20):
        metrics.observe_request(
            verb="get", tenant=1, outcome="ok", duration_s=0.01 * (i + 1), request_id=f"r{i}"
        )
    kept = metrics.slowest()
    assert len(kept) == 3
    assert [round(e.duration_s, 3) for e in kept] == [0.2, 0.19, 0.18]
    assert [e.request_digest for e in kept] == [M.request_digest(f"r{i}") for i in (19, 18, 17)]


def test_the_trace_records_where_a_slow_recall_spent_its_time(pool, sock_dir):
    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv, slow_seconds=0.0, log_requests=False)
    try:
        srv.call(
            protocol.Request(
                verb="remember",
                tenant=1,
                args={"content": "Ada likes coffee", "embedding": vec(1, 0)},
            ),
            ANY_TENANT,
        ).raise_for_status()
        srv.call(
            protocol.Request(
                verb="recall", tenant=1, args={"query": "coffee", "embedding": vec(1, 0)}
            ),
            ANY_TENANT,
        ).raise_for_status()
        traced = [e for e in metrics.slowest() if e.verb == "recall"]
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)

    assert len(traced) == 1
    assert set(traced[0].arms) >= {"vector", "text", "hydrate"}
    assert set(traced[0].as_dict()) == {
        "verb",
        "tenant",
        "outcome",
        "duration_ms",
        "queue_wait_ms",
        "arms_ms",
        "request_digest",
        "principal",
        "at",
    }


def test_a_trace_of_zero_keeps_nothing(metered):
    _srv, _ = metered
    metrics = M.Metrics(slow_seconds=0.0, slow_trace=0, log_requests=False)
    metrics.observe_request(verb="get", tenant=1, outcome="ok", duration_s=1.0)
    assert metrics.slowest() == []
    assert values(metrics.snapshot(), "anatid_slow_requests_total") == {(("verb", "get"),): 1.0}


# --------------------------------------------------------------------------- attach / detach


def test_detach_puts_every_probe_back(pool, sock_dir):
    from anatid import recall as recall_module

    srv = build_server(pool, sock_dir)
    before = {
        "call": srv.call,
        "acall": srv.acall,
        "run_read": srv.run_read,
        "submit_write": srv.submit_write,
        "submit": srv.queue.submit,
        "lookup": srv.idempotency.lookup,
        "vector_arm": recall_module.vector_arm,
    }
    metrics = M.attach(srv)
    try:
        assert srv.call is not before["call"]
        assert recall_module.vector_arm is not before["vector_arm"]
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)

    for name, original in before.items():
        current = {
            "call": srv.call,
            "acall": srv.acall,
            "run_read": srv.run_read,
            "submit_write": srv.submit_write,
            "submit": srv.queue.submit,
            "lookup": srv.idempotency.lookup,
            "vector_arm": recall_module.vector_arm,
        }[name]
        assert current == original, f"{name} was not restored"


def test_detaching_twice_is_harmless(pool, sock_dir):
    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv)
    metrics.detach()
    metrics.detach()
    srv.queue.stop(timeout=10.0)
    assert "metrics" not in S.VERBS


def test_attaching_the_same_metrics_twice_is_refused(pool, sock_dir):
    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv)
    try:
        with pytest.raises(RuntimeError):
            metrics.attach(srv)
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)


def test_two_servers_can_be_measured_at_once(pool, sock_dir, tmp_path):
    """The arm probes and the verb entry are process-wide, so they are refcounted."""
    from anatid import recall as recall_module

    original = recall_module.vector_arm
    first = build_server(pool, sock_dir)
    with DatabasePool(
        str(tmp_path / "other" / "t_{tenant}.anatid"), embedding_dim=DIM, max_open=4
    ) as second_pool:
        second = build_server(second_pool, sock_dir)
        a = M.attach(first)
        b = M.attach(second)
        try:
            first.call(
                protocol.Request(verb="remember", tenant=1, args={"content": "one"}), ANY_TENANT
            ).raise_for_status()
            second.call(
                protocol.Request(verb="remember", tenant=1, args={"content": "two"}), ANY_TENANT
            ).raise_for_status()
            assert counts(a.snapshot(), "anatid_request_duration_seconds")
            assert counts(b.snapshot(), "anatid_request_duration_seconds")
            a.detach()
            assert recall_module.vector_arm is not original, "b still wants the probes"
            assert "metrics" in S.VERBS
            b.detach()
        finally:
            first.queue.stop(timeout=10.0)
            second.queue.stop(timeout=10.0)
    assert recall_module.vector_arm is original
    assert "metrics" not in S.VERBS


def test_arms_can_be_skipped(pool, sock_dir):
    from anatid import recall as recall_module

    original = recall_module.vector_arm
    srv = build_server(pool, sock_dir)
    metrics = M.attach(srv, arms=False, slow_seconds=10.0)
    try:
        assert recall_module.vector_arm is original
        srv.call(
            protocol.Request(
                verb="remember",
                tenant=1,
                args={"content": "Ada likes coffee", "embedding": vec(1, 0)},
            ),
            ANY_TENANT,
        ).raise_for_status()
        srv.call(
            protocol.Request(verb="recall", tenant=1, args={"embedding": vec(1, 0)}), ANY_TENANT
        ).raise_for_status()
        snapshot = metrics.snapshot()
        assert counts(snapshot, "anatid_recall_arm_duration_seconds") == {}
        assert counts(snapshot, "anatid_request_duration_seconds"), "the rest still works"
    finally:
        metrics.detach()
        srv.queue.stop(timeout=10.0)


# --------------------------------------------------------------------------- end to end


@posix_only
def test_metrics_survive_a_real_socket_round_trip(pool, sock_dir):
    sock_path = sock_dir / "run" / "metered.sock"
    seen: dict[str, object] = {}

    async def scenario():
        server = S.AnatidServer(
            pool=pool, config=S.ServerConfig(socket_path=sock_path, tenants=(1,), workers=1)
        )
        metrics = M.attach(server, slow_seconds=10.0)
        await server.start()
        try:
            # The client blocks on recv, so it runs on a worker thread.  Driving it inline
            # would block the event loop that owes it the reply, and the read would time out
            # against a server that never got the chance to answer.
            def client():
                sock = S.connect_unix(str(sock_path))
                try:
                    protocol.write_frame(
                        sock,
                        protocol.Request(
                            verb="remember", tenant=1, args={"content": "over the socket"}
                        ),
                    )
                    protocol.Response.decode(protocol.read_frame(sock)).raise_for_status()
                    protocol.write_frame(sock, protocol.Request(verb="metrics", tenant=1))
                    seen["metrics"] = protocol.Response.decode(
                        protocol.read_frame(sock)
                    ).raise_for_status()
                finally:
                    sock.close()

            await asyncio.get_running_loop().run_in_executor(None, client)
        finally:
            metrics.detach()
            await server.shutdown()

    _run(scenario())
    payload = seen["metrics"]
    families = {f["name"]: f for f in payload["families"]}
    remember = [
        s
        for s in families["anatid_request_duration_seconds"]["series"]
        if s["labels"] == {"verb": "remember", "outcome": "ok"}
    ]
    assert remember and remember[0]["count"] == 1


def test_the_answers_do_not_change_when_metrics_are_attached(pool, sock_dir):
    """Instrumentation that altered a result would be worse than no instrumentation."""
    plain = build_server(pool, sock_dir)
    try:
        plain.call(
            protocol.Request(
                verb="remember",
                tenant=1,
                args={"content": "Ada likes coffee", "embedding": vec(1, 0), "memory_id": 11},
            ),
            ANY_TENANT,
        ).raise_for_status()
        without = plain.call(
            protocol.Request(
                verb="recall", tenant=1, args={"query": "coffee", "embedding": vec(1, 0)}
            ),
            ANY_TENANT,
        ).raise_for_status()
    finally:
        plain.queue.stop(timeout=10.0)

    instrumented = build_server(pool, sock_dir)
    metrics = M.attach(instrumented, slow_seconds=10.0)
    try:
        with_metrics = instrumented.call(
            protocol.Request(
                verb="recall", tenant=1, args={"query": "coffee", "embedding": vec(1, 0)}
            ),
            ANY_TENANT,
        ).raise_for_status()
    finally:
        metrics.detach()
        instrumented.queue.stop(timeout=10.0)

    assert [h.memory.memory_id for h in without] == [h.memory.memory_id for h in with_metrics]
    assert [round(h.score, 9) for h in without] == [round(h.score, 9) for h in with_metrics]


def test_the_instrumentation_does_not_halve_the_throughput(pool, sock_dir):
    """A loose floor, not a benchmark.

    The measured cost on this machine is about 0.8% of a ``get``; the numbers and the scripts
    that produced them are in :mod:`anatid.server.metrics`.  What a test can assert without
    becoming a flake on a loaded machine is that the cost has not become a FACTOR, so the bound
    is half the uninstrumented rate -- sixty times the measured overhead.

    The margin is measured rather than hoped for.  Fifteen runs of exactly this pairing put the
    ratio between 0.897 and 1.013 (median 0.992), and 25 consecutive runs of this test failed
    none, so 0.5 sits far below the worst observed sample.  The pairing is what makes that
    true: both rates are taken back to back in one process, so the comparison survives a
    machine whose absolute throughput moves 45% between runs.
    """
    import time

    def rate(server, calls):
        request = protocol.Request(verb="get", tenant=1, args={"memory_id": 11})
        for _ in range(50):
            server.call(request, ANY_TENANT)
        started = time.perf_counter()
        for _ in range(calls):
            server.call(request, ANY_TENANT)
        return calls / (time.perf_counter() - started)

    plain = build_server(pool, sock_dir)
    try:
        plain.call(
            protocol.Request(
                verb="remember", tenant=1, args={"content": "measured", "memory_id": 11}
            ),
            ANY_TENANT,
        ).raise_for_status()
        without = rate(plain, 500)
    finally:
        plain.queue.stop(timeout=10.0)

    instrumented = build_server(pool, sock_dir)
    metrics = M.attach(instrumented, log_requests=False, slow_seconds=10.0)
    try:
        with_metrics = rate(instrumented, 500)
    finally:
        metrics.detach()
        instrumented.queue.stop(timeout=10.0)

    assert with_metrics > without * 0.5, (
        f"{with_metrics:.0f} req/s instrumented against {without:.0f} req/s plain"
    )


def test_the_write_http_wrapper_leaves_every_other_route_alone(pool):
    """Installed once per process and never removed, so it must be inert for other replies."""
    seen: dict[str, object] = {}

    async def scenario():
        server = S.AnatidServer(
            pool=pool,
            config=S.ServerConfig(
                socket_path=None, http_host="127.0.0.1", http_port=0, tenants=(1,), workers=1
            ),
        )
        await server.start()  # no metrics attached at all
        port = server._servers[0].sockets[0].getsockname()[1]
        try:
            seen["health"] = await _http(port, "GET", "/health")
            seen["ready"] = await _http(port, "GET", "/ready")
            seen["metrics"] = await _http(port, "GET", "/metrics")
        finally:
            await server.shutdown()

    _run(scenario())
    assert seen["health"][0] == 200
    assert seen["health"][1]["content-type"] == "application/json"
    assert seen["ready"][0] == 200
    assert seen["metrics"][0] == 404, "an unattached server has no /metrics"


@posix_only
def test_connect_unix_is_still_the_documented_client(sock_dir):
    """Guards the import used above: a socket helper that moved would fail here, not in a loop."""
    assert callable(S.connect_unix)
    assert socket.AF_UNIX is not None
