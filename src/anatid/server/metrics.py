"""What the server is doing, in numbers, when someone asks why it is slow or refusing work.

The problem this solves
-----------------------
A server that answers "the request took 40 ms" has told you nothing you can act on.  Forty
milliseconds of what?  Waiting behind another tenant's batch in the write queue, scanning three
thousand embeddings in the vector arm, or hydrating rows the fusion had already ranked?  The
three have different fixes and no shared symptom, so one duration number sends whoever is
holding the pager to guess.  This module makes each of them a separate measurement.

The same argument applies to refusals.  ``BusyError`` means one tenant's queue is full.  Which
tenant, how full and for how long is the difference between "shard that tenant" and "raise
``batch_max``", and neither is inferable from a rate of 429s.

What it measures
----------------
Latency, split by where it went:

``anatid_request_duration_seconds{verb,outcome}``
    The whole server-side handling of one request: dispatch, tenant check, queue wait for a
    write, the verb itself.  Not the network -- the server cannot see the client's clock.
``anatid_recall_arm_duration_seconds{arm}``
    One histogram per retrieval arm: ``vector``, ``text``, ``graph``, plus ``vector_scan`` (the
    brute-force ceiling check, which is a query of its own), ``hydrate`` and ``about``.  A
    recall that got slower is now attributable to an arm instead of guessed at.  Measured on a
    3,001-memory tenant at 384 dimensions the vector arm is most of a recall, so "recall got
    slower and the vector arm did not" is a genuinely different fault from "both did".
``anatid_queue_wait_seconds{tenant}``
    How long a write sat in its tenant's queue before a worker opened its transaction.  Per
    tenant, because one number across tenants cannot show that one is starving behind another.
``anatid_write_batch_duration_seconds`` and ``anatid_write_batch_size``
    What a worker turn took, and how many writes it carried.

Refusals and repeats, counted where they happen:

``anatid_backpressure_rejections_total{tenant}``
    Writes refused because a tenant's queue was full.  The write did NOT happen.
``anatid_conflicts_total{verb,retryable}``
    ``retryable="true"`` is DuckDB's optimistic MVCC aborting a transaction; ``"false"`` is a
    compare-and-swap whose expected version is gone.  Same class, opposite advice, so they are
    separate series rather than one number that averages "retry" and "do not retry".
``anatid_idempotency_lookups_total{verb}`` / ``anatid_idempotency_hits_total{verb}``
    A hit is a client retry this server did not write twice.  The ratio is how much of the
    write traffic is duplicate.
``anatid_write_batch_splits_total``
    Server-side retries: a batch that failed and was re-run one write at a time.  A write-write
    conflict inside a batch usually shows up HERE and not in ``anatid_conflicts_total``, because
    the split re-runs the batch and the neighbour that was not at fault then succeeds.  A rising
    split count with a flat conflict count is contention the queue is absorbing.

Plus the queue's own counters read live (``submitted``, ``completed``, ``failed``, ``expired``,
``replayed``, ``batches``), per-tenant depth as a gauge, and process liveness.

Where you read it
-----------------
Three surfaces, one set of numbers::

    GET /metrics            Prometheus text format, on the HTTP listener
    {"verb": "metrics"}     the same numbers as JSON, over the socket protocol
    metrics.snapshot()      the same mapping, in process

and a log line per request::

    event=request verb=recall tenant=1 outcome=ok duration_ms=25.379 queue_ms=0.000
      vector_ms=20.114 text_ms=2.006 graph_ms=1.284 id=4f9a2c7e1b3d5068 principal=local

What the log line never contains
--------------------------------
Memory content, entity names, query text, embeddings, idempotency keys, bearer tokens.  Not by
filtering them out, which is a promise that decays the first time somebody adds a field, but
because the line is assembled from a closed list of values that cannot carry them:

* ``verb`` is looked up in :data:`~anatid.server.server.VERBS` and rendered as ``<unknown>``
  when it is not there, so a client cannot write into the log by inventing a verb name, nor
  blow up metric cardinality by sending a thousand of them;
* ``tenant`` is an integer, ``duration_ms`` and ``queue_ms`` are floats, ``outcome`` is one of
  the nine words in :data:`OUTCOMES`;
* ``error`` is the exception CLASS name, never its message -- a message is written by whichever
  verb raised it and can quote its arguments, so it is not a safe field for a line that
  promises this;
* ``principal`` is the name the AUTHENTICATOR gave the connection, which is operator code:
  :class:`BearerTokenAuthenticator <anatid.server.auth.BearerTokenAuthenticator>` derives it
  from eight hex characters of the token's SHA-256, never from the token, and
  :class:`UnixPeerAuthenticator <anatid.server.auth.UnixPeerAuthenticator>` from the peer uid.
  It is the one field whose value an operator chooses, and the rule that comes with that is
  short: do not build a principal name out of anything a client sent;
* ``id`` is :func:`request_digest` of the client's request id, sixteen hex characters, and NOT
  the id itself.  Echoing the id was the earlier design and it does not hold up.  A character
  class cannot rescue it: strip a request id to ``[A-Za-z0-9_.:+/=@-]`` and what is left is
  exactly the alphabet ``key=value`` forgery needs, so a client sending
  ``a\nevent=request verb=get tenant=999 outcome=ok`` still lands ``tenant=999`` in the line.
  A digest is sixteen characters of ``[0-9a-f]``, so it cannot forge a field, cannot carry
  content and cannot inflate the line.  Correlation survives because the digest is
  deterministic: a client holding its own request id computes the same value with
  ``anatid.server.metrics.request_digest(request_id)``, or with
  ``printf %s "$id" | shasum -a 256 | cut -c1-16``.

``args`` and the verb's return value are never read by this module at all.
``tests/test_server_metrics.py`` writes a memory whose content is a distinctive string, drives
every path that logs, and greps the whole captured log output for it.

Overhead
--------
The honest answer has two halves, because the end-to-end number and the thing it is supposed
to measure are different sizes.

The per-request cost is a fixed amount of work, and it is measurable.  On this machine (macOS
arm64, Python 3.12.9, best of 7 runs of 200,000 calls, no server and no socket, minus the cost
of an empty loop; ``scratchpad/m_metrics3.py``):

=======================================  =========
call                                     net cost
=======================================  =========
:meth:`Metrics.observe_request`, no log    0.51 us
:meth:`Metrics.observe_arm`                0.34 us
:meth:`Metrics.observe_queue_wait`         0.38 us
one ``logfmt`` line at INFO                5.12 us
one ``json`` line at INFO                  6.09 us
=======================================  =========

The log line is measured into a ``NullHandler``, so it is the cost of formatting the line and
not of whatever the operator points the logger at.  Multiply by what a request actually does:
a ``get`` is one ``observe_request``; a write adds one ``observe_queue_wait`` and a share of a
batch observation; a ``recall`` adds up to six ``observe_arm`` calls.  So the worst request in
the verb table pays about 3 us with the log off and about 8 us with it on.

Set against the measured service times -- a ``get`` is 360 us through the dispatcher in process
and about 1,000 us over the socket, a ``recall`` on a 3,001-memory tenant is 25,000 us -- that
is 0.8% of the cheapest call anatid has with the log on, and under a thousandth of a recall.

End to end, the ratio is confirmable but only from a PAIRED measurement: the two rates taken
back to back in one process, on one file, and compared to each other rather than to a number
recorded earlier.  Fifteen such pairs (500 ``get`` calls each after 50 warm-up calls, metrics
attached with the log off; ``scratchpad/m_margin.py``) put the instrumented rate at 0.897 to
1.013 of the plain rate, median 0.992.  So the end-to-end cost of the probes on the cheapest
verb is about 0.8%, which is what the per-observation arithmetic above predicts.

Unpaired comparisons across process launches do NOT resolve it, which is worth stating because
it is the obvious way to measure this and it does not work.  Driving a real socket at N=1500
per round, five rounds, three configurations in rotation
(``scratchpad/m_metrics2.py``), one configuration ranged 417 to 1001 requests per second across
its own five rounds; repeating the in-process dispatcher benchmark three times put "metrics
off" at 505.7 us and "metrics on" at 454.9 us for the same ``get``, the instrumented build
winning by ten times the effect it is supposed to lose by.  A machine with other work on it has
a run-to-run spread near 45%, and a 1% signal does not survive it.  An earlier revision of this
docstring quoted one ordered pass as ``988.7`` versus ``947.0`` requests per second; that was
the first configuration in the pass paying for a cold cache, and this section replaces it.

How it is wired
---------------
:func:`attach` installs the probes on an existing :class:`~anatid.server.server.AnatidServer` by
replacing bound methods on that one instance, and :meth:`Metrics.detach` puts them back.  Two
consequences worth stating rather than discovering:

* A server with no metrics attached pays nothing at all, which is why this is a separate module
  and not a field on :class:`~anatid.server.server.ServerConfig`.
* The per-arm probes are the exception.  Retrieval arms are module-level functions that
  :func:`anatid.recall.hybrid_recall` resolves from its own module globals at call time, so
  instrumenting them replaces those globals for the whole process rather than for one server.
  The replacement records nothing unless an instrumented request is running on the calling
  thread, so an in-process :class:`anatid.Anatid` in the same interpreter is unaffected beyond
  one extra call frame per arm.  Pass ``arms=False`` to skip it.

Every observation method is also callable directly (:meth:`Metrics.observe_request`,
:meth:`Metrics.observe_arm`, ...), so a later release that puts the calls in ``server.py``
itself can drop the wrappers and keep the metrics.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import hashlib
import importlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .protocol import BusyError

log = logging.getLogger("anatid.metrics")

__all__ = [
    "LATENCY_BUCKETS",
    "WAIT_BUCKETS",
    "BATCH_BUCKETS",
    "OUTCOMES",
    "ARMS",
    "CONTENT_TYPE",
    "METRICS_PATH",
    "UNKNOWN_VERB",
    "OVERFLOW_TENANT",
    "SlowRequest",
    "DIGEST_CHARS",
    "Metrics",
    "attach",
    "outcome_for",
    "request_digest",
]


# --------------------------------------------------------------------------- fixed vocabulary

#: Histogram bounds for a duration in seconds.  Dense from 100 us to 100 ms because that is
#: where anatid's calls live: a ``get`` over the socket is 1.0 ms at p50 and a ``recall`` on a
#: 3,001-memory tenant is 25 ms, so buckets starting at 5 ms would put both in the same one.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

#: Queue wait is a different distribution from service time: an empty queue answers in
#: microseconds and a full one in seconds, and the question is which side of a tenth of a
#: second it is on.
WAIT_BUCKETS: tuple[float, ...] = (
    0.0001,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

#: Writes taken from one tenant's queue in one worker turn.  ``batch_max`` defaults to 32.
BATCH_BUCKETS: tuple[float, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

#: Every value the ``outcome`` label can take.  Closed, so the label cannot grow without an edit
#: here, and so a dashboard can enumerate it.
OUTCOMES: tuple[str, ...] = (
    "ok",
    "busy",
    "conflict",
    "denied",
    "not_found",
    "invalid",
    "deadline",
    "shutting_down",
    "error",
)

#: Every value the ``arm`` label can take.
ARMS: tuple[str, ...] = ("vector", "vector_scan", "text", "graph", "hydrate", "about")

#: The exposition format Prometheus parses.  Served on ``GET /metrics``.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: The HTTP path :func:`attach` adds to the server's HTTP surface.
METRICS_PATH = "/metrics"

#: The ``verb`` label for a request naming a verb this server does not have.  A client supplies
#: that string, so it never reaches a label value or a log line unchanged.
UNKNOWN_VERB = "<unknown>"

#: The ``tenant`` label for every tenant past ``max_tenant_labels``.  A tenant id arrives on the
#: wire and is unbounded; a metric label must not be.
OVERFLOW_TENANT = "other"

#: Exception class name -> outcome.  Anything unlisted is ``"error"``, which is the honest
#: answer for a fault this module has no opinion about.
_OUTCOME_BY_ERROR: dict[str, str] = {
    "BusyError": "busy",
    "ConflictError": "conflict",
    "AuthenticationError": "denied",
    "AuthorizationError": "denied",
    "TenantIsolationError": "denied",
    "NotFoundError": "not_found",
    "ProtocolError": "invalid",
    "FrameError": "invalid",
    "ValidationError": "invalid",
    "RangeError": "invalid",
    "EmbeddingDimensionError": "invalid",
    "EmbeddingValueError": "invalid",
    "DuplicateIdError": "invalid",
    "IdempotencyConflict": "invalid",
    "DeadlineExceeded": "deadline",
    "ShuttingDown": "shutting_down",
}

#: The retrieval functions the arm probes replace: (module, attribute, arm label).
#: :func:`anatid.recall.hybrid_recall` resolves each of these from its own module globals at
#: call time, which is what makes replacing the global enough and editing the call site
#: unnecessary.  ``anatid.fts.search`` is reached as ``_fts.search`` on the module object, so
#: the same holds for it.
_ARM_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("anatid.recall", "vector_arm", "vector"),
    ("anatid.recall", "vector_scan_rows", "vector_scan"),
    ("anatid.recall", "graph_arm", "graph"),
    ("anatid.recall", "hydrate", "hydrate"),
    ("anatid.recall", "about_names", "about"),
    ("anatid.fts", "search", "text"),
)

#: Characters an operator-supplied log value may contain.  Everything else becomes ``_``.
#: The point is not tidiness: a newline would forge a second log line and a quote would break
#: whatever parses the first one.  It is applied to ``principal`` and to an exception class
#: name, both of which come from code the operator deploys.  It is NOT enough for a value a
#: client chooses, which is why a request id is digested rather than filtered; see
#: :func:`request_digest`.
_SAFE = re.compile(r"[^A-Za-z0-9_.:+/=@-]")

#: How many hex characters of a request id's digest the log line and the slow trace carry.
#: Sixteen is 64 bits, far more than a log file needs to tell two live requests apart and short
#: enough to read off a screen.
DIGEST_CHARS = 16


def outcome_for(response: Any) -> str:
    """Which of :data:`OUTCOMES` describes ``response``.

    Classified from the error CLASS, never from its message and never from the result.  A
    response carrying an error this build does not know about is ``"error"`` rather than a new
    label value: a client that can invent label values can exhaust the server's memory one time
    series at a time.
    """
    status = getattr(response, "status", None)
    name = getattr(status, "value", status)
    if name == "ok":
        return "ok"
    if name == "busy":
        return "busy"
    error = getattr(response, "error", None)
    error_class = getattr(error, "error_class", "") or ""
    return _OUTCOME_BY_ERROR.get(error_class, "error")


def _safe(text: Any, limit: int = 64) -> str:
    """``text`` reduced to characters that cannot break or forge a log line, then truncated.

    For operator-supplied values only.  A client-supplied one goes through
    :func:`request_digest`, because filtering characters out of a string still leaves the string.
    """
    cleaned = _SAFE.sub("_", str(text))
    return cleaned[:limit]


def request_digest(request_id: Any) -> str:
    """The :data:`DIGEST_CHARS` hex characters this module logs in place of a request id.

    The log line and the slow trace carry this rather than the id itself, so that nothing a
    client wrote reaches either one.  It is a truncated SHA-256 of the id's UTF-8 bytes and
    nothing else, so a client that wants to find its own request in a server log computes the
    same value::

        anatid.server.metrics.request_digest(request_id)
        printf %s "$request_id" | shasum -a 256 | cut -c1-16

    An empty id digests to the empty string rather than to the digest of nothing, because a
    request with no id has nothing to correlate and a constant in that field would read as one
    client sending everything.
    """
    text = "" if request_id is None else str(request_id)
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:DIGEST_CHARS]


def _ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f}"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _iso(unix_seconds: float) -> str:
    if not unix_seconds:
        return ""
    moment = _dt.datetime.fromtimestamp(unix_seconds, tz=_dt.timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _number(value: float) -> str:
    """A metric value as Prometheus writes it: no trailing ``.0``, no thousands separator."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: Mapping[str, str], extra: str = "") -> str:
    parts = [f'{name}="{_escape(value)}"' for name, value in labels.items()]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


# --------------------------------------------------------------------------- metric families
#
# A deliberately small implementation rather than prometheus_client: anatid's only runtime
# dependency is duckdb, and adding one so the server can count things would be a poor trade.
# What is here is the exposition format and nothing else -- no push gateway, no registry
# discovery, no exemplars.


class _Series:
    """One label combination of a histogram: per-bucket counts, sum and count."""

    __slots__ = ("counts", "total", "count")

    def __init__(self, buckets: int) -> None:
        self.counts = [0] * (buckets + 1)  # the last slot is +Inf
        self.total = 0.0
        self.count = 0


class _Family:
    """A named metric with a fixed label schema.

    The schema is fixed at construction because that is what bounds cardinality: a family that
    accepts whatever labels a caller passes is one client away from a million time series.
    """

    __slots__ = ("name", "kind", "description", "labels", "tenant_index")

    def __init__(self, name: str, kind: str, description: str, labels: Sequence[str] = ()) -> None:
        self.name = name
        self.kind = kind
        self.description = description
        self.labels = tuple(labels)
        self.tenant_index = self.labels.index("tenant") if "tenant" in self.labels else None


class _Scalar(_Family):
    """A counter or a gauge: one float per label combination."""

    __slots__ = ("values",)

    def __init__(self, name: str, kind: str, description: str, labels: Sequence[str] = ()) -> None:
        super().__init__(name, kind, description, labels)
        self.values: dict[tuple[str, ...], float] = {}


class _Histogram(_Family):
    """A histogram with explicit bucket bounds, shared by every label combination."""

    __slots__ = ("bounds", "series")

    def __init__(
        self,
        name: str,
        description: str,
        bounds: Sequence[float],
        labels: Sequence[str] = (),
    ) -> None:
        super().__init__(name, "histogram", description, labels)
        self.bounds = tuple(float(b) for b in bounds)
        self.series: dict[tuple[str, ...], _Series] = {}


# --------------------------------------------------------------------------- the slow trace


@dataclass(frozen=True, slots=True)
class SlowRequest:
    """One of the slowest requests this process has answered, kept for the shape of its cost.

    A sample of the tail, not a full trace: the N slowest since the last reset, which is the
    sample an operator actually wants, because the median request is already in the histogram
    and the p99 is the one nobody can reproduce.  It carries the same closed set of fields as
    the log line, for the same reason: no arguments, no results, no content.  ``request_digest``
    is :func:`request_digest` of the client's request id and not the id, for the reason given
    there.
    """

    verb: str
    tenant: int
    outcome: str
    duration_s: float
    queue_wait_s: float | None = None
    arms: dict[str, float] = field(default_factory=dict)
    request_digest: str = ""
    principal: str = ""
    at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """The JSON-safe form, with every duration in milliseconds."""
        return {
            "verb": self.verb,
            "tenant": int(self.tenant),
            "outcome": self.outcome,
            "duration_ms": round(self.duration_s * 1000.0, 3),
            "queue_wait_ms": (
                None if self.queue_wait_s is None else round(self.queue_wait_s * 1000.0, 3)
            ),
            "arms_ms": {k: round(v * 1000.0, 3) for k, v in sorted(self.arms.items())},
            "request_digest": self.request_digest,
            "principal": self.principal,
            "at": _iso(self.at),
        }


class _Slot:
    """Per-request scratch space, carried from the dispatcher to the parts that measure.

    A :class:`~anatid.server.protocol.Request` is a frozen slots dataclass and a write's future
    is created several frames below the dispatcher, so there is nowhere on either of them to
    hang a timing.  This is that place: created by the request probe, found by the read probe
    through a per-request key, and by the queue probe through the closure it builds.
    """

    __slots__ = ("arms", "queue_wait")

    def __init__(self) -> None:
        self.arms: dict[str, float] | None = None
        self.queue_wait: float | None = None


# --------------------------------------------------------------------------- the collector


class Metrics:
    """Everything one server process has measured, and the surfaces that render it.

    Construct it and hand it to :func:`attach`, or let :func:`attach` construct one::

        metrics = anatid.server.metrics.attach(server)
        ...
        print(metrics.render())  # Prometheus text
        metrics.snapshot()  # the same numbers as JSON
        metrics.slowest()  # the tail, with its arm breakdown

    ``slow_seconds``
        A request over this is logged at ``slow_level`` and counted in
        ``anatid_slow_requests_total``.  The default of 0.25 s is an order of magnitude above
        the slowest thing measured on a normal tenant (a ``recall`` p99 of 33 ms over the
        socket), so it fires on a fault and not on traffic.
    ``slow_trace``
        How many of the slowest requests to keep.  0 disables the trace.
    ``max_tenant_labels``
        The ceiling on distinct ``tenant`` label values.  A tenant id comes off the wire, so
        without a ceiling a client could mint time series until the process ran out of memory.
        Past it, per-tenant series are recorded under ``tenant="other"``.
    ``log_requests`` / ``log_level``
        The line per request, and where it goes.  INFO by default, which is what a request log
        normally is.  Formatting one line costs 5.1 us, which is the largest single piece of
        the instrumentation and still under 1% of the cheapest verb; ``log_requests=False``
        keeps the metrics and drops the line.
    ``log_format``
        ``"logfmt"`` (the default) or ``"json"``.  Both carry the same closed set of fields.
        ``json`` costs 6.1 us a line against ``logfmt``'s 5.1 us.
    """

    def __init__(
        self,
        *,
        slow_seconds: float = 0.25,
        slow_trace: int = 20,
        max_tenant_labels: int = 64,
        log_requests: bool = True,
        log_level: int = logging.INFO,
        slow_level: int = logging.WARNING,
        log_format: str = "logfmt",
        logger: logging.Logger | None = None,
        buckets: Sequence[float] = LATENCY_BUCKETS,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if log_format not in ("logfmt", "json"):
            raise ValueError(f"log_format is 'logfmt' or 'json', got {log_format!r}")
        if max_tenant_labels < 1:
            raise ValueError("max_tenant_labels must be at least 1")
        self.slow_seconds = float(slow_seconds)
        self.slow_trace = int(slow_trace)
        self.max_tenant_labels = int(max_tenant_labels)
        self.log_requests = bool(log_requests)
        self.log_level = int(log_level)
        self.slow_level = int(slow_level)
        self.log_format = log_format
        self.log = logger or log
        self._clock = clock

        #: One lock for every counter in this object.  An observation is a bisect and three
        #: increments, so holding one lock across it costs less than a lock per family would in
        #: bookkeeping; the measured overhead in the module docstring is the evidence.
        self._lock = threading.Lock()
        self._families: dict[str, _Family] = {}
        self._tenant_labels: dict[int, str] = {}
        self._slow: list[SlowRequest] = []
        self._in_flight = 0
        self._started_at = time.time()
        self._verbs: Mapping[str, Any] | None = None

        self._histogram(
            "anatid_request_duration_seconds",
            "Server-side handling of one request, end to end, in seconds.",
            buckets,
            ("verb", "outcome"),
        )
        self._histogram(
            "anatid_recall_arm_duration_seconds",
            "One retrieval arm of one recall, in seconds.",
            buckets,
            ("arm",),
        )
        self._histogram(
            "anatid_queue_wait_seconds",
            "How long a write waited in its tenant's queue before its transaction opened.",
            WAIT_BUCKETS,
            ("tenant",),
        )
        self._histogram(
            "anatid_write_batch_duration_seconds",
            "One worker turn on one tenant, in seconds.",
            buckets,
        )
        self._histogram(
            "anatid_write_batch_size",
            "Writes taken from one tenant's queue in one worker turn.",
            BATCH_BUCKETS,
        )
        self._scalar(
            "anatid_backpressure_rejections_total",
            "counter",
            "Writes refused because the tenant's queue was full. The write did not happen.",
            ("tenant",),
        )
        self._scalar(
            "anatid_conflicts_total",
            "counter",
            "ConflictError answers. retryable=true is an MVCC abort, false a stale "
            "compare-and-swap.",
            ("verb", "retryable"),
        )
        self._scalar(
            "anatid_idempotency_lookups_total",
            "counter",
            "Writes that arrived carrying an idempotency key.",
            ("verb",),
        )
        self._scalar(
            "anatid_idempotency_hits_total",
            "counter",
            "Writes replayed from a recorded key instead of written a second time.",
            ("verb",),
        )
        self._scalar(
            "anatid_slow_requests_total",
            "counter",
            "Requests over the slow threshold.",
            ("verb",),
        )
        self._scalar(
            "anatid_encode_failures_total",
            "counter",
            "Replies the server computed but could not encode.",
        )
        self._scalar(
            "anatid_queue_depth",
            "gauge",
            "Writes queued for one tenant right now.",
            ("tenant",),
        )

        self._server: Any = None
        self._restore: list[tuple[Any, str, Any]] = []
        self._local = threading.local()
        self._slots: dict[int, _Slot] = {}
        self._arms_installed = False
        self._verb_registered = False

    # -- family construction -----------------------------------------------------------

    def _histogram(
        self, name: str, description: str, bounds: Sequence[float], labels: Sequence[str] = ()
    ) -> None:
        self._families[name] = _Histogram(name, description, bounds, labels)

    def _scalar(self, name: str, kind: str, description: str, labels: Sequence[str] = ()) -> None:
        self._families[name] = _Scalar(name, kind, description, labels)

    # -- label discipline --------------------------------------------------------------

    def verb_label(self, verb: Any) -> str:
        """``verb`` if the server dispatches it, :data:`UNKNOWN_VERB` otherwise.

        The whole of the cardinality defence for this label: an unknown verb is a string the
        client chose, and a metric label is memory the server holds for the life of the process.
        """
        verbs = self._verbs
        if verbs is None:
            from .server import VERBS

            verbs = self._verbs = VERBS
        text = str(verb)
        return text if text in verbs else UNKNOWN_VERB

    def tenant_label(self, tenant_id: Any) -> str:
        """``tenant_id`` as a label value, or :data:`OVERFLOW_TENANT` past the ceiling.

        The caller holds :attr:`_lock`.
        """
        try:
            key = int(tenant_id)
        except (TypeError, ValueError):
            return OVERFLOW_TENANT
        known = self._tenant_labels.get(key)
        if known is not None:
            return known
        if len(self._tenant_labels) >= self.max_tenant_labels:
            return OVERFLOW_TENANT
        label = str(key)
        self._tenant_labels[key] = label
        return label

    # -- observation -------------------------------------------------------------------
    #
    # The public API.  A later release that calls these from server.py directly can drop every
    # wrapper below and keep every number.

    def observe(self, name: str, value: float, labels: Sequence[str] = ()) -> None:
        """Record one value in histogram ``name`` under ``labels``."""
        with self._lock:
            self._observe_locked(name, value, labels)

    def _observe_locked(self, name: str, value: float, labels: Sequence[str]) -> None:
        family = self._families.get(name)
        if not isinstance(family, _Histogram):
            return
        key = tuple(labels)
        series = family.series.get(key)
        if series is None:
            series = family.series[key] = _Series(len(family.bounds))
        series.counts[bisect.bisect_left(family.bounds, value)] += 1
        series.total += value
        series.count += 1

    def increment(self, name: str, labels: Sequence[str] = (), amount: float = 1.0) -> None:
        """Add ``amount`` to counter ``name`` under ``labels``."""
        with self._lock:
            self._increment_locked(name, labels, amount)

    def _increment_locked(self, name: str, labels: Sequence[str], amount: float) -> None:
        family = self._families.get(name)
        if not isinstance(family, _Scalar):
            return
        key = tuple(labels)
        family.values[key] = family.values.get(key, 0.0) + amount

    def observe_request(
        self,
        *,
        verb: Any,
        tenant: Any,
        outcome: str,
        duration_s: float,
        queue_wait_s: float | None = None,
        arms: Mapping[str, float] | None = None,
        request_id: str = "",
        principal: str = "",
        error_class: str = "",
        retryable: bool | None = None,
    ) -> None:
        """Record one finished request: its histogram, its trace entry and its log line.

        The one entry point a request has to reach to be visible at all.  Every argument is a
        number or a value from a closed set; nothing here reads the request's arguments or its
        result.

        ``queue_wait_s`` is reported here and NOT put in ``anatid_queue_wait_seconds``.  That
        histogram belongs to :meth:`observe_queue_wait`, which the queue probe calls for every
        item in a worker turn -- including one whose deadline expired and one an idempotency key
        replayed, neither of which reaches a verb and both of which waited.  Observing it in
        both places would double-count the writes that did reach a verb and undercount the rest,
        which is worse than either.
        """
        verb_label = self.verb_label(verb)
        slow = duration_s >= self.slow_seconds
        with self._lock:
            self._observe_locked(
                "anatid_request_duration_seconds", duration_s, (verb_label, outcome)
            )
            if outcome == "conflict":
                label = "unknown" if retryable is None else ("true" if retryable else "false")
                self._increment_locked("anatid_conflicts_total", (verb_label, label), 1.0)
            if slow:
                self._increment_locked("anatid_slow_requests_total", (verb_label,), 1.0)
                if self.slow_trace > 0:
                    self._remember_slow_locked(
                        SlowRequest(
                            verb=verb_label,
                            tenant=_as_int(tenant),
                            outcome=outcome,
                            duration_s=duration_s,
                            queue_wait_s=queue_wait_s,
                            arms=dict(arms or {}),
                            request_digest=request_digest(request_id),
                            principal=_safe(principal, 48),
                            at=time.time(),
                        )
                    )
        self._log_request(
            verb=verb_label,
            tenant=tenant,
            outcome=outcome,
            duration_s=duration_s,
            queue_wait_s=queue_wait_s,
            arms=arms,
            request_id=request_id,
            principal=principal,
            error_class=error_class,
            slow=slow,
        )

    def observe_arm(self, arm: str, duration_s: float) -> None:
        """Record one retrieval arm's cost."""
        self.observe("anatid_recall_arm_duration_seconds", duration_s, (arm,))

    def observe_queue_wait(self, tenant: Any, waited_s: float) -> None:
        """Record how long one write sat in its tenant's queue."""
        with self._lock:
            self._observe_locked(
                "anatid_queue_wait_seconds", waited_s, (self.tenant_label(tenant),)
            )

    def observe_batch(self, size: int, duration_s: float) -> None:
        """Record one worker turn: how many writes it took, and how long it held the file."""
        with self._lock:
            self._observe_locked("anatid_write_batch_size", float(size), ())
            self._observe_locked("anatid_write_batch_duration_seconds", duration_s, ())

    def note_backpressure(self, tenant: Any) -> None:
        """Record one write refused because its tenant's queue was full."""
        with self._lock:
            self._increment_locked(
                "anatid_backpressure_rejections_total", (self.tenant_label(tenant),), 1.0
            )

    def note_idempotency(self, verb: Any, *, hit: bool) -> None:
        """Record one idempotency lookup and whether it replayed a recorded result."""
        verb_label = self.verb_label(verb)
        with self._lock:
            self._increment_locked("anatid_idempotency_lookups_total", (verb_label,), 1.0)
            if hit:
                self._increment_locked("anatid_idempotency_hits_total", (verb_label,), 1.0)

    def note_encode_failure(self) -> None:
        """Record a reply the server computed and could not put on the wire."""
        self.increment("anatid_encode_failures_total")

    def _remember_slow_locked(self, entry: SlowRequest) -> None:
        """Keep the ``slow_trace`` slowest.  The caller holds :attr:`_lock`.

        A sorted list rather than a heap: it holds twenty entries by default and is touched only
        by requests already over the slow threshold, so a heap's constant factor would buy
        nothing and the list reads out in order with no copy.
        """
        self._slow.append(entry)
        self._slow.sort(key=lambda e: e.duration_s, reverse=True)
        del self._slow[self.slow_trace :]

    # -- the log line ------------------------------------------------------------------

    def _log_request(
        self,
        *,
        verb: str,
        tenant: Any,
        outcome: str,
        duration_s: float,
        queue_wait_s: float | None,
        arms: Mapping[str, float] | None,
        request_id: str,
        principal: str,
        error_class: str,
        slow: bool,
    ) -> None:
        if not self.log_requests:
            return
        level = self.slow_level if slow else self.log_level
        if not self.log.isEnabledFor(level):
            return
        fields: dict[str, Any] = {
            "event": "request",
            "verb": verb,
            "tenant": _as_int(tenant),
            "outcome": outcome,
            "duration_ms": _ms(duration_s),
        }
        if queue_wait_s is not None:
            fields["queue_ms"] = _ms(queue_wait_s)
        # Iterating ARMS rather than the mapping keeps the field names a closed set even if a
        # future probe records an arm under a name this build does not know.
        for arm in ARMS:
            value = None if arms is None else arms.get(arm)
            if value is not None:
                fields[f"{arm}_ms"] = _ms(value)
        if error_class:
            fields["error"] = _safe(error_class, 48)
        if slow:
            fields["slow"] = "true"
        fields["id"] = request_digest(request_id)
        fields["principal"] = _safe(principal, 48)
        if self.log_format == "json":
            self.log.log(level, "%s", json.dumps(fields, separators=(",", ":")))
        else:
            self.log.log(level, "%s", " ".join(f"{k}={v}" for k, v in fields.items()))

    # -- rendering ---------------------------------------------------------------------

    def snapshot(self, *, tenants: Iterable[int] | None = None) -> dict[str, Any]:
        """Every metric as a JSON-safe mapping, shaped as families.

        Shaped as families rather than a flat dictionary so the two surfaces cannot drift:
        :meth:`render` walks exactly this, so a family added here appears in both with no second
        edit.

        ``tenants`` limits per-tenant series to those tenants, and drops the
        :data:`OVERFLOW_TENANT` bucket with them, because a principal that may not name a tenant
        may not learn its queue depth either.  None means no limit.

        Deliberately runs no SQL.  Reading a schema version per tenant would make every scrape a
        query against every open file, which turns monitoring into load; readiness is the
        endpoint that pays for that answer, and it is a different question.
        """
        allowed = None if tenants is None else {str(int(t)) for t in tenants}
        families: list[dict[str, Any]] = []
        with self._lock:
            self._refresh_live_locked()
            for name in sorted(self._families):
                rendered = self._family_dict(self._families[name], allowed)
                if rendered is not None:
                    families.append(rendered)
            slowest = [entry.as_dict() for entry in self._slow]
        return {
            "generated_at": _iso(time.time()),
            "since": _iso(self._started_at),
            "families": families,
            "slowest": slowest,
        }

    def _family_dict(self, family: _Family, allowed: set[str] | None) -> dict[str, Any] | None:
        """One family's series, or None when the reader may see none of them."""
        series: list[dict[str, Any]] = []
        if isinstance(family, _Histogram):
            for key in sorted(family.series):
                if not self._visible(family, key, allowed):
                    continue
                measured = family.series[key]
                cumulative = 0
                buckets: dict[str, int] = {}
                for bound, count in zip(family.bounds, measured.counts[:-1], strict=True):
                    cumulative += count
                    buckets[_number(bound)] = cumulative
                buckets["+Inf"] = cumulative + measured.counts[-1]
                series.append(
                    {
                        "labels": dict(zip(family.labels, key, strict=True)),
                        "buckets": buckets,
                        "sum": measured.total,
                        "count": measured.count,
                    }
                )
        elif isinstance(family, _Scalar):
            for key in sorted(family.values):
                if not self._visible(family, key, allowed):
                    continue
                series.append(
                    {
                        "labels": dict(zip(family.labels, key, strict=True)),
                        "value": family.values[key],
                    }
                )
        if not series:
            return None
        return {
            "name": family.name,
            "type": family.kind,
            "help": family.description,
            "series": series,
        }

    def _visible(self, family: _Family, key: tuple[str, ...], allowed: set[str] | None) -> bool:
        if allowed is None or family.tenant_index is None:
            return True
        # OVERFLOW_TENANT holds several tenants at once, so it is never shown to a restricted
        # principal: "other" would be a number about tenants that principal may not name.
        return key[family.tenant_index] in allowed

    def render(self, *, tenants: Iterable[int] | None = None) -> str:
        """Every metric in Prometheus text exposition format, from :meth:`snapshot`."""
        lines: list[str] = []
        for family in self.snapshot(tenants=tenants)["families"]:
            name = family["name"]
            lines.append(f"# HELP {name} {family['help']}")
            lines.append(f"# TYPE {name} {family['type']}")
            for series in family["series"]:
                labels = series["labels"]
                if "buckets" in series:
                    for bound, count in series["buckets"].items():
                        rendered = _render_labels(labels, f'le="{bound}"')
                        lines.append(f"{name}_bucket{rendered} {count}")
                    plain = _render_labels(labels)
                    lines.append(f"{name}_sum{plain} {_number(series['sum'])}")
                    lines.append(f"{name}_count{plain} {series['count']}")
                else:
                    lines.append(f"{name}{_render_labels(labels)} {_number(series['value'])}")
        lines.append("")
        return "\n".join(lines)

    def slowest(self) -> list[SlowRequest]:
        """The slowest requests seen since the last :meth:`reset`, slowest first."""
        with self._lock:
            return list(self._slow)

    def reset(self) -> None:
        """Forget every observation.  For tests, and for a process that has been reconfigured."""
        with self._lock:
            for family in self._families.values():
                if isinstance(family, _Histogram):
                    family.series.clear()
                elif isinstance(family, _Scalar):
                    family.values.clear()
            self._slow.clear()
            self._tenant_labels.clear()
            self._started_at = time.time()

    # -- live values -------------------------------------------------------------------

    def _refresh_live_locked(self) -> None:
        """Fill the gauges and pass-through counters that are read rather than accumulated.

        Queue depth is a gauge, and reading it from the queue at render time is the only way to
        report it truthfully; accumulating it would report a history of depths nobody asked for.
        The queue's own counters are copied verbatim rather than re-counted, so the two can
        never disagree.
        """
        self._set_locked(
            "anatid_build_info",
            "gauge",
            "Build identity of this server process.",
            1.0,
            ("version", "protocol"),
            _build_labels(),
        )
        self._set_locked(
            "anatid_requests_in_flight",
            "gauge",
            "Requests the dispatcher is handling right now.",
            float(self._in_flight),
        )
        server = self._server
        if server is None:
            return
        try:
            health = server.health()
            queue = server.queue
            stats = queue.stats()
            open_handles = server._open_handles()  # noqa: SLF001
        except Exception:  # a server mid-shutdown is not a reason to fail a scrape
            log.debug("could not read live server state for metrics", exc_info=True)
            return
        self._set_locked(
            "anatid_up",
            "gauge",
            "1 while the process is serving or draining.",
            1.0 if health.ok else 0.0,
        )
        self._set_locked(
            "anatid_uptime_seconds",
            "gauge",
            "Seconds since the server started.",
            float(health.uptime_s),
        )
        self._set_locked(
            "anatid_accepting_writes",
            "gauge",
            "1 while the write queue is accepting, 0 while it is draining.",
            1.0 if queue.accepting else 0.0,
        )
        self._set_locked(
            "anatid_open_files",
            "gauge",
            "Tenant database files this process holds open.",
            float(len(open_handles)),
        )
        self._set_locked(
            "anatid_queue_in_flight",
            "gauge",
            "Tenants a worker is writing right now.",
            float(stats.in_flight),
        )
        self._set_locked(
            "anatid_queue_max_depth",
            "gauge",
            "Configured per-tenant queue limit.",
            float(stats.max_depth),
        )
        self._set_locked(
            "anatid_queue_batch_max",
            "gauge",
            "Configured writes per worker turn.",
            float(stats.batch_max),
        )
        self._set_locked(
            "anatid_queue_workers",
            "gauge",
            "Worker threads draining the queues.",
            float(stats.workers),
        )
        self._set_locked(
            "anatid_queue_high_water",
            "gauge",
            "Per-tenant depth at which readiness turns false.",
            float(max(1, int(stats.max_depth * server.config.high_water))),
        )
        depth = self._families["anatid_queue_depth"]
        if isinstance(depth, _Scalar):
            depth.values.clear()
            for tenant_id, queued in stats.depth.items():
                key = (self.tenant_label(tenant_id),)
                depth.values[key] = depth.values.get(key, 0.0) + float(queued)
        for name, value, description in (
            ("anatid_writes_submitted_total", stats.submitted, "Writes accepted into a queue."),
            ("anatid_writes_completed_total", stats.completed, "Writes that committed."),
            ("anatid_writes_failed_total", stats.failed, "Writes that raised."),
            (
                "anatid_writes_rejected_total",
                stats.rejected,
                (
                    "Writes refused for a full queue, summed over every tenant. The per-tenant "
                    "series is anatid_backpressure_rejections_total."
                ),
            ),
            (
                "anatid_writes_expired_total",
                stats.expired,
                "Writes whose deadline passed while queued. Nothing was written.",
            ),
            (
                "anatid_writes_replayed_total",
                stats.replayed,
                "Writes answered from a recorded idempotency key.",
            ),
            ("anatid_write_batches_total", stats.batches, "Transactions the workers opened."),
            (
                "anatid_write_batch_items_total",
                stats.batched_items,
                (
                    "Writes carried by those transactions. Divided by batches, this is writes "
                    "per transaction."
                ),
            ),
            (
                "anatid_write_batch_splits_total",
                stats.split_batches,
                "Batches that failed and were re-run one write at a time.",
            ),
        ):
            self._set_locked(name, "counter", description, float(value))

    def _set_locked(
        self,
        name: str,
        kind: str,
        description: str,
        value: float,
        labels: Sequence[str] = (),
        key: Sequence[str] = (),
    ) -> None:
        family = self._families.get(name)
        if not isinstance(family, _Scalar):
            family = _Scalar(name, kind, description, labels)
            self._families[name] = family
        family.values[tuple(key)] = float(value)

    # -- attaching ---------------------------------------------------------------------

    def attach(
        self, server: Any, *, arms: bool = True, http: bool = True, verb: bool = True
    ) -> "Metrics":
        """Install the probes on ``server``.  See :func:`attach`."""
        if self._server is not None:
            raise RuntimeError("these metrics are already attached to a server")
        self._server = server
        self._install_dispatch(server)
        self._install_queue(server)
        if verb:
            _acquire_metrics_verb()
            self._verb_registered = True
        if http:
            self._install_http(server)
        if arms:
            _acquire_arm_probes(self)
            self._arms_installed = True
        return self

    def detach(self) -> None:
        """Put every wrapped method back and stop measuring.  Safe to call twice."""
        for target, name, original in reversed(self._restore):
            try:
                setattr(target, name, original)
            except Exception:  # an object torn down before us
                log.debug("could not restore %s", name, exc_info=True)
        self._restore.clear()
        if self._arms_installed:
            _release_arm_probes(self)
            self._arms_installed = False
        if self._verb_registered:
            _release_metrics_verb()
            self._verb_registered = False
        self._server = None
        self._slots.clear()

    def _capture(self, target: Any, name: str) -> Any:
        """Remember ``target.name`` so :meth:`detach` can put it back, and return it."""
        original = getattr(target, name)
        self._restore.append((target, name, original))
        return original

    # -- the dispatch probes -----------------------------------------------------------

    def _install_dispatch(self, server: Any) -> None:
        """Time every request, and give the read path somewhere to put its arm timings.

        Both entry points are wrapped because neither calls the other: ``call`` is the
        synchronous dispatcher and ``acall`` is what the transports use.  ``run_read`` is
        wrapped as well, but only to open and close the arm accumulator and to answer the
        ``metrics`` verb; it never records a request, so nothing is counted twice.
        """
        original_call = self._capture(server, "call")
        original_acall = self._capture(server, "acall")
        original_read = self._capture(server, "run_read")
        original_submit = self._capture(server, "submit_write")
        original_render = self._capture(server, "_render")
        original_plan = server._plan  # noqa: SLF001

        def call(request: Any, principal: Any) -> Any:
            slot = self._open_slot(request)
            started = self._clock()
            try:
                response = original_call(request, principal)
            finally:
                self._close_slot(request)
            self._finish(request, principal, response, self._clock() - started, slot)
            return response

        async def acall(request: Any, principal: Any) -> Any:
            slot = self._open_slot(request)
            started = self._clock()
            try:
                response = await original_acall(request, principal)
            finally:
                self._close_slot(request)
            self._finish(request, principal, response, self._clock() - started, slot)
            return response

        def run_read(request: Any, principal: Any) -> Any:
            if getattr(request, "verb", "") == "metrics":
                # _plan first, so the metrics verb gets the same envelope discipline as every
                # other one: the principal must be allowed to name the tenant it sent, and a
                # 'tenant' smuggled into args is still refused.
                original_plan(request, principal)
                return self._metrics_verb(principal)
            slot = self._slots.get(id(request))
            if slot is None:
                return original_read(request, principal)
            previous = getattr(_ARM_LOCAL, "current", None)
            arms: dict[str, float] = {}
            _ARM_LOCAL.current = arms
            try:
                return original_read(request, principal)
            finally:
                _ARM_LOCAL.current = previous
                if arms:
                    slot.arms = arms

        def submit_write(request: Any, principal: Any) -> Any:
            # Read by the queue probe, which runs on this thread one frame down.  A
            # thread-local rather than an argument because the queue's signature is not this
            # module's to change.
            self._local.slot = self._slots.get(id(request))
            try:
                return original_submit(request, principal)
            finally:
                self._local.slot = None

        def render(response: Any) -> Any:
            rendered, body = original_render(response)
            if rendered is not response:
                self.note_encode_failure()
            return rendered, body

        server.call = call
        server.acall = acall
        server.run_read = run_read
        server.submit_write = submit_write
        server._render = render  # noqa: SLF001

    def _open_slot(self, request: Any) -> _Slot:
        slot = _Slot()
        self._slots[id(request)] = slot
        with self._lock:
            self._in_flight += 1
        return slot

    def _close_slot(self, request: Any) -> None:
        self._slots.pop(id(request), None)
        with self._lock:
            self._in_flight -= 1

    def _finish(
        self, request: Any, principal: Any, response: Any, duration_s: float, slot: _Slot
    ) -> None:
        error = getattr(response, "error", None)
        self.observe_request(
            verb=getattr(request, "verb", ""),
            tenant=getattr(request, "tenant", -1),
            outcome=outcome_for(response),
            duration_s=duration_s,
            queue_wait_s=slot.queue_wait,
            arms=slot.arms,
            request_id=getattr(request, "request_id", ""),
            principal=getattr(principal, "name", ""),
            error_class=getattr(error, "error_class", "") or "",
            retryable=None if error is None else getattr(error, "retryable", None),
        )

    # -- the queue probes --------------------------------------------------------------

    def _install_queue(self, server: Any) -> None:
        """Measure the write queue: backpressure at submit, wait and batch size at execution.

        ``submit`` is where a refusal happens and where the request that owns a write is still
        identifiable, so the slot travels into the queue on the ``work`` callable itself, which
        is the only object on that path.  ``_run_batch`` is where the wait ends, and it sees
        every item in the turn -- including one whose deadline has passed and one an idempotency
        key will replay, both of which waited and neither of which reaches the verb.
        """
        queue = server.queue
        clock = getattr(queue, "_clock", time.monotonic)
        original_submit = self._capture(queue, "submit")
        original_batch = self._capture(queue, "_run_batch")

        def submit(tenant_id: Any, verb: Any, work: Any, **kw: Any) -> Any:
            slot = getattr(self._local, "slot", None)
            carrier = work if slot is None else _Carrier(work, slot)
            try:
                return original_submit(tenant_id, verb, carrier, **kw)
            except BusyError:
                self.note_backpressure(tenant_id)
                raise

        def run_batch(tenant_id: Any, batch: Any) -> Any:
            now = clock()
            items = list(batch)
            with self._lock:
                label = (self.tenant_label(tenant_id),)
                for item in items:
                    waited = max(0.0, now - item.enqueued_at)
                    self._observe_locked("anatid_queue_wait_seconds", waited, label)
                    if isinstance(item.work, _Carrier):
                        item.work.slot.queue_wait = waited
                self._observe_locked("anatid_write_batch_size", float(len(items)), ())
            started = self._clock()
            try:
                return original_batch(tenant_id, batch)
            finally:
                self.observe("anatid_write_batch_duration_seconds", self._clock() - started, ())

        queue.submit = submit
        queue._run_batch = run_batch  # noqa: SLF001

        store = getattr(server, "idempotency", None)
        if store is None:
            return
        original_lookup = self._capture(store, "lookup")

        def lookup(db: Any, tenant_id: Any, key: Any, *, verb: Any, digest: Any) -> Any:
            record = original_lookup(db, tenant_id, key, verb=verb, digest=digest)
            self.note_idempotency(verb, hit=record is not None)
            return record

        store.lookup = lookup

    # -- the metrics verb --------------------------------------------------------------

    def _metrics_verb(self, principal: Any) -> dict[str, Any]:
        """``{"verb": "metrics"}`` over the socket protocol.

        Scoped to the calling principal.  A per-tenant queue depth is not that tenant's data,
        but it does say the tenant exists and how busy it is, and the server's rule is that a
        principal cannot learn whether a tenant it may not name exists.  An unrestricted
        principal sees everything; a scoped one sees the process-wide numbers and only its own
        tenants' series.
        """
        tenants = getattr(principal, "tenants", None)
        return self.snapshot(tenants=None if tenants is None else sorted(tenants))

    # -- the HTTP endpoint -------------------------------------------------------------

    def _install_http(self, server: Any) -> None:
        """Add ``GET /metrics`` to the server's HTTP surface.

        Authenticated exactly when ``/rpc`` is: a listener whose authenticator requires a token
        requires one here too, because per-tenant queue depths are the same kind of thing that
        token protects.  A listener with no token (a Unix-socket-only or loopback deployment)
        serves it open, as ``/health`` and ``/ready`` already are.
        """
        original_route = self._capture(server, "_http_route")
        _patch_http_content_type()

        async def http_route(
            method: str, target: str, headers: Mapping[str, str], body: bytes
        ) -> Any:
            path = target.split("?", 1)[0]
            if method != "GET" or path != METRICS_PATH:
                return await original_route(method, target, headers, body)
            from . import protocol
            from .auth import ConnectionContext

            authenticator = server.authenticator
            tenants: Iterable[int] | None = None
            if getattr(authenticator, "requires_token", False):
                try:
                    principal = authenticator.authenticate(
                        ConnectionContext(transport="http", socket=None, headers=headers)
                    )
                except Exception as exc:
                    return 401, protocol.dumps(protocol.Response.failure(exc).to_wire())
                allowed = getattr(principal, "tenants", None)
                tenants = None if allowed is None else sorted(allowed)
            payload = self.render(tenants=tenants).encode("utf-8")
            _mark_content_type(payload, CONTENT_TYPE)
            return 200, payload

        server._http_route = http_route  # noqa: SLF001


class _Carrier:
    """One write's ``work`` callable with its request's slot attached.

    A :class:`~anatid.server.queue.QueuedWrite` is built inside ``submit`` and never handed
    back, so the callable is the only object that travels from the dispatcher to the worker
    thread.  This is how the wait measured on the worker thread finds the request that has been
    waiting for it.
    """

    __slots__ = ("work", "slot")

    def __init__(self, work: Any, slot: _Slot) -> None:
        self.work = work
        self.slot = slot

    def __call__(self, db: Any) -> Any:
        return self.work(db)


def _build_labels() -> tuple[str, str]:
    from .. import __version__
    from .protocol import PROTOCOL_VERSION

    return (str(__version__), str(PROTOCOL_VERSION))


# --------------------------------------------------------------------------- process globals
#
# Three things here outlive a single Metrics instance, because the thing they touch does: the
# retrieval functions, the server's verb table, and the HTTP writer.  Each is refcounted or
# idempotent so two servers in one interpreter do not take an installation away from each other.

#: Where the arm probes find the accumulator for the request running on this thread.
_ARM_LOCAL = threading.local()

_ARM_LOCK = threading.Lock()
_ARM_ORIGINALS: dict[tuple[str, str], Any] = {}
#: Every attached :class:`Metrics` that wants arm histograms.  A list, not one reference,
#: because two servers can share one interpreter.
_ARM_CONSUMERS: list[Metrics] = []

_VERB_LOCK = threading.Lock()
_VERB_USERS = 0

_HTTP_LOCK = threading.Lock()
_HTTP_PATCHED = False
#: Bodies waiting to be written with a content type other than JSON, keyed by ``id``.  The body
#: itself is kept in the value so the id cannot be reused while the entry is pending.
_PENDING_TYPES: dict[int, tuple[bytes, str]] = {}


def _armed(label: str, function: Any) -> Any:
    """``function`` with its duration charged to the arm accumulator on the calling thread.

    Records nothing at all when no instrumented request is running here, which is what keeps an
    in-process :class:`anatid.Anatid` in the same interpreter unaffected: the probe is then one
    attribute read and one call.
    """

    def probe(*args: Any, **kwargs: Any) -> Any:
        arms = getattr(_ARM_LOCAL, "current", None)
        if arms is None:
            return function(*args, **kwargs)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            arms[label] = arms.get(label, 0.0) + elapsed
            for metrics in _ARM_CONSUMERS:
                metrics.observe_arm(label, elapsed)

    probe.__name__ = getattr(function, "__name__", label)
    probe.__doc__ = getattr(function, "__doc__", None)
    return probe


def _acquire_arm_probes(metrics: Metrics) -> None:
    with _ARM_LOCK:
        _ARM_CONSUMERS.append(metrics)
        if len(_ARM_CONSUMERS) > 1:
            return
        for module_name, attribute, label in _ARM_TARGETS:
            module = importlib.import_module(module_name)
            original = getattr(module, attribute, None)
            if original is None:  # pragma: no cover - only if a function is renamed
                log.debug("no %s.%s to instrument", module_name, attribute)
                continue
            _ARM_ORIGINALS[(module_name, attribute)] = original
            setattr(module, attribute, _armed(label, original))


def _release_arm_probes(metrics: Metrics) -> None:
    with _ARM_LOCK:
        if metrics in _ARM_CONSUMERS:
            _ARM_CONSUMERS.remove(metrics)
        if _ARM_CONSUMERS:
            return
        for (module_name, attribute), original in _ARM_ORIGINALS.items():
            setattr(importlib.import_module(module_name), attribute, original)
        _ARM_ORIGINALS.clear()


def _acquire_metrics_verb() -> None:
    """Add ``metrics`` to the server's verb table.

    :data:`~anatid.server.server.VERBS` is a mutable mapping and adding an entry is the
    documented way to expose a call; nothing reaches a verb the table does not name.  The entry
    is refcounted so two servers can each attach and detach without taking the verb from the
    other, and it is removed on the last detach so a server with no metrics does not advertise a
    verb it has no handler for.
    """
    global _VERB_USERS
    from .server import VERBS, VerbSpec

    with _VERB_LOCK:
        _VERB_USERS += 1
        if _VERB_USERS > 1:
            return
        VERBS["metrics"] = VerbSpec(
            name="metrics",
            write=False,
            batchable=False,
            tenant_arg=False,
            server=True,
            summary="counters, histograms and the slow-request trace, as JSON",
        )


def _release_metrics_verb() -> None:
    global _VERB_USERS
    from .server import VERBS

    with _VERB_LOCK:
        _VERB_USERS = max(0, _VERB_USERS - 1)
        if _VERB_USERS:
            return
        VERBS.pop("metrics", None)


def _mark_content_type(body: bytes, content_type: str) -> None:
    """Ask the patched writer to send ``body`` with ``content_type`` instead of JSON."""
    with _HTTP_LOCK:
        if len(_PENDING_TYPES) > 16:
            # A reply that was never written, because its connection went away. The content
            # type is a nicety and the body is not; dropping the whole table is the cheap fix.
            _PENDING_TYPES.clear()
        _PENDING_TYPES[id(body)] = (body, content_type)


def _take_content_type(body: bytes) -> str | None:
    with _HTTP_LOCK:
        entry = _PENDING_TYPES.pop(id(body), None)
    return entry[1] if entry is not None and entry[0] is body else None


def _patch_http_content_type() -> None:
    """Let one reply out of the HTTP surface carry a content type other than JSON.

    ``anatid.server.server._write_http`` hard-codes ``application/json``, which is right for
    every route it has and wrong for a Prometheus exposition.  Rather than duplicate the writer,
    this wraps it: a body the metrics route registered is written here, and everything else
    goes straight through.

    Matched on the identity of the body object rather than on a flag, because ``_http_handler``
    awaits between building a reply and writing it, and another connection's reply can be
    written in between.  Installed once per process and never removed: taking it out while a
    connection was mid-reply would be worse than leaving a wrapper that does nothing.
    """
    global _HTTP_PATCHED
    with _HTTP_LOCK:
        if _HTTP_PATCHED:
            return
        _HTTP_PATCHED = True
    from . import server as server_module

    original = getattr(server_module, "_write_http", None)
    if original is None:  # pragma: no cover - only if server.py stops having one
        log.debug("no _write_http to wrap; /metrics will be served as application/json")
        return

    async def write_http(writer: Any, status: int, body: bytes, keep_alive: bool) -> None:
        content_type = _take_content_type(body)
        if content_type is None:
            await original(writer, status, body, keep_alive)
            return
        head = (
            f"HTTP/1.1 {status} OK\r\n"
            f"content-type: {content_type}\r\n"
            f"content-length: {len(body)}\r\n"
            f"connection: {'keep-alive' if keep_alive else 'close'}\r\n\r\n"
        ).encode("latin-1")
        writer.write(head + body)
        await writer.drain()

    server_module._write_http = write_http  # noqa: SLF001


# --------------------------------------------------------------------------- entry point


def attach(
    server: Any,
    *,
    metrics: Metrics | None = None,
    arms: bool = True,
    http: bool = True,
    verb: bool = True,
    **options: Any,
) -> Metrics:
    """Instrument ``server`` and return the :class:`Metrics` collecting from it.

    ::

        server = AnatidServer(pool=pool, config=config)
        metrics = anatid.server.metrics.attach(server)
        await server.start()

    Attach before or after :meth:`~anatid.server.server.AnatidServer.start`; the probes replace
    bound methods on that one instance, and the listeners call whatever is on the instance when
    a request arrives.  ``options`` go to the :class:`Metrics` constructor.

    ``arms=False`` skips the per-arm probes, which are the only ones that reach outside this
    server: they replace module-level functions in :mod:`anatid.recall` and :mod:`anatid.fts`
    for the whole process.  ``http=False`` skips ``GET /metrics``, ``verb=False`` skips the
    ``metrics`` verb.  :meth:`Metrics.detach` reverses all of it.
    """
    collector = metrics or Metrics(**options)
    return collector.attach(server, arms=arms, http=http, verb=verb)
