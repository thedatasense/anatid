"""Coordination-free 63-bit id allocation.

anatid keeps the spike's BIGINT ids (they are what makes the joins and zone maps cheap), so it
needs ids that several writers -- threads, processes, machines -- can mint without agreeing on
anything.  A shared counter row would be correct but would turn every ``remember()`` into a
write-write conflict on the same row under DuckDB's optimistic MVCC, which is exactly the
contention the Phase 0 benchmark showed we do not have.

Layout (63 bits, always positive, roughly time-ordered so inserts stay append-friendly)::

    bits 62..22   milliseconds since 2020-01-01 UTC   (41 bits, good to the year 2089)
    bits 21..10   per-process worker id, random        (12 bits)
    bits  9..0    per-millisecond sequence             (10 bits, 1024 ids/ms/process)

Collision requires two processes to draw the same 12-bit worker id AND mint an id in the same
millisecond AND land on the same sequence number.  That is a deliberate trade, not an oversight:
if you need a hard guarantee, pass explicit ids (every verb accepts one) or install your own
allocator with :func:`set_allocator`.

The clock is a *logical* one, and that is load-bearing
-----------------------------------------------------
``time.time()`` is not monotonic: NTP steps it, a VM resumes with a stale clock, an operator
sets it by hand.  anatid 0.1.0 handled a backwards step by pinning the timestamp and advancing
the 10-bit sequence with ``(self._seq + 1) & _SEQ_MASK`` -- which wraps to 0 after 1,024 ids and
then **hands out the id it handed out 1,024 ids ago**.  A simulated rollback reproduced it
exactly: id 1,025 equalled id 1.

So :class:`IdAllocator` keeps its own millisecond counter that only ever moves forward:

* the wall clock is *read*, never *trusted*: ``_last_ms`` takes the wall clock only when the
  wall clock is ahead of it;
* a backwards step therefore changes nothing at all -- the allocator keeps minting from the
  logical millisecond it had reached;
* when the 1,024 sequence slots of a logical millisecond run out, the logical clock **borrows
  the next millisecond from the future** instead of wrapping.  Ids stay unique and strictly
  increasing per process; the timestamp bits can run ahead of the wall clock for as long as a
  process sustains more than a million ids/second (:attr:`IdAllocator.drift_ms` reports by how
  much) and re-converge as soon as the wall clock catches up.

Two ids from one process are therefore never equal and always ordered, whatever the clock does.
Across processes the 12-bit worker id is still the only separation, exactly as before.
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Callable

__all__ = ["new_id", "set_allocator", "reset_allocator", "IdAllocator", "EPOCH_MS"]

EPOCH_MS = 1_577_836_800_000  # 2020-01-01T00:00:00Z in milliseconds

_SEQ_BITS = 10
_WORKER_BITS = 12
_TIME_BITS = 41
_SEQ_MASK = (1 << _SEQ_BITS) - 1
_WORKER_MASK = (1 << _WORKER_BITS) - 1
_MAX_MS = (1 << _TIME_BITS) - 1


def _wall_ms() -> int:
    """Milliseconds since :data:`EPOCH_MS`, floored at 0.

    Floored because a clock set before 2020 would otherwise shift a negative number into the
    sign bit and mint negative ids, which every BIGINT id column in the schema assumes cannot
    happen.
    """
    return max(0, int(time.time() * 1000) - EPOCH_MS)


class IdAllocator:
    """Thread-safe monotonic id source.  One instance per process is enough.

    Monotonic *by construction*, not by trusting the clock: see the module docstring.  Ids from
    one instance are strictly increasing and never repeat, including across an NTP step
    backwards, a suspend/resume, and sequence exhaustion.
    """

    def __init__(self, worker_id: int | None = None) -> None:
        if worker_id is None:
            # os.urandom-seeded: distinct across processes even when forked in the same tick.
            worker_id = random.SystemRandom().getrandbits(_WORKER_BITS)
        self.worker_id = int(worker_id) & _WORKER_MASK
        self._lock = threading.Lock()
        self._last_ms = -1
        self._seq = 0
        self._pid = os.getpid()

    def __call__(self) -> int:
        return self.next_id()

    @property
    def last_ms(self) -> int:
        """The logical millisecond the allocator has reached (-1 before the first id)."""
        return self._last_ms

    @property
    def drift_ms(self) -> int:
        """How far the logical clock has borrowed ahead of the wall clock, in ms (0 normally).

        Non-zero means either this process is minting more than 1,024 ids per millisecond, or
        the wall clock has stepped backwards and the allocator is refusing to follow it.
        """
        return max(0, self._last_ms - _wall_ms())

    def next_id(self) -> int:
        with self._lock:
            if os.getpid() != self._pid:  # forked: take a fresh worker id
                self._pid = os.getpid()
                self.worker_id = random.SystemRandom().getrandbits(_WORKER_BITS)
                self._last_ms, self._seq = -1, 0
            ms = _wall_ms()
            if ms > self._last_ms:
                # The wall clock is ahead: adopt it and start a fresh sequence.
                self._last_ms = ms
                self._seq = 0
            else:
                # Same millisecond, or the clock went BACKWARDS.  Either way the logical clock
                # holds its ground and the sequence advances.  No masking here: `& _SEQ_MASK`
                # is what made a rolled-back clock repeat an id after 1,024 of them.
                self._seq += 1
                if self._seq > _SEQ_MASK:
                    # This millisecond is spent.  Borrow the next one rather than wrap or
                    # sleep: sleeping cannot help when the wall clock is *behind* us.
                    self._last_ms += 1
                    self._seq = 0
            if self._last_ms > _MAX_MS:      # pragma: no cover - year 2089
                raise OverflowError(
                    f"id timestamp {self._last_ms} ms past {EPOCH_MS} exceeds the {_TIME_BITS} "
                    f"bits reserved for it; install a different allocator with set_allocator()")
            return (self._last_ms << (_WORKER_BITS + _SEQ_BITS)) \
                | (self.worker_id << _SEQ_BITS) | self._seq


_default = IdAllocator()
_allocator: Callable[[], int] = _default


def new_id() -> int:
    """Mint a fresh 63-bit id."""
    return _allocator()


def set_allocator(fn: Callable[[], int]) -> None:
    """Replace the process-wide id source (e.g. with a database sequence or a UUID-derived one)."""
    global _allocator
    _allocator = fn


def reset_allocator() -> None:
    """Restore the built-in time-ordered allocator."""
    global _allocator
    _allocator = _default
