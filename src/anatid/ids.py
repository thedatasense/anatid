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
_SEQ_MASK = (1 << _SEQ_BITS) - 1
_WORKER_MASK = (1 << _WORKER_BITS) - 1


class IdAllocator:
    """Thread-safe monotonic id source.  One instance per process is enough."""

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

    def next_id(self) -> int:
        with self._lock:
            if os.getpid() != self._pid:  # forked: take a fresh worker id
                self._pid = os.getpid()
                self.worker_id = random.SystemRandom().getrandbits(_WORKER_BITS)
                self._last_ms, self._seq = -1, 0
            ms = int(time.time() * 1000) - EPOCH_MS
            if ms == self._last_ms:
                self._seq = (self._seq + 1) & _SEQ_MASK
                if self._seq == 0:  # exhausted this millisecond, wait for the next one
                    while ms <= self._last_ms:
                        time.sleep(0.0002)
                        ms = int(time.time() * 1000) - EPOCH_MS
            elif ms < self._last_ms:  # clock went backwards; keep ids monotonic
                ms = self._last_ms
                self._seq = (self._seq + 1) & _SEQ_MASK
            else:
                self._seq = 0
            self._last_ms = ms
            return (ms << (_WORKER_BITS + _SEQ_BITS)) | (self.worker_id << _SEQ_BITS) | self._seq


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
