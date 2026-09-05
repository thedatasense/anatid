"""Two writer processes and a reader against one anatid memory, then the same program embedded.

Run it::

    python examples/server_demo.py

No API key, no network, about twenty seconds.  It makes its own temporary directory and removes
it at the end.

What it is for
--------------
anatid is a DuckDB file, and DuckDB allows exactly ONE read-write process per file.  Two agents
in two processes cannot both remember into the same memory, and the second one does not get a
queue or a wait, it gets an exception at ``Anatid.open``.  Act 1 of this demo shows that
happening.  Act 2 starts a server that owns the file and runs the same two writers plus a reader
against it, concurrently, and they all succeed.  Act 3 runs the identical workload function
embedded, by changing which handle it is given.

The one line that changes::

    db = Anatid.open(path, tenant=1)  # embedded: one writer process
    db = AnatidClient.connect(socket_path, tenant=1)  # server: as many as you like

``workload()`` below is written once and run three times: by each server client subprocess, and
by the embedded act.  It never asks which one it has.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anatid import Anatid, DatabasePool
from anatid.server import AnatidServer, ServerConfig
from anatid.server.client import AnatidClient

DIM = 8
WRITES = 25
TENANT = 1


# --------------------------------------------------------------------------------------
# The program.  One function, either handle, no idea which.
# --------------------------------------------------------------------------------------


def workload(db, *, agent: str, writes: int = WRITES) -> dict:
    """Remember some things, relate them, correct one, and read the graph back.

    ``db`` is an :class:`anatid.Anatid` or an :class:`anatid.server.client.AnatidClient`.  Every
    call below exists on both with the same arguments and returns the same objects, which is the
    claim this file is here to demonstrate.
    """
    db.upsert_entity(agent, kind="agent")
    db.relate(agent, "shared-topic", rel_kind="works_on")
    first = None
    for i in range(writes):
        memory = db.remember(
            f"{agent} observed event {i}",
            entities=[agent, "shared-topic"],
            embedding=[i / writes] + [0.0] * (DIM - 1),
            kind="observation",
        )
        first = first or memory
    corrected = db.update(
        first.memory_id, f"{agent} observed event 0, corrected", expected_version=first.version
    )
    db.reinforce(corrected.memory_id, amount=3)
    hits = db.recall(f"{agent} observed", k=5)
    return {
        "agent": agent,
        "written": writes,
        # What this agent wrote: the memories the agent entity is directly about.  NOT
        # recall_2hop(agent), which is the point of the graph and would be the wrong number
        # here: two hops from alpha reaches shared-topic and then everything beta wrote too.
        "about_me": len(db.context(agent, limit=1000)),
        "reachable": len(db.recall_2hop(agent, limit=1000)),
        "everyones": len(db.recall_2hop("shared-topic", limit=1000)),
        "corrected": db.get(corrected.memory_id).content,
        "access_count": db.get(corrected.memory_id).access_count,
        "top_hit": hits[0].memory.content if hits else None,
        "arms": ",".join(hits.arms),
        "memories_in_tenant": db.stats()["memories"],
    }


# --------------------------------------------------------------------------------------
# Subprocess entry points.  Each one is a whole separate process talking to the server.
# --------------------------------------------------------------------------------------


def run_writer(socket_path: str, agent: str) -> int:
    with AnatidClient.connect(socket_path, tenant=TENANT) as db:
        started = time.perf_counter()
        result = workload(db, agent=agent)
        elapsed = time.perf_counter() - started
    print(
        f"  writer {agent:<6} pid {os.getpid():<7} {result['written']} writes in "
        f"{elapsed:.2f}s  about_me={result['about_me']:<3} reachable={result['reachable']:<3} "
        f"corrected={result['corrected']!r}"
    )
    return 0


def run_reader(socket_path: str, seconds: float = 6.0) -> int:
    """Read the same tenant, from a third process, while the two writers are writing."""
    reads = 0
    seen: list[int] = []
    with AnatidClient.connect(socket_path, tenant=TENANT) as db:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            hits = db.recall("observed event", k=5)
            seen.append(db.stats()["memories"])
            reads += 1
            if not hits:
                time.sleep(0.02)
    print(
        f"  reader        pid {os.getpid():<7} {reads} recalls while they wrote; the tenant grew "
        f"{min(seen)} -> {max(seen)} under it"
    )
    return 0


def try_second_writer(path: str) -> int:
    """Open an anatid file that another process already holds.  This is expected to fail."""
    try:
        Anatid.open(path, tenant=TENANT, embedding_dim=DIM).close()
    except Exception as exc:  # noqa: BLE001 - the whole point is to print what it was
        first = str(exc).splitlines()[0]
        print(f"  second process: {type(exc).__name__}: {first[:120]}")
        return 0
    print("  second process: opened it, which this build was not expected to allow")
    return 1


# --------------------------------------------------------------------------------------
# The server, on its own event loop in a background thread.
# --------------------------------------------------------------------------------------


class ServerThread:
    """A running :class:`AnatidServer`, startable and stoppable from ordinary blocking code.

    The loop is on a thread because the rest of this file is blocking.  It waits on an event
    rather than ``serve_forever()``, which installs signal handlers and so only works on the
    main thread.
    """

    def __init__(self, pool: DatabasePool, config: ServerConfig) -> None:
        self.pool = pool
        self.config = config
        self.loop = asyncio.new_event_loop()
        self.server: AnatidServer | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="anatid-server", daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)

        async def main() -> None:
            try:
                self.server = AnatidServer(pool=self.pool, config=self.config)
                await self.server.start()
            except BaseException as exc:  # noqa: BLE001 - handed to the caller through _error
                self._error = exc
                self._ready.set()
                return
            self._ready.set()
            await asyncio.Event().wait()

        try:
            self.loop.run_until_complete(main())
        except RuntimeError:
            pass  # the loop was stopped by stop(), below
        finally:
            self.loop.close()

    def start(self) -> "ServerThread":
        self._thread.start()
        if not self._ready.wait(30):
            raise RuntimeError("the server did not start within 30 seconds")
        if self._error is not None:
            raise self._error
        return self

    def stop(self) -> None:
        """Drain and release the files.

        ``AnatidServer.shutdown()`` is bounded by ``ServerConfig.shutdown_timeout`` whether or
        not a client is still connected: it stops the queue accepting, drains under that budget,
        cancels the open connections and only then waits on the listener.  This demo calls it
        after its clients have finished anyway, because it wants their writes, not because it
        has to.
        """
        if self.server is not None:
            future = asyncio.run_coroutine_threadsafe(self.server.shutdown(), self.loop)
            future.result(60)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(30)


# --------------------------------------------------------------------------------------
# The three acts.
# --------------------------------------------------------------------------------------


def act_one_two_processes_one_file(base: Path) -> None:
    print("ACT 1  Two processes, one file, no server")
    print("-" * 78)
    path = base / "embedded" / "solo.anatid"
    path.parent.mkdir(parents=True, exist_ok=True)
    with Anatid.open(path, tenant=TENANT, embedding_dim=DIM) as db:
        db.remember("the first process holds this file")
        print(f"  first process:  holding {path.name} read-write, {db.stats()['memories']} memory")
        done = subprocess.run(
            [sys.executable, __file__, "--role", "second-writer", "--path", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        sys.stdout.write(done.stdout)
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
    print("  So: one writer process per file. That is the constraint the server profile exists")
    print("  for. It is not a bug in anatid and it is not fixed by a lock or a retry.\n")


def act_two_many_processes_one_server(base: Path) -> Path:
    print("ACT 2  Two writer processes and a reader, through one server")
    print("-" * 78)
    socket_dir = Path(tempfile.mkdtemp(prefix="anatid-demo-sock-", dir="/tmp"))
    socket_path = socket_dir / "anatid.sock"
    template = str(base / "served" / "t_{tenant}.anatid")
    pool = DatabasePool(template, embedding_dim=DIM, max_open=4)
    config = ServerConfig(
        socket_path=socket_path, tenants=(TENANT,), workers=1, batch_max=8, max_depth=256
    )
    server = ServerThread(pool, config).start()
    assert server.server is not None
    print(f"  server pid {os.getpid()} listening on {', '.join(server.server.endpoints())}")
    try:
        started = time.perf_counter()
        procs = [
            subprocess.Popen(
                [sys.executable, __file__, "--role", role, "--socket", str(socket_path)]
                + (["--agent", agent] if agent else []),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for role, agent in (("writer", "alpha"), ("writer", "beta"), ("reader", None))
        ]
        failed = 0
        for proc in procs:
            out, err = proc.communicate(timeout=180)
            sys.stdout.write(out)
            if proc.returncode != 0:
                failed += 1
                sys.stderr.write(err)
        elapsed = time.perf_counter() - started
        with AnatidClient.connect(socket_path, tenant=TENANT) as db:
            stats = db.stats()
            report = db.doctor()
            queue_state = db.ready()
            print(
                f"  all three finished in {elapsed:.2f}s with {failed} failures; the tenant holds "
                f"{stats['memories']} memories and {stats['entities']} entities"
            )
            alpha = len(db.context("alpha", limit=1000))
            beta = len(db.context("beta", limit=1000))
            shared = len(db.recall_2hop("shared-topic", limit=1000))
            print(
                f"  alpha wrote {alpha} current memories, beta wrote {beta}, and shared-topic "
                f"reaches all {shared} of them"
            )
            print(
                f"  ({stats['memories']} rows against {alpha + beta} current: each agent "
                f"superseded its first memory, and the closed version stays on disk)"
            )
            print(
                f"  doctor: ok={report.ok} findings={len(report.findings)}   ready={queue_state.ready}"
            )
    finally:
        server.stop()
        pool.close_all()
        shutil.rmtree(socket_dir, ignore_errors=True)
    print("  Two writer processes wrote one tenant at the same time. The file was never held by")
    print("  more than one process: the server held it, and they held sockets.\n")
    return base / "served" / f"t_{TENANT}.anatid"


def act_three_the_same_program_embedded(base: Path) -> None:
    print("ACT 3  The same workload function, embedded, one line different")
    print("-" * 78)
    path = base / "embedded" / "single.anatid"
    path.parent.mkdir(parents=True, exist_ok=True)
    print("  db = AnatidClient.connect(socket_path, tenant=1)   <- act 2 used this")
    print("  db = Anatid.open(path, tenant=1)                   <- this act uses this")
    with Anatid.open(path, tenant=TENANT, embedding_dim=DIM) as db:
        started = time.perf_counter()
        result = workload(db, agent="alpha")
        elapsed = time.perf_counter() - started
    print(
        f"  embedded alpha: {result['written']} writes in {elapsed:.2f}s  "
        f"about_me={result['about_me']} corrected={result['corrected']!r} arms={result['arms']}"
    )
    print("  Same function, same keywords, same objects back. The handle is the only difference.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", default="demo", choices=("demo", "writer", "reader", "second-writer")
    )
    parser.add_argument("--socket", default="")
    parser.add_argument("--agent", default="alpha")
    parser.add_argument("--path", default="")
    args = parser.parse_args()

    if args.role == "writer":
        return run_writer(args.socket, args.agent)
    if args.role == "reader":
        return run_reader(args.socket)
    if args.role == "second-writer":
        return try_second_writer(args.path)

    base = Path(tempfile.mkdtemp(prefix="anatid-demo-"))
    print(f"anatid server demo   python {sys.version.split()[0]}   workspace {base}\n")
    try:
        act_one_two_processes_one_file(base)
        served = act_two_many_processes_one_server(base)
        act_three_the_same_program_embedded(base)
        print("-" * 78)
        print(
            f"The served tenant is a file like any other: {served.name}, "
            f"{served.stat().st_size / 1024 / 1024:.1f} MB, still openable embedded now that"
        )
        print("the server has released it.")
        with Anatid.open(served, tenant=TENANT, embedding_dim=DIM) as db:
            print(
                f"  reopened embedded: {db.stats()['memories']} memories, schema v{db.info().schema_version}"
            )
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
