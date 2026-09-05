# The anatid server profile

anatid has two deployment profiles.

**Embedded** is the default and nothing here replaces it. One process opens the file, that
process may write from as many threads as it likes, and there is no daemon, no socket and no
extra hop. If your agent is one process, stop reading: `Anatid.open(...)` is the whole answer
and it is faster than anything in this document.

**Server** is for the case the embedded profile cannot serve: two or more processes that must
write the same memory. One process owns the files and everybody else talks to it over a Unix
socket or HTTP. This document is how to run that process.

The server is not a storage abstraction. There is one engine, DuckDB, and there is no pluggable
Neo4j or Postgres backend, because a client over someone else's engine gives up the measured
reason this architecture exists (2-hop recall 2.5x to 3.6x faster than the maintained Kuzu
fork). What the server changes is who holds the file, not what the file is.

---

## 1. Why you cannot avoid it

DuckDB gives one process exclusive use of a database file. This is stricter than the
documentation suggests, so it was measured directly on duckdb 1.5.5, macOS arm64:

| first process holds | second process wants | result |
| --- | --- | --- |
| read-write | read-write | `duckdb.IOException`, "Could not set lock on file" |
| read-write | **read-only** | `duckdb.IOException`, "Could not set lock on file" |
| read-only | read-write | `duckdb.IOException`, "Could not set lock on file" |
| read-only | read-only | ok |

Read the second row again. A writer excludes *readers* as well. That kills the arrangement
everybody proposes first, which is "write through the server and read the file directly for
speed". There is no such split. While the server holds a tenant's file, nothing else on the
machine can open it at all, and every read comes back over the wire along with every write.

Two consequences follow for the rest of this document. Backups have to be taken by the server
process (section 8), and a restore has to happen while the server is stopped.

## 2. What a round trip costs

Measured on this machine (macOS arm64, duckdb 1.5.5, Python 3.12), one tenant of 3,001 memories
with 384-dimension embeddings, 300 calls each, p50 in process against p50 over a Unix
socket:

| call | embedded | over the socket | difference |
| --- | --- | --- | --- |
| `get()` | 0.49 ms | 1.01 ms | +0.52 ms, 2.07x |
| `remember()` | 2.83 ms | 4.24 ms | +1.40 ms, 1.50x |
| `recall()`, vector arm on | 25.38 ms | 24.99 ms | none measurable |
| `recall()`, no vector arm | 17.09 ms | 19.12 ms | +2.03 ms, 1.12x |

About half a millisecond of fixed cost per call. It doubles the cheapest read anatid has and
disappears entirely into a `recall()`, which is the call this profile exists to serve.

Do not quote the socket as the cost. A bare Unix-socket echo of a 128-byte frame is 7.5 us at
p50, which is 1.3% of the 589 us the round trip actually adds. The breakdown, on a reply
carrying one hydrated memory with a 384-float embedding:

```
db.get() in process                    479 us
server.call() dispatch, no socket      441 us   dispatch adds nothing measurable
encode + decode of that reply          260 us   reply frame 8,252 bytes
full round trip over the socket       1068 us   +589 us, of which codec 260, loop 329, dispatch ~0
```

The codec half is the part you can do something about. `--embeddings f32` sends embeddings as
base64 float32 instead of a JSON array of doubles:

| | bytes | encode + decode | |
| --- | --- | --- | --- |
| dim 384, `list` | 7,725 | 274.5 us | |
| dim 384, `f32` | 2,408 | 68.8 us | 3.2x smaller, 4.0x faster |
| dim 1536, `list` | 29,934 | 1,049.7 us | |
| dim 1536, `f32` | 8,552 | 210.9 us | 3.5x smaller, 5.0x faster |

The rounding is real and bounded: maximum absolute error 2.97e-08 per component at 384
dimensions, 2.98e-08 at 1536. An embedding your model produced in float32 and Python widened to
float64 loses nothing. One you computed in float64 and depend on to full precision does. It is
off by default for that reason.

## 3. Quick start

```
anatid-server start \
  --socket /run/anatid/anatid.sock \
  --pool '/var/lib/anatid/tenant-{tenant}.anatid' \
  --tenant 1 --tenant 2
```

Check the configuration before you deploy it. `--check` validates everything that can be known
without I/O, prints what it resolved and exits without binding a socket or creating a file:

```
$ anatid-server start --socket /run/anatid/anatid.sock \
    --pool '/var/lib/anatid/tenant-{tenant}.anatid' --tenant 1 --check
anatid-server: the configuration is usable.
  storage        --pool /var/lib/anatid/tenant-{tenant}.anatid
  transports     unix:/run/anatid/anatid.sock
  tenants        1
  write queue    max_depth=256 batch_max=32 workers=4
  read threads   8
  idempotency    on (ttl 86400s)
  deadline       30.0s default per request
  drain timeout  30s
  embeddings     list
  backup dir     not configured; the backup verb is refused
```

Connect from Python:

```python
from anatid.server.client import connect

with connect("/run/anatid/anatid.sock", tenant=1) as memory:
    memory.remember("the deploy at 14:05 rolled back cleanly")
    for hit in memory.recall("deploy rollback"):
        print(hit.content)
```

`anatid-server` is also `python -m anatid.server`, for a container that runs the interpreter
directly or a virtual environment that was never activated. They are the same function.

## 4. The security model

State it plainly, because most of it is filesystem permissions rather than cryptography.

### The Unix socket

The socket is created mode 0600 in a directory the server chmods to 0700. Only the user the
server runs as can connect. That is the whole authentication story for the Unix transport, and
on a machine where the server has its own user it is a good one: no token to leak, no token to
rotate, and the kernel enforces it.

The default authenticator on that socket is `AllowAllAuthenticator`, which is to say the
permissions *are* the policy. If several local users share the machine and only some of them
should reach the server, add `--allow-uid`, which checks peer credentials as well:

```
anatid-server start --socket /run/anatid/anatid.sock --allow-uid 1001 --allow-uid 1002 ...
```

### HTTP

HTTP has no filesystem permissions, so it needs a bearer token. `--http` without `--token-file`
is refused at startup:

```
$ anatid-server start --http 127.0.0.1:8787 --pool '...' --socket /run/anatid/a.sock
anatid-server: the configuration is not usable:
  --http needs --token-file. Without a token, anyone who can reach the port can read and write every tenant this server holds, and on a shared machine that includes every other local user even when the bind is 127.0.0.1. Pass --http-no-auth if this port is genuinely unreachable by anyone else.
```

A non-loopback bind without a token is refused outright and `--http-no-auth` will not override
it. Note the sentence about 127.0.0.1: a loopback port is reachable by every local user, so
`--http-no-auth` is correct only on a single-user box or in a container with one process in it.

A token file is one token per line:

```
# <token> [name=NAME] [tenants=1,2] [read-only]
7f3c9e01a4b2  name=ingest-worker  tenants=1
2b8d40fe66a1  name=dashboard      tenants=1,2  read-only
9c11ab73de05  name=admin
```

A token with no `tenants=` may name every tenant the server holds and is the only kind that may
ask for a backup. `read-only` refuses every write verb. The file should be mode 0600; the
server warns on stderr when it is not, and still loads it, because a secret mounted into a
container is routinely 0444 and root-owned and refusing to start would be the worse failure.

The two transports authenticate differently and a server can serve both at once. `--socket` and
`--http --token-file` together give you local processes with no token and remote ones with one.

### The tenant boundary

Two shapes, and they are not equally strong.

`--pool '/var/lib/anatid/tenant-{tenant}.anatid'` gives each tenant its own file. That is the
isolation anatid actually offers: a bug in a query cannot reach another tenant's rows because
they are not in the file the query runs against.

`--db /var/lib/anatid/anatid.duckdb` puts every tenant in one file with a `tenant_id` column.
**In a shared file, a tenant is a namespace and not a security boundary.** A shared file also
has a concrete consequence worth naming: four verbs (`info`, `fts_status`, `rebuild_fts_index`,
`recluster`) take no tenant argument and act on the whole file, so a principal scoped to
tenant 1 can trigger a file-wide index rebuild that touches tenant 2's rows. Use `--pool` unless
you have a specific reason not to.

The boundary the server does enforce, in either shape, is enforced before any file is opened. A
request names its tenant in the envelope, the principal is checked against it, and a request for
a tenant the caller may not name never reaches that tenant's file. A refusal for a tenant you
are not allowed to see is indistinguishable from a refusal for a tenant that does not exist.
`tenant` inside `args` is rejected outright, so there is exactly one place a tenant can be
named.

### What the server does not give you

**It does not make DuckDB serializable.** DuckDB provides optimistic snapshot isolation with
write-write aborts. Routing writes through one process does not upgrade that, and no
configuration flag here changes it. `db.atomic()` still raises `ConflictError` and callers still
have to retry. What the server does change is where the aborts happen: between threads inside
one process, where the per-tenant queue can order and batch them, instead of between processes,
where they could not have happened at all because the second process could not open the file.

**It is not a replication or failover system.** One process owns the files. If it dies, nothing
serves those tenants until it or a replacement starts. Health and readiness (section 7) are
there so a supervisor can do the restarting.

**A deadline does not cancel a running write.** Deadlines are enforced before a write's
transaction opens. A write whose deadline expires while it is queued is dropped cleanly and
reported. One that expires while it is running finishes or aborts on its own, and the client
gets `DeadlineExceeded` for a write that may well have committed. Retry those with the same
idempotency key; that is what the key is for.

## 5. The operator surface

```
anatid-server start    --socket PATH [--http HOST:PORT --token-file F]
                       [--db PATH | --pool TEMPLATE] [--max-queue N] [--drain-timeout S]
                       [--check]
anatid-server stop     --socket PATH [--timeout S] [--force]
anatid-server status   --socket PATH [--json]
anatid-server backup   --socket PATH --tenant N [--name FILE]        (online)
anatid-server backup   --pool TEMPLATE --tenant N --to PATH          (offline)
anatid-server restore  --pool TEMPLATE --tenant N BACKUP             (offline)
anatid-server doctor   --socket PATH --tenant N [--shallow]
```

Every client subcommand takes `--http HOST:PORT --token TOKEN` in place of `--socket`.

Exit codes, because a script reads them:

| code | meaning |
| --- | --- |
| 0 | the command did what it was asked |
| 1 | it failed at run time: could not connect, the server returned an error, a copy failed |
| 2 | the configuration or the arguments are wrong; nothing was started, nothing was written |
| 3 | the server answered and the answer is bad news: not ready, or `doctor` found errors |

`start` exits 0 when its shutdown drained cleanly and 1 when the drain timed out with writes
still queued. Writes that were accepted and then abandoned are a failure an operator has to see,
so they are not quiet.

### The flags that decide behaviour

| flag | default | what it does |
| --- | --- | --- |
| `--pool TEMPLATE` | | one file per tenant. Must contain `{tenant}` or `{label}` |
| `--db PATH` | | one shared file, tenants as namespaces. Exactly one of the two |
| `--tenant N` | none | open at startup and hold readiness false until migrated. Repeatable |
| `--no-create-tenants` | off | serve only `--tenant` plus the files that already exist |
| `--max-queue N` | 256 | queued writes per tenant before the server answers busy |
| `--batch-max N` | 32 | writes that may share one transaction |
| `--workers N` | 4 | tenants written in parallel |
| `--read-workers N` | 8 | threads answering reads |
| `--drain-timeout S` | 30 | seconds to finish queued writes on shutdown |
| `--deadline S` | 30 | default per-request budget |
| `--embeddings f32` | `list` | base64 float32 embeddings on the wire (section 2) |
| `--backup-dir DIR` | none | where the online backup verb writes. Without it, backups are refused |
| `--pid-file PATH` | none | written after the socket appears, removed after the drain |
| `--allow-uid UID` | none | restrict the Unix socket to these uids. Repeatable |

`--no-create-tenants` is worth a sentence. By default the server opens a tenant's file the first
time a request names it, which is what a service that provisions tenants from its own traffic
wants. It also means an authorised client that may name any tenant can turn a loop over
`remember(tenant=i)` into a directory of database files. That is not a tenant boundary problem,
because the caller was entitled to name them; it is unbounded resource use. `--no-create-tenants`
confines the server to the tenants it was told about plus the files already on disk, so a tenant is
created out of band or not at all. A request for anything else is refused with
`TenantIsolationError` and nothing is created.

Batching is worth setting deliberately. Measured with 16 clients writing 100 memories each to
one tenant:

| `--batch-max` | wall clock | writes/s | writes per transaction | transactions |
| --- | --- | --- | --- | --- |
| 1 | 3597 ms | 445 | 1.0 | 1600 |
| 32 | 2772 ms | 577 | 8.0 | 199 |

1.30x, from turning 1600 transactions into 199. Only verbs marked batchable share a
transaction; `forget`, `prune`, `maintain_indexes`, `rebuild_fts_index` and `recluster` always
get one of their own.

### Backpressure and idempotency

When a tenant's queue reaches `--max-queue`, the server answers `busy` with a `retry_after`
rather than blocking. A client that retries on `busy` sees a slow server; a client that does not
sees an error it must handle. Over HTTP, `busy` is status 429. Nothing was written: a `busy`
answer is safe to send again as it stands.

`retry_after` is an upper bound to spread the crowd, not a reservation. It is computed from the
rate the tenant is actually draining at, multiplied by the number of writes that tenant has
refused since it was last under its high-water mark, and then jittered down by up to half. The
multiplier is what makes it usable by more than one caller: the time to drain the overflow is the
time until *one* more write fits, so sixteen clients told that number came back together and
fifteen were refused again. The jitter is what keeps them from returning in step. A client under
sustained overload should still back off further than the number it was handed.

Idempotency keys are on by default with a 24 hour TTL. A client that sends the same key twice
gets the first response back rather than a second write. The key and the write it guards commit
in one transaction in the tenant's own file (table `anatid_idempotency`), so a crash cannot
separate them. This is the correct answer to the `DeadlineExceeded` sharp edge above: retry with
the key you used, and a write that already committed will not commit twice.

## 6. Running it under an init system

### systemd

`/etc/systemd/system/anatid.service`:

```ini
[Unit]
Description=anatid memory server
After=network.target

[Service]
Type=simple
User=anatid
Group=anatid

# /run/anatid, 0700, created at start and removed at stop.  The socket lives here.
RuntimeDirectory=anatid
RuntimeDirectoryMode=0700
# /var/lib/anatid, where the tenant files live.
StateDirectory=anatid
StateDirectoryMode=0700

ExecStartPre=/opt/anatid/venv/bin/anatid-server start \
    --socket /run/anatid/anatid.sock \
    --pool '/var/lib/anatid/tenant-{tenant}.anatid' \
    --tenant 1 --backup-dir /var/lib/anatid/backups --check
ExecStart=/opt/anatid/venv/bin/anatid-server start \
    --socket /run/anatid/anatid.sock \
    --pool '/var/lib/anatid/tenant-{tenant}.anatid' \
    --tenant 1 --backup-dir /var/lib/anatid/backups \
    --drain-timeout 30 --log-level info

# SIGTERM starts the drain.  TimeoutStopSec MUST be larger than --drain-timeout, or systemd
# sends SIGKILL in the middle of a write transaction and the drain never finishes.
KillSignal=SIGTERM
TimeoutStopSec=45
Restart=on-failure
RestartSec=2

NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes

[Install]
WantedBy=multi-user.target
```

Three things in there are load-bearing. `ExecStartPre` with `--check` fails the unit at start
rather than at the first request. `RuntimeDirectoryMode=0700` is the authentication for the Unix
socket; the default 0755 would let any local user reach it. `TimeoutStopSec` larger than
`--drain-timeout` is the difference between a clean drain and a killed transaction.

For a readiness probe, `anatid-server status --socket /run/anatid/anatid.sock` exits 0 when
ready and 3 when not.

### launchd

`~/Library/LaunchAgents/dev.anatid.server.plist` for a user agent, or
`/Library/LaunchDaemons/` for a system one:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.anatid.server</string>

  <key>ProgramArguments</key>
  <array>
    <string>/opt/anatid/venv/bin/anatid-server</string>
    <string>start</string>
    <string>--socket</string>
    <string>/tmp/anatid/anatid.sock</string>
    <string>--pool</string>
    <string>/Users/Shared/anatid/tenant-{tenant}.anatid</string>
    <string>--tenant</string>
    <string>1</string>
    <string>--drain-timeout</string>
    <string>30</string>
  </array>

  <key>RunAtLoad</key>       <true/>
  <key>KeepAlive</key>       <dict><key>SuccessfulExit</key><false/></dict>

  <!-- launchd sends SIGTERM and then SIGKILL after ExitTimeOut.  The default is 20 seconds,
       which is SHORTER than the 30 second drain above, so it has to be raised here too. -->
  <key>ExitTimeOut</key>     <integer>45</integer>

  <key>StandardOutPath</key> <string>/Users/Shared/anatid/server.out</string>
  <key>StandardErrorPath</key><string>/Users/Shared/anatid/server.err</string>
</dict>
</plist>
```

Load with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.anatid.server.plist`,
unload with `launchctl bootout gui/$(id -u)/dev.anatid.server`.

The socket path is the one macOS-specific trap. `sun_path` in `sockaddr_un` is 104 bytes here
against 108 on Linux, so a socket under a long home directory path fails to bind with an error
that never mentions length. `--check` measures it and says so, which is why `/tmp/anatid/` is in
the example rather than a path under `~/Library/Application Support/`.

## 7. Health and readiness

Two questions, deliberately separate, because a supervisor and a load balancer want different
answers.

**Health is: is this process alive.** `GET /health`, or the `health` verb. It answers 200 while
the server is `serving` and also while it is `draining`, because a draining server is finishing
work someone is waiting for and restarting it would throw that work away. It answers 503 only
when the process is up but not serving. A supervisor restarts on this.

**Readiness is: can this process take traffic.** `GET /ready`, or the `ready` verb, or
`anatid-server status`. It is true only when all four of these hold:

| condition | false means |
| --- | --- |
| every tenant named by `--tenant` has its file open | the file is missing, unreadable, or locked by another process |
| every one of them is at the schema version this build expects | a migration is running or has not run. Sending traffic now would run it under load |
| the queue is accepting | the server is draining toward shutdown |
| no tenant's queue is at the high-water mark (80% of `--max-queue`) | that tenant is saturated. Shed load rather than deepening the queue |

A load balancer reads readiness. A server that is merely busy is healthy and not ready, and that
distinction is exactly what keeps a busy server from being restarted and an unmigrated one from
being sent traffic.

`Readiness.detail` says which of the four failed, in words. `anatid-server status` prints it on
the last line:

```
$ anatid-server status --socket /run/anatid/anatid.sock
anatid 0.3.0  protocol 1  pid 64968
address        unix:/run/anatid/anatid.sock
status         serving
uptime         4.0s
ready          yes
accepting      yes
open files     1
schema         expected v4  1=v4
queue depth    0 deepest tenant (high water 204)
queue totals   submitted=0 completed=0 failed=0 rejected=0 expired=0 replayed=0
batching       0.0 writes per transaction over 0 transactions
```

`--json` prints the same three objects (`health`, `ready`, `queue`) for a monitor. Exit code 0
when ready, 3 when not, 1 when the server could not be reached at all.

### Which probes need a credential

| route | without a token | with a token |
| --- | --- | --- |
| `GET /health` | the whole record. It names a pid, an uptime and a version and nothing about tenants, and a supervisor has no credential to present | the same |
| `GET /ready` | the verdict, which is what a load balancer reads: `ready`, `accepting`, `queues_below_high_water` | the tenant list and the per-tenant schema versions, for the tenants this principal may name |
| `GET /metrics` | the whole record on a listener with no token; 401 on one that has a token | series for the tenants this principal may name |
| `POST /rpc` | 401 | the verbs this principal may call |

On a listener with no token at all (a Unix-socket or loopback deployment) nothing is reduced,
because there is no credential to present and nothing to hide behind one. On a token-protected
listener the tenant list is scoped for the reason the tenant boundary exists: a principal must not
be able to learn whether a tenant it may not name exists, and an anonymous caller may name none of
them. The `ready` verb over `/rpc` and over the Unix socket is scoped the same way, so the probe
is not a second door onto the question the boundary already answers. `Readiness.ready` itself is
never withheld: it describes the process, not a tenant.

### HTTP status codes

| code | when |
| --- | --- |
| 200 | the verb ran |
| 400 | malformed request, bad arguments, or an idempotency key reused for a different request |
| 401 | no token, or a token this server does not know |
| 403 | a tenant this principal may not name, or a write from a read-only principal |
| 404 | no such route, or the verb found nothing |
| 409 | `ConflictError`: a write-write conflict. Retry the transaction |
| 429 | `busy`: this tenant's queue is full. Honour `retry_after` |
| 500 | an error inside a verb |
| 503 | the server is shutting down, or a deadline expired |

## 8. Backup and restore

**Do not `cp` a file the server holds.** Two reasons. The lock matrix in section 1 means you
cannot open it to read it anyway, and even offline, a copy taken without its `.wal` sidecar is a
database missing its most recent commits.

### Online, while the server runs

This is the normal path and the only one available while the server is up. The server takes the
copy, because it is the only process that can open the file.

```
anatid-server start ... --backup-dir /var/lib/anatid/backups
anatid-server backup --socket /run/anatid/anatid.sock --tenant 1
anatid-server: tenant 1 backed up to /var/lib/anatid/backups/tenant-1-20260904T192944Z.anatid (2895872 bytes), written by the server that holds the file.
```

What happens: the server puts a barrier on that tenant's write queue, and while the barrier holds
the tenant's single serving slot it runs DuckDB's `COPY FROM DATABASE` inside the handle that owns
the file. That gives the copy a boundary in the tenant's own write order: every write acknowledged
before the call is in it, and no write that commits after the barrier closes is. The command prints
which guarantee it got:

```
anatid-server: tenant 1 backed up to /var/lib/anatid/backups/tenant-1-20260904T192944Z.anatid (2895872 bytes), written by the server that holds the file.
anatid-server: guarantee quiesced. Every write acknowledged before the call is in it and nothing that committed after is. Tenant 1 was paused 0.502s for it; no other tenant was.
```

Other tenants keep serving throughout and are never paused. Only the tenant named stops, and only
for the length of its own copy: measured on a 42.3 MiB file with 100,000 memories, 502 ms, which is
8.8 ms more than the same copy taken with nothing paused.

This used to be a drain rather than a barrier, and the difference is the whole point.
`WriteQueue.drain_tenant` waits for the tenant's queue to empty and then returns, stopping nothing,
so a write submitted between the drain returning and the copy starting committed into the copy. The
boundary that describes is the weaker `snapshot` one however long the drain waited, and under
continuous writes the drain could spend its whole `--drain-timeout` budget and still deliver only
that. A barrier holds.

If the barrier cannot be taken inside `--drain-timeout` the copy is refused rather than downgraded,
and nothing was paused and nothing was written. If it is taken but expires mid-copy (see
`hold_timeout`, ten minutes by default), the tenant resumes and the report says `snapshot` rather
than claiming a boundary it lost:

```
anatid-server: guarantee snapshot. The barrier could not be held, so this is a consistent copy of committed data rather than the point where everything acknowledged so far had landed. A write acknowledged during the copy may or may not be in it. Retry when the tenant is quieter, or raise --drain-timeout, which is the budget the barrier waits within.
```

The reply carries `guarantee`, `quiesced`, `paused_s` and `waited_s` for a monitor that wants to
decide on the numbers. In the library the same thing is `AnatidServer.backup_tenant(tenant, dest)`,
which returns a `BackupReport`, or `anatid.server.BackupCoordinator` for the fuller surface:
several tenants, Parquet export, restore, retention and verification.

Two restrictions, both because a backup writes a file:

- the client chooses a **name**, the server chose the **directory**. `--name` must be a plain
  file name. A path, a `..`, or a leading dot is refused.
- the caller must be an administrator: a principal with no tenant restriction and not read-only.
  Over the Unix socket that is the default principal. Over HTTP it is a token with no `tenants=`
  field. A token scoped to one tenant can read and write that tenant's memories and cannot make
  the server write files.

A server started without `--backup-dir` does not have the verb at all and says so.

### Offline, with the server stopped

```
anatid-server stop --socket /run/anatid/anatid.sock
anatid-server backup --pool '/var/lib/anatid/tenant-{tenant}.anatid' --tenant 1 \
    --to /backups/tenant-1.anatid
```

Same mechanism, run by this process instead of the server. An existing destination is refused
unless you pass `--overwrite`, because a backup that silently replaces the previous one is one
crash away from leaving you with neither.

The refusal is the same on all three paths and never writes anything. Offline it exits 2, an
argument problem; online it exits 1, because the server is the one that refused; over HTTP it
is a 400. Passing `--overwrite` lifts it in every case. Prefer the default timestamped name to
`--name` plus `--overwrite`: a backup that replaces yesterday's is not a backup history.

### Restore

Restore is offline only. DuckDB gives one process exclusive use of a file, so the server has to
be stopped first, and `restore` refuses to run if it can tell that one is still up.

```
anatid-server stop --socket /run/anatid/anatid.sock
anatid-server restore --pool '/var/lib/anatid/tenant-{tenant}.anatid' --tenant 1 \
    --socket /run/anatid/anatid.sock /backups/tenant-1.anatid
```

Passing `--socket` to `restore` is what lets it refuse while a server is listening there. It
also checks that the file is a DuckDB database, that it has an anatid schema in it, and that its
schema version is not newer than this build understands, before it touches anything.

The previous file is renamed aside, not deleted:

```
anatid-server: restored /backups/tenant-1.anatid to /var/lib/anatid/tenant-1.anatid (anatid schema v4).
  the previous file is at /var/lib/anatid/tenant-1.anatid.replaced-20260904T193012Z; delete it once the restore is verified.
```

Its write-ahead log moves with it. A restored file left beside the previous file's `.wal` is a
database DuckDB will try to finish writing on the next open, which is how a restore turns into a
corruption.

If the backup is at an older schema version, the restore says so and the next open migrates it.
Readiness stays false while that runs, which is exactly what you want: the migration finishes
before traffic arrives.

### A backup schedule that works

```
0 3 * * *  anatid-server backup --socket /run/anatid/anatid.sock --tenant 1 \
             && find /var/lib/anatid/backups -name 'tenant-1-*.anatid' -mtime +14 -delete
```

The default name carries a UTC timestamp (`tenant-1-20260904T192944Z.anatid`), so backups do not
overwrite each other and sort chronologically. Verify one occasionally by opening it:
`anatid-server doctor --db /var/lib/anatid/backups/tenant-1-20260904T192944Z.anatid --tenant 1`.

## 9. Integrity checks

`doctor` runs anatid's integrity checks and never repairs anything. Over the wire against a
running server, or directly against a file when none is running:

```
anatid-server doctor --socket /run/anatid/anatid.sock --tenant 1
anatid-server doctor --pool '/var/lib/anatid/tenant-{tenant}.anatid' --tenant 1
```

Exit 0 when nothing of severity ERROR was found (warnings included), 3 when something was.
`--shallow` skips the row-scanning checks, which is the difference between a probe you can run
every minute and one you run nightly.

## 10. Shutdown, and what it guarantees

`anatid-server stop` sends SIGTERM and waits. SIGTERM starts a drain. A second SIGTERM abandons
it, because an operator who sends one twice has decided that waiting is worse than losing what
is queued.

The order the server shuts down in is the guarantee:

1. Listeners are told to close, so nothing new arrives.
2. The queue stops accepting **immediately**, so a request already decoded gets `ShuttingDown`
   rather than being committed by a server that has announced it is leaving.
3. The queues drain under `--drain-timeout`. What is left when the budget runs out is completed
   with `ShuttingDown` and named per tenant in the report.
4. The open connections are cancelled.
5. The listeners are waited on, under a bound of at most one second.
6. The read threads are joined, so nothing is still touching a file.
7. Every open file is released, which folds its write-ahead log in and gives up the lock.
8. The socket is unlinked.

Steps 2 and 5 are in that order for a reason, and it is not the obvious one.
`asyncio.Server.wait_closed()` does not return while a connection handler is still running, and a
live deployment always has connections open, so a shutdown that waits on the listener first lasts
as long as the last client chooses to stay connected. Measured on Python 3.12 with the wait in the
wrong place: one idle client held a server past 25 seconds against a 5 second budget and it exited
0.03 seconds after that socket closed, and 64 writing clients had about 2,100 further writes
accepted and committed after the SIGTERM. Under a supervisor with a TERM-then-KILL policy that
means the server is normally killed rather than drained. The exit is now bounded by
`--drain-timeout` plus about a second, whether or not anyone is connected.

Step 7 before step 8 is why `stop` can report "its files are released" honestly: a socket that is
gone means the drain finished and the next process can open the files. That is the signal to
wait on, not the pid, because a process that has exited but has not been reaped still answers
`kill(pid, 0)`.

A second SIGTERM reaches the drain rather than queueing behind it. The waiting happens on a
worker thread whose deadline is already fixed, so the second signal tells the queue to give up:
whatever is still queued is completed with `ShuttingDown` and counted in the report the server
logs on the way out.

If the drain does not finish inside `--timeout`, `stop` says so and exits 1 without killing
anything. `--force` sends SIGKILL instead, and the queued writes are lost. Prefer raising the
timeout.

Note for anyone embedding `AnatidServer` rather than using this CLI: `shutdown()` closes the
pooled handles it opened. If you passed a `DatabasePool` in and keep using it afterwards, you
get reopened handles rather than the same objects. `pool.get()` reopens transparently, so
restart in place works, but the files are not held after shutdown. A server built with
`database=` does not close that handle, because it belongs to you.

## 11. Troubleshooting

**"Could not set lock on file ... Conflicting lock is held in ... (PID nnnn)"**
Another process has the file. That is almost always a server you thought you stopped, or a
Python REPL with an `Anatid.open` still live. `anatid-server stop`, or find the pid the message
names. A read-only open will not get you around it; see section 1.

**"--socket ... is 130 characters; this platform's sockaddr_un holds 103"**
Use a shorter path. `/run/anatid/anatid.sock` on Linux, `/tmp/anatid/anatid.sock` on macOS.

**The server starts and readiness stays false**
Read the last line of `anatid-server status`. The usual cause is a migration in progress on a
tenant named by `--tenant`, which is the system working: readiness goes true when the migration
finishes.

**Clients get 429 or `BusyError`**
A tenant's queue hit `--max-queue`. Either the writers are faster than the disk, in which case
raise `--batch-max` and `--workers` before you raise `--max-queue`, or one tenant is being
hammered and should be rate-limited at the client. Raising `--max-queue` alone converts a fast
rejection into a slow one.

**`DeadlineExceeded` on a write that seems to have happened**
It probably did. See section 4: deadlines are not enforced inside a running transaction. Retry
with the same idempotency key.

**`ConflictError`**
Two writers touched the same rows and DuckDB aborted one. Retry the transaction. This is
snapshot isolation working as designed and is not something the server can remove.

**A checkpoint hangs, or raises "Cannot CHECKPOINT: there are other write transactions active"**
Use `db.checkpoint()`, not `db.execute('CHECKPOINT')`. Measured on duckdb 1.5.5, the raw statement
runs on this thread's cursor, where it succeeds on a handle for a file this process created and
raises that `TransactionException` on a handle for a file that already existed, from any thread,
on a handle that has run nothing else. `FORCE CHECKPOINT`, which DuckDB's message suggests, is
worse: from a cursor it does not raise, it never returns. `Anatid.checkpoint()` issues it on the
handle's root connection, where it works in both cases (measured: a 2,146,504 byte `.wal` folded
to 0). It does not force, so another thread with a write transaction genuinely open still raises,
which is the honest answer.

The server does not call it on shutdown. It releases the files instead, which folds the log as a
side effect and additionally gives up DuckDB's exclusive lock, so the replacement process can open
the file at all. A server built on `--db` gets a checkpoint and nothing more, because that handle
belongs to the caller.

## 12. Connections and concurrency, for client authors

Requests on one connection are answered strictly in order. There is no pipelining and no
out-of-order interleaving. The `id` field is echoed back so a future client could match replies
to requests, but today a client that wants two calls in flight opens two connections.

The wire is length-prefixed JSON: a 4-byte big-endian length, then that many bytes of UTF-8
JSON, one frame per message, 64 MiB maximum checked before allocation. `POST /rpc` takes the
same request object as a plain body with no length prefix. `deadline` is relative seconds
measured by the server from the moment it decodes the request, not an absolute timestamp,
because a client whose clock is a minute fast would otherwise expire every request on arrival.

### Ids are not JSON numbers

anatid mints 63-bit ids. A JSON number is a double in JavaScript, whose largest exact integer is
`2**53 - 1`, so an id sent as a number comes back changed and nothing raises. Measured on this
build, 1,984 of 2,000 freshly minted ids change value under a plain `JSON.parse`. So any integer
this codec cannot fit in a JSON number travels tagged, with its digits in a string:

```json
{"memory_id": {"__anatid__": "id", "v": "883768514279557120"}}
```

The test is on magnitude, not on a field name, so a `count(*)` of 12 is still the number 12 and
the id nobody thought of is still safe. It applies in both directions and to error details as
well as results. `anatid.server.protocol.decode_value` turns the tag back into an integer, so a
Python client sees `int` on both sides; a client in another language reads `v` and keeps the
digits. This is the same rule `anatid.integrations.wire` applies at the MCP and Agents SDK
boundaries, and it uses that module's definition rather than a second copy of it.

`anatid.server.protocol` is the whole format and `anatid.server.client.AnatidClient` is a
reference implementation of it.

---

## See also

- `docs/architecture.md`: what is in the file and how a `recall()` is answered.
- `docs/design/derived-index-framework.md`, section "Concurrency and isolation": the design this
  profile implements.
- `anatid-server start --help`, and `--help` on each subcommand.
