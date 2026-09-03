"""Visibility is one abstraction: a static scan of the source tree and a runtime canary.

The static half walks every ``.py`` under ``src/anatid`` except ``visibility.py`` and fails when
a SQL string literal (docstrings and prose excluded) contains a hand-written tenant or time
predicate.  The runtime half builds a file with two tenants and two points in time, runs every
public read verb on both graph-expansion paths, and checks every returned row against a
pure-Python oracle built from the raw tables: nothing from the wrong tenant, nothing from the
wrong time, ever.
"""

from __future__ import annotations

import ast
import datetime as _dt
import itertools
import re
from collections import deque
from pathlib import Path

import pytest

import anatid
from anatid import Anatid, AsOf, NotFoundError, Visibility, visible_at
from anatid import ids as ids_mod
from anatid import schema as schema_mod
from anatid import visibility as visibility_mod
from anatid.csr import discover_extension_path, extension_unsupported, frontier_sql

from conftest import BUILT_EXTENSION, DIM, SPIKE_EXTENSION, T0, vec

SRC = Path(anatid.__file__).resolve().parent
HOUR = _dt.timedelta(hours=1)
MINUTE = _dt.timedelta(minutes=1)

# ============================================================================ static scan

#: Fragments of the time predicate.  Any of these in a SQL literal outside visibility.py is a
#: hand-written copy of the rule that module owns.  Aliases (``m.valid_to``) match too.
TEMPORAL_FRAGMENTS = [
    re.compile(r"valid_to\s+IS\s+NULL", re.IGNORECASE),
    re.compile(r"tx_to\s+IS\s+NULL", re.IGNORECASE),
    re.compile(r"valid_from\s*<=", re.IGNORECASE),
    re.compile(r"tx_from\s*<=", re.IGNORECASE),
    re.compile(r"valid_to\s*>", re.IGNORECASE),
    re.compile(r"tx_to\s*>", re.IGNORECASE),
]

#: A tenant predicate written by hand into a read statement (a literal that starts with
#: ``SELECT`` or ``WITH``).  Writes (INSERT / UPDATE / DELETE) identify rows; reads scope them,
#: and scoping is visibility's job.
TENANT_FRAGMENT = re.compile(r"tenant_id\s*=\s*(\?|\{)", re.IGNORECASE)
READ_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

#: What makes a literal SQL rather than prose: it starts with a statement or clause keyword,
#: or it starts with a bitemporal or tenant column reference (a bare predicate fragment such
#: as ``"valid_to IS NULL AND tx_to IS NULL"`` assigned to a constant).  A tool description or a
#: log message that quotes the rule in a sentence is neither, and is not a second copy of the
#: predicate: nothing executes it.
SQL_START = re.compile(
    r"^\s*\(?\s*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|FROM|WHERE|AND|OR|JOIN|"
    r"LEFT|INNER|CROSS|ON|ORDER|GROUP|HAVING|VALUES|SET|LIMIT|UNION|PRAGMA|RETURNING|"
    r"QUALIFY|EXISTS|NOT|CASE)\b",
    re.IGNORECASE,
)
PREDICATE_START = re.compile(
    r"^\s*\(?\s*(\w+\.)?(valid_to|valid_from|tx_to|tx_from|tenant_id)\b", re.IGNORECASE
)


def _is_sql(text: str) -> bool:
    """True when ``text`` is a statement, a clause, or a bare predicate fragment."""
    return bool(SQL_START.match(text) or PREDICATE_START.match(text))


def _string_literals(source: str) -> list[tuple[int, str]]:
    """Every string literal in ``source`` with its line, docstrings excluded.

    Implicitly concatenated literals are one ``Constant`` already; an f-string is a
    ``JoinedStr`` whose constant parts are joined with ``{}`` so a predicate split around an
    interpolation is still seen whole.
    """
    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    out: list[tuple[int, str]] = []
    inside_joined: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                    inside_joined.add(id(v))
                else:
                    parts.append("{}")
            out.append((node.lineno, "".join(parts)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and id(node) not in inside_joined
        ):
            out.append((node.lineno, node.value))
    return out


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "visibility.py")


@pytest.mark.parametrize("module", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_module_writes_the_visibility_predicate_by_hand(module: Path):
    """Every tenant and time predicate in a read comes from :mod:`anatid.visibility`.

    One test per module so a failure names the file.  The check covers SQL literals
    (:func:`_is_sql`): a statement, a clause fragment, or a bare predicate assigned to a
    constant.  Prose that quotes the rule, such as an MCP tool description telling the model
    which rows are current, is not a second copy of the predicate because nothing runs it.
    """
    violations = []
    for lineno, text in _string_literals(module.read_text(encoding="utf-8")):
        if not _is_sql(text):
            continue
        for rx in TEMPORAL_FRAGMENTS:
            if rx.search(text):
                violations.append((lineno, rx.pattern, text.strip()[:90]))
                break
        else:
            if READ_START.match(text) and TENANT_FRAGMENT.search(text):
                violations.append((lineno, "tenant predicate in a read", text.strip()[:90]))
    assert violations == [], (
        f"{module.relative_to(SRC)} writes a visibility predicate by hand; take it from "
        f"anatid.visibility instead:\n"
        + "\n".join(f"  line {ln}: [{why}] {t}" for ln, why, t in violations)
    )


def test_visibility_is_the_only_source_and_the_schema_reexports_it():
    assert schema_mod.temporal_predicate is visibility_mod.temporal_predicate
    assert visibility_mod.CURRENT_ROW_SQL in "\n".join(schema_mod.ddl_statements())
    assert Visibility(1).predicate("m", inline_tenant=True) == (
        "m.tenant_id = 1 AND m.valid_to IS NULL AND m.tx_to IS NULL",
        [],
    )
    sql, params = Visibility.at(7, T0).predicate("e")
    assert sql.startswith("e.tenant_id = ? AND e.valid_from <= ?")
    assert params == [7, T0, T0, T0, T0]
    assert visible_at(3, valid_time=T0).predicate() == (
        (
            "tenant_id = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?) "
            "AND tx_to IS NULL"
        ),
        [3, T0, T0],
    )
    assert Visibility(2).temporal("x", valid_only=True) == ("x.valid_to IS NULL", [])
    with pytest.raises(TypeError):
        Visibility(True)
    assert Visibility.at(1, AsOf(valid_time=T0)).as_of == AsOf(valid_time=T0)
    assert Visibility.at(1).as_of.is_current


def test_live_is_the_tenant_plus_the_transaction_axis():
    """``get()`` by id addresses the live version: tenant, then ``tx_to IS NULL``, no valid time."""
    assert Visibility(4).live() == ("tenant_id = ? AND tx_to IS NULL", [4])
    assert Visibility(4).live("m", inline=True) == ("m.tenant_id = 4 AND m.tx_to IS NULL", [])
    assert visibility_mod.live_row_sql("x") == "x.tx_to IS NULL"
    assert visibility_mod.LIVE_ROW_SQL == "tx_to IS NULL"
    assert not hasattr(visibility_mod, "open_interval_sql"), (
        "the valid-time-only guard would match a closed version too; the CAS guard is "
        "current_row_sql()"
    )


def test_admits_mirrors_the_sql_rule():
    v = Visibility.at(1, T0 + HOUR)
    row = {"tenant_id": 1, "valid_from": T0, "valid_to": None, "tx_from": T0, "tx_to": None}
    assert v.admits(**row)
    assert not v.admits(**{**row, "tenant_id": 2})
    assert not v.admits(**{**row, "valid_to": T0 + HOUR})  # half-open: closed at T
    assert v.admits(**{**row, "valid_to": T0 + HOUR + MINUTE})
    assert not v.admits(**{**row, "valid_from": T0 + 2 * HOUR})
    assert not v.admits(**{**row, "tx_to": T0 + HOUR})
    assert v.admits(**{**row, "tx_to": T0 + 2 * HOUR})
    assert not v.admits(**{**row, "tx_from": T0 + 2 * HOUR})
    assert Visibility(1).admits(**row)
    assert not Visibility(1).admits(**{**row, "valid_to": T0})
    assert not Visibility(1).admits(**{**row, "tx_to": T0})
    assert Visibility.at(1, AsOf(valid_time=None, tx_time=T0 + HOUR)).admits(**row)


# ============================================================================ runtime canary


class _Counter:
    """Dense ids, so the C++ extension's dense per-tenant CSR can be built in the canary."""

    def __init__(self, start: int = 1) -> None:
        self._it = itertools.count(start)

    def __call__(self) -> int:
        return next(self._it)


@pytest.fixture
def dense_ids():
    ids_mod.set_allocator(_Counter(1))
    try:
        yield
    finally:
        ids_mod.reset_allocator()


def _extension() -> Path | None:
    """The C++ extension to run the canary's extension arm against.

    Same order as ``tests/test_core.py``: ``discover_extension_path()`` first, then the Phase 0
    spike build.  The fallback matters when anatid is installed from a wheel --
    ``discover_extension_path`` walks up from ``anatid.__file__`` and finds nothing under
    site-packages -- and without it this arm skips in exactly the configuration a release is
    validated in, while the oracle test in test_core.py still runs.
    """
    for candidate in (discover_extension_path(), BUILT_EXTENSION, SPIKE_EXTENSION):
        if candidate is None or not Path(candidate).is_file():
            continue
        if extension_unsupported(candidate) is None:
            return Path(candidate)
    return None


EXPAND_PATHS = [
    "sql",
    pytest.param(
        "extension",
        marks=pytest.mark.skipif(_extension() is None, reason="C++ anatid extension not built"),
    ),
]

T1 = T0 + HOUR
SHARED_ID = 4242


def _populate(db: Anatid, tenant: int) -> dict:
    """The same story in every tenant, with tenant-specific names and text."""
    s = f"-t{tenant}"
    ents = {}
    for name in ("A", "B", "C", "D", "E"):
        ents[name] = db.upsert_entity(name + s, tenant=tenant, now=T0).entity_id
    db.relate(ents["A"], ents["B"], tenant=tenant, now=T0)
    db.relate(ents["B"], ents["C"], tenant=tenant, now=T0)
    db.relate(ents["C"], ents["D"], tenant=tenant, now=T0)  # D is three hops from A
    mems = {}
    mems["A"] = db.remember(
        f"about A{s} likes things",
        entities=[ents["A"]],
        tenant=tenant,
        embedding=vec(1, 0, 0),
        episode=f"evidence for A{s}",
        now=T0,
    ).memory_id
    mems["B"] = db.remember(
        f"about B{s} likes things",
        entities=[ents["B"]],
        tenant=tenant,
        embedding=vec(0, 1, 0),
        now=T0 + MINUTE,
    ).memory_id
    mems["C"] = db.remember(
        f"about C{s} likes things",
        entities=[ents["C"]],
        tenant=tenant,
        embedding=vec(0, 0, 1),
        now=T0 + 2 * MINUTE,
    ).memory_id
    mems["D"] = db.remember(
        f"about D{s} likes things",
        entities=[ents["D"]],
        tenant=tenant,
        embedding=vec(1, 1, 0),
        now=T0 + 3 * MINUTE,
    ).memory_id
    db.remember(
        f"shared id{s} likes things",
        entities=[ents["A"]],
        tenant=tenant,
        memory_id=SHARED_ID,
        embedding=vec(1, 0, 1),
        now=T0 + 4 * MINUTE,
    )
    # T1: the world changes
    mems["B2"] = db.supersede(
        mems["B"], f"about B{s} likes other things", tenant=tenant, embedding=vec(0, 1, 1), now=T1
    ).memory_id
    db.forget(mems["C"], tenant=tenant, now=T1)  # soft: history keeps it
    db.unrelate(ents["C"], ents["D"], tenant=tenant, now=T1)  # a closed edge: D is now 3 hops
    db.relate(ents["A"], ents["E"], tenant=tenant, now=T1)  # a new edge
    mems["E"] = db.remember(
        f"about E{s} likes things",
        entities=[ents["E"]],
        tenant=tenant,
        embedding=vec(0, 1, 0, 1),
        now=T1 + MINUTE,
    ).memory_id
    mems["gone"] = db.remember(
        f"erased{s}",
        entities=[ents["A"]],
        tenant=tenant,
        embedding=vec(1, 1, 1),
        now=T1 + 2 * MINUTE,
    ).memory_id
    db.forget(mems["gone"], hard=True, tenant=tenant, now=T1 + 3 * MINUTE)
    return {"entities": ents, "memories": mems}


class _Oracle:
    """Pure-Python visibility over the raw tables, for one tenant and one scope."""

    def __init__(self, db: Anatid, vis: Visibility) -> None:
        self.vis = vis
        con = db.connection
        self.memories = con.execute(
            "SELECT memory_id, tenant_id, created_at, valid_from, valid_to, tx_from, tx_to, "
            "content FROM memories"
        ).fetchall()
        self.about = con.execute(
            "SELECT src, dst, tenant_id, valid_from, valid_to, tx_from, tx_to FROM edges_about"
        ).fetchall()
        self.relates = con.execute(
            "SELECT src, dst, tenant_id, valid_from, valid_to, tx_from, tx_to FROM edges_relates"
        ).fetchall()
        self.entities = con.execute("SELECT entity_id, tenant_id, name FROM entities").fetchall()

    def _ok(self, tenant, vf, vt, tf, tt) -> bool:
        return self.vis.admits(tenant_id=tenant, valid_from=vf, valid_to=vt, tx_from=tf, tx_to=tt)

    def visible_memories(self) -> list[tuple]:
        return [r for r in self.memories if self._ok(r[1], r[3], r[4], r[5], r[6])]

    def frontier(self, seed: int, hops: int) -> set[int]:
        adj: dict[int, set[int]] = {}
        for src, dst, tenant, vf, vt, tf, tt in self.relates:
            if self._ok(tenant, vf, vt, tf, tt):
                adj.setdefault(src, set()).add(dst)
                adj.setdefault(dst, set()).add(src)
        seen = {seed}
        frontier = deque([(seed, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth == hops:
                continue
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, depth + 1))
        return seen

    def recall_2hop(self, seed: int, hops: int = 2, limit: int = 20) -> list[int]:
        reach = self.frontier(seed, hops)
        about_src = {
            src
            for src, dst, tenant, vf, vt, tf, tt in self.about
            if dst in reach and self._ok(tenant, vf, vt, tf, tt)
        }
        rows = [r for r in self.visible_memories() if r[0] in about_src]
        rows.sort(key=lambda r: (r[2], r[0]), reverse=True)
        return [int(r[0]) for r in rows[:limit]]

    def entities_of(self, memory_id: int) -> list[int]:
        return sorted(
            dst
            for src, dst, tenant, vf, vt, tf, tt in self.about
            if src == memory_id and self._ok(tenant, vf, vt, tf, tt)
        )


@pytest.fixture(params=EXPAND_PATHS)
def canary_db(request, tmp_path, dense_ids):
    path = tmp_path / f"canary-{request.param}.anatid"
    kwargs = {}
    if request.param == "extension":
        kwargs = {
            "use_csr_extension": True,
            "extension_path": _extension(),
            "require_extension": True,
        }
    db = Anatid.open(path, tenant=1, embedding_dim=DIM, **kwargs)
    try:
        facts = {t: _populate(db, t) for t in (1, 2)}
        db.rebuild_fts_index(now=T1 + 4 * MINUTE)
        if request.param == "extension":
            info = db.build_csr()
            assert info is not None and info.tenants == 2, db.csr.describe()
            assert db.expand_path == "extension"
        yield db, facts, request.param
    finally:
        db.close()


SCOPES = {
    "current": None,
    "t0+30m": T0 + 30 * MINUTE,  # before any T1 change
    "t1+90s": T1 + 90 * _dt.timedelta(seconds=1),  # after the supersede, before E's memory
    "valid-only-t0": AsOf(valid_time=T0 + 30 * MINUTE, tx_time=None),
    "tx-only-t0": AsOf(valid_time=None, tx_time=T0 + 30 * MINUTE),
}


def _suffix(tenant: int) -> str:
    return f"-t{tenant}"


def test_every_public_read_verb_respects_tenant_and_time(canary_db):
    """Two tenants, several instants, every read verb, both expansion paths, one oracle."""
    db, facts, path = canary_db
    checked = 0
    for tenant, (scope_name, scope) in itertools.product((1, 2), SCOPES.items()):
        vis = db.visibility(tenant=tenant, as_of=scope)
        oracle = _Oracle(db, vis)
        ents = facts[tenant]["entities"]
        mems = facts[tenant]["memories"]
        other = 2 if tenant == 1 else 1
        s = _suffix(tenant)

        def admits(row, *, tenant=tenant, vis=vis) -> bool:
            return row.tenant_id == tenant and (vis.is_current or vis.admits(row))

        # which path the frontier takes: the extension serves only current-state reads
        _sql, _p, used = frontier_sql(
            tenant, ents["A"], 2, as_of=AsOf.coerce(scope), backend=db.csr
        )
        assert used == ("extension" if (path == "extension" and scope is None) else "sql")

        # recall_2hop_ids / recall_2hop / context / the as_of view, against the oracle
        for seed_name, hops in (("A", 2), ("A", 1), ("A", 0), ("A", 3), ("B", 2), ("E", 2)):
            seed = ents[seed_name]
            expected = oracle.recall_2hop(seed, hops=hops, limit=50)
            got = db.recall_2hop_ids(seed, tenant=tenant, as_of=scope, hops=hops, limit=50)
            assert [m for m, _ in got] == expected, (tenant, scope_name, seed_name, hops)
            rows = db.recall_2hop(seed, tenant=tenant, as_of=scope, hops=hops, limit=50)
            assert [m.memory_id for m in rows] == expected
            # the hydrated row is the VERSION visible under the scope, not the live row
            assert all(admits(m) for m in rows)
            assert all(s in m.content for m in rows)
            ctx = db.context(seed, tenant=tenant, as_of=scope, hops=hops, limit=50)
            assert [m.memory_id for m in ctx] == expected
            if scope is not None:
                view = db.as_of(scope)
                assert [
                    m.memory_id for m in view.recall_2hop(seed, tenant=tenant, hops=hops, limit=50)
                ] == expected
            checked += 1

        # hybrid recall: every arm, every hit admitted, every about-name from this tenant
        hits = db.recall(
            "likes things",
            embedding=vec(1, 0, 0),
            seed_entity=ents["A"],
            tenant=tenant,
            as_of=scope,
            k=20,
            on_stale_fts="ignore",
        )
        assert set(hits.arms) == {"vector", "text", "graph"}
        assert hits, (tenant, scope_name)
        for h in hits:
            assert admits(h.memory), (tenant, scope_name, h.memory)
            assert s in h.memory.content
            assert all(name.endswith(s) for name in h.about), h.about
        visible_ids = {r[0] for r in oracle.visible_memories()}
        assert {h.memory_id for h in hits} <= visible_ids
        assert visible_ids <= {h.memory_id for h in hits}  # k=20 covers the whole tenant

        # get(): the shared id resolves to this tenant's row, and as-of hides the invisible
        shared = db.get(SHARED_ID, tenant=tenant, as_of=scope)
        assert shared is not None and shared.tenant_id == tenant and s in shared.content
        for mid in mems.values():
            m = db.get(mid, tenant=tenant, as_of=scope)
            if m is not None:
                assert admits(m)
            if scope is not None:
                assert (m is not None) == (mid in visible_ids), (tenant, scope_name, mid)
        assert db.get(mems["gone"], tenant=tenant, as_of=scope) is None
        # the other tenant's ids do not resolve here (except the shared one, which is ours)
        for mid in facts[other]["memories"].values():
            if mid != SHARED_ID:
                got_other = db.get(mid, tenant=tenant, as_of=scope)
                assert got_other is None or got_other.tenant_id == tenant

        # entities_of / get_entity / get_episode
        for mid in (mems["A"], mems["B"], mems["B2"], mems["C"], mems["E"], SHARED_ID):
            got_e = db.entities_of(mid, tenant=tenant, as_of=scope)
            assert [e.entity_id for e in got_e] == oracle.entities_of(mid)
            assert all(e.tenant_id == tenant and e.name.endswith(s) for e in got_e)
        assert db.get_entity("A" + s, tenant=tenant).tenant_id == tenant
        assert db.get_entity("A" + _suffix(other), tenant=tenant) is None
        assert db.get_entity(ents["A"], tenant=tenant).name == "A" + s
        ep = db.get(mems["A"], tenant=tenant).episode_id
        assert db.get_episode(ep, tenant=tenant).content.endswith(s)
        other_ep = db.get(facts[other]["memories"]["A"], tenant=other).episode_id
        assert db.get_episode(other_ep, tenant=tenant) is None

        # provenance: the chain never leaves the tenant
        prov = db.provenance(mems["B2"], tenant=tenant)
        assert [m.memory_id for m in prov.chain] == [mems["B2"], mems["B"]]
        assert all(m.tenant_id == tenant for m in prov.chain)
        assert all(e.tenant_id == tenant for e in prov.episodes)
        with pytest.raises(NotFoundError):
            db.provenance(facts[other]["memories"]["E"], tenant=tenant)

        # stats and prune only see this tenant's current rows
        if scope is None:
            current = {r[0] for r in oracle.visible_memories()}
            assert db.stats(tenant=tenant)["current_memories"] == len(current)
            report = db.prune(older_than=T1 + HOUR, tenant=tenant)
            assert set(report.memory_ids) == current
            mine = [r for r in oracle.memories if r[1] == tenant]
            # memories = logical rows (live versions); memory_versions = physical rows
            assert db.stats(tenant=tenant)["memories"] == len([r for r in mine if r[6] is None])
            assert db.stats(tenant=tenant)["memory_versions"] == len(mine)
            assert len(mine) > len([r for r in mine if r[6] is None])  # corrections happened
    assert checked == 2 * len(SCOPES) * 6


def test_scan_helper_sees_through_fstrings_and_skips_docstrings():
    """The scanner itself: a predicate split around an interpolation is still caught, and a
    docstring that explains the rule is not."""
    src = '''
"""Module doc: valid_to IS NULL is the current-state rule."""
def f(alias):
    """valid_to IS NULL again, in a docstring."""
    return f"SELECT x FROM t WHERE {alias}.valid_to IS NULL AND {alias}.tx_to IS NULL"
g = "SELECT y FROM t WHERE tenant_id = ?"
h = "DELETE FROM t WHERE tenant_id = ?"
prose = "Current rows are `valid_to IS NULL AND tx_to IS NULL`; add WHERE tenant_id = ? yourself."
fragment = " AND m.valid_to IS NULL AND m.tx_to IS NULL"
constant = "valid_to IS NULL AND tx_to IS NULL"
'''
    literals = _string_literals(src)
    texts = [t for _ln, t in literals]
    assert not any("Module doc" in t for t in texts)
    assert not any("again, in a docstring" in t for t in texts)
    assert any(TEMPORAL_FRAGMENTS[0].search(t) for t in texts)
    reads = [t for t in texts if READ_START.match(t) and TENANT_FRAGMENT.search(t)]
    assert reads == ["SELECT y FROM t WHERE tenant_id = ?"]
    # the SQL classifier: statements, clause fragments and bare predicates are SQL; a sentence
    # that quotes the rule is not, so help text cannot trip the scan while a copy of the rule
    # assigned to a constant still does
    flagged = [t for t in texts if _is_sql(t) and TEMPORAL_FRAGMENTS[0].search(t)]
    assert flagged == [
        "SELECT x FROM t WHERE {}.valid_to IS NULL AND {}.tx_to IS NULL",
        " AND m.valid_to IS NULL AND m.tx_to IS NULL",
        "valid_to IS NULL AND tx_to IS NULL",
    ]
    assert not _is_sql("Current rows are `valid_to IS NULL AND tx_to IS NULL`.")
    assert not _is_sql("READ-ONLY SQL over the anatid database, for questions the verbs do not")
