"""The demo scenarios must leave the graph saying one thing rather than two.

``examples/scenarios.py`` is the flagship demonstration, and its second act is a belief
being corrected: Priya turns out to react to prawns and not pine nuts, Cy takes the
ingest-service over from Bo.  A correction that rewrites the sentence and leaves the
contradicting edge open teaches the opposite of what the database is for, because the
graph is what recall walks.  Before this file, the dinner story ended with Priya reacting
to both pine nuts and prawns, and the on-call story with two current maintainers.

So these tests read the graph back out after the correction and check it against the
story: exactly one current reacts_to edge from Priya, to prawns; exactly one current
maintains edge on the ingest-service, from Cy.  The last test forces the correction to
fail part-way and checks that nothing moved, because a correction that half-lands is the
same inconsistency arriving by a different route.

The examples are not a package.  They are loaded by path here and registered under the
names they import each other by, so ``examples/dinner_party.py``'s ``import scenarios``
finds the same module object these tests hold.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from anatid import Anatid
from anatid.visibility import current_row_sql

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    path = EXAMPLES / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{path} is not in this checkout")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scenarios():
    return _load("scenarios")


@pytest.fixture(scope="module")
def dinner_party():
    _load("scenarios")  # dinner_party imports it by that name
    return _load("dinner_party")


# --------------------------------------------------------------------------------------
# Reading the graph back out.  "Current" is both columns null: closing an edge stamps
# tx_to on the version being corrected and inserts a successor carrying valid_to, so a
# query that tests only valid_to keeps returning the corrected version for ever.
# --------------------------------------------------------------------------------------


def current_edges(db) -> list[tuple[str, str, str]]:
    """Every relates edge the database still believes, as (source, target, rel_kind)."""
    rows = db.execute(
        f"SELECT s.name, d.name, r.rel_kind FROM edges_relates r "
        f"JOIN entities s ON s.entity_id = r.src AND {current_row_sql('s')} "
        f"JOIN entities d ON d.entity_id = r.dst AND {current_row_sql('d')} "
        f"WHERE {current_row_sql('r')} ORDER BY 1, 2, 3"
    ).fetchall()
    return [(src, dst, kind) for src, dst, kind in rows]


def current_memories(db) -> list[str]:
    rows = db.execute(f"SELECT content FROM memories WHERE {current_row_sql()}").fetchall()
    return sorted(row[0] for row in rows)


def edge_versions(db, src: str, dst: str) -> list[tuple]:
    """Every version of every edge between two named entities, oldest first."""
    return db.execute(
        "SELECT r.rel_kind, r.valid_from, r.valid_to, r.tx_from, r.tx_to "
        "FROM edges_relates r JOIN entities s ON s.entity_id = r.src "
        "JOIN entities d ON d.entity_id = r.dst "
        "WHERE s.name = ? AND d.name = ? ORDER BY r.tx_from, r.version",
        [src, dst],
    ).fetchall()


# --------------------------------------------------------------------------------------
# The two stories, checked against what a reader is told happened.
# --------------------------------------------------------------------------------------


def test_dinner_correction_leaves_priya_reacting_to_prawns_and_nothing_else(db, scenarios):
    scenario = scenarios.SCENARIOS["dinner"]
    scenarios.build(db, scenario)
    assert ("Priya", "pine nuts", "reacts_to") in current_edges(db)

    scenarios.apply_supersede(db, scenario)

    reactions = [
        edge for edge in current_edges(db) if edge[0] == "Priya" and edge[2] == "reacts_to"
    ]
    assert reactions == [("Priya", "prawns", "reacts_to")]
    # Maya's allergy is untouched: the correction closes one edge, not a rel_kind.
    assert ("Maya", "prawns", "reacts_to") in current_edges(db)


def test_oncall_correction_leaves_cy_as_the_only_maintainer(db, scenarios):
    scenario = scenarios.SCENARIOS["oncall"]
    scenarios.build(db, scenario)
    assert ("Bo", "ingest-service", "maintains") in current_edges(db)

    scenarios.apply_supersede(db, scenario)

    maintainers = [
        edge for edge in current_edges(db) if edge[1] == "ingest-service" and edge[2] == "maintains"
    ]
    assert maintainers == [("Cy", "ingest-service", "maintains")]


@pytest.mark.parametrize("key", ["dinner", "oncall"])
def test_the_correction_moves_every_edge_the_scenario_says_it_moves(db, scenarios, key):
    scenario = scenarios.SCENARIOS[key]
    scenarios.build(db, scenario)
    change = scenario.supersede
    assert change.removed_relations, f"{key} corrects a sentence and no edge"

    correction = scenarios.apply_supersede(db, scenario)

    after = current_edges(db)
    for edge in change.removed_relations:
        assert edge not in after
    for edge in change.extra_relations:
        assert edge in after
    assert correction.closed_relations == change.removed_relations
    assert correction.opened_relations == tuple(change.extra_relations)
    assert sorted(after) == sorted(scenarios.current_relations(scenario))


@pytest.mark.parametrize("key", ["dinner", "oncall"])
def test_the_edge_the_correction_closes_is_closed_and_not_deleted(db, scenarios, key):
    scenario = scenarios.SCENARIOS[key]
    scenarios.build(db, scenario)
    src, dst, _ = scenario.supersede.removed_relations[0]

    scenarios.apply_supersede(db, scenario)

    versions = edge_versions(db, src, dst)
    assert len(versions) == 2, versions
    corrected, successor = versions
    # The version that was believed until the correction: closed on the transaction axis,
    # so a read as of an earlier transaction time still sees the edge open-ended.
    assert corrected[2] is None and corrected[4] is not None
    # Its successor is live and carries the instant belief ended, which is the correction's.
    assert successor[2] == scenarios.parse_time(scenario.supersede.when)
    assert successor[4] is None


@pytest.mark.parametrize("key", ["dinner", "oncall"])
def test_a_correction_that_fails_part_way_changes_nothing(db, scenarios, key, monkeypatch):
    """supersede, unrelate and relate are one transaction or none of it.

    The failure is forced at the last step, after the memory has been superseded and the
    old edge closed, which is the state that would leave the graph saying nothing where
    it used to say something.
    """
    scenario = scenarios.SCENARIOS[key]
    scenarios.build(db, scenario)
    old = scenarios.find_fact(db, scenario)
    edges_before = current_edges(db)
    memories_before = current_memories(db)

    def boom(*args, **kwargs):
        raise RuntimeError("the edge write failed")

    monkeypatch.setattr(Anatid, "relate", boom)
    with pytest.raises(RuntimeError, match="the edge write failed"):
        scenarios.apply_supersede(db, scenario)

    assert current_edges(db) == edges_before
    assert current_memories(db) == memories_before
    assert scenario.supersede.new_content not in memories_before
    reread = db.get(old.memory_id)
    assert reread is not None and reread.is_current
    for name, _ in scenario.supersede.extra_entities:
        assert db.get_entity(name) is None, f"{name} survived a rolled-back correction"

    # Nothing was consumed by the failure: the same correction still applies cleanly.
    monkeypatch.undo()
    scenarios.apply_supersede(db, scenario)
    after = current_edges(db)
    for edge in scenario.supersede.removed_relations:
        assert edge not in after
    for edge in scenario.supersede.extra_relations:
        assert edge in after


def test_the_demo_script_prints_only_edges_the_database_still_believes(db, scenarios, dinner_party):
    """read_edges is what the script prints its paths from, so it has to agree with recall.

    It filtered on valid_to alone, which is the corrected version's own column: the closed
    edge came back for ever and the demo printed a graph contradicting its own sentence.
    """
    scenario = scenarios.SCENARIOS["dinner"]
    scenarios.build(db, scenario)
    scenarios.apply_supersede(db, scenario)

    printed = dinner_party.read_edges(db)
    assert ("Priya", "pine nuts", "reacts_to") not in printed
    assert ("Priya", "prawns", "reacts_to") in printed
    assert sorted(printed) == sorted(current_edges(db))


def test_a_read_before_the_correction_still_walks_the_edge_it_closed(db, scenarios):
    """Closing an edge is a correction, not a deletion, and the demo says so in step 4.

    The pesto card is two hops from Friday dinner only through Priya's pine nut allergy.
    A read as of the March instant the demo replays still crosses that edge; today's read
    does not, because the correction closed it in August.
    """
    scenario = scenarios.SCENARIOS["dinner"]
    scenarios.build(db, scenario)
    scenarios.apply_supersede(db, scenario)
    card = "The pesto is basil, pine nuts, parmesan and olive oil."

    march = scenarios.parse_time(scenario.asof_time)
    then = [m.content for m in db.as_of(march).recall_2hop(scenario.seed_entity, limit=40)]
    now = [m.content for m in db.recall_2hop(scenario.seed_entity, limit=40)]

    assert card in then
    assert card not in now
    # The card itself was never closed: what changed is the edge on the way to it.
    live = [
        h.memory for h in db.recall(card, k=10, on_stale_fts="ignore") if h.memory.content == card
    ]
    assert len(live) == 1 and live[0].is_current
