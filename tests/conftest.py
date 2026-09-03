"""pytest fixtures for the anatid core tests.

Two kinds of database are on offer:

* :func:`db` / :func:`file_db` -- tiny in-memory or on-disk databases with an 8-dimensional
  embedding, for verb-level tests.
* :func:`spike_db` -- the Phase 0 spike's 100k-memory dataset loaded into a real anatid database,
  session-scoped because loading it costs a few seconds.  Used by the oracle test, which checks
  anatid's 2-hop recall against ``spike/bench/common.py``'s pure-numpy reference implementation.

The spike tree is READ ONLY.  ``spike/bench/common.py`` is imported by path so nothing there is
touched, and its Parquet files are read, never written.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPIKE_DIR = REPO_ROOT / "spike"
SPIKE_COMMON = SPIKE_DIR / "bench" / "common.py"
SPIKE_SMALL = SPIKE_DIR / "data" / "small"
SPIKE_EXTENSION = (SPIKE_DIR / "extension" / "build" / "release" / "extension" / "anatid"
                   / "anatid.duckdb_extension")
#: What ``cd ext && GEN=ninja make release`` produces.  Named here as well as being found by
#: :func:`anatid.csr.discover_extension_path`, because that function walks up from
#: ``anatid.__file__`` and finds nothing from site-packages: a suite run against an INSTALLED
#: wheel would otherwise skip every extension test, which is exactly the configuration a release
#: is validated in.
BUILT_EXTENSION = (REPO_ROOT / "ext" / "build" / "release" / "extension" / "anatid"
                   / "anatid.duckdb_extension")

#: Fixed clock so tests never depend on wall time.
T0 = _dt.datetime(2026, 1, 1, 0, 0, 0)

DIM = 8


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the 100k-memory spike dataset")
    config.addinivalue_line("markers", "oracle: checked against the spike reference implementation")


@pytest.fixture(scope="session")
def spike_common():
    """``spike/bench/common.py`` imported by path (the spike tree is never modified)."""
    if not SPIKE_COMMON.is_file():
        pytest.skip(f"spike harness not present at {SPIKE_COMMON}")
    if not (SPIKE_SMALL / "memories.parquet").is_file():
        pytest.skip(f"spike small dataset not present at {SPIKE_SMALL}")
    spec = importlib.util.spec_from_file_location("anatid_spike_common", SPIKE_COMMON)
    module = importlib.util.module_from_spec(spec)
    sys.modules["anatid_spike_common"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def db():
    """A fresh in-memory anatid database, tenant 1, FLOAT[8] embeddings."""
    from anatid import Anatid

    with Anatid.open(":memory:", tenant=1, embedding_dim=DIM) as handle:
        yield handle


@pytest.fixture
def file_db(tmp_path):
    """A fresh on-disk anatid database (needed for multi-connection / conflict tests)."""
    from anatid import Anatid

    with Anatid.open(tmp_path / "core.anatid", tenant=1, embedding_dim=DIM) as handle:
        yield handle


@pytest.fixture
def legacy_db():
    """A database with 0.1.1's accelerators and none of the derived ones.

    ``Anatid.open(accelerators=False)``.  Two kinds of test open this rather than :func:`db`:
    tests of the 0.1.1 half itself (the non-incremental file-wide BM25 index, whose whole
    subject is that a write is invisible until a rebuild), and tests of the derived-index
    framework, which build their own indexes over the placeholder registry and would otherwise
    be counting the shipped accelerators' journal rows as well as their own.
    """
    from anatid import Anatid

    with Anatid.open(":memory:", tenant=1, embedding_dim=DIM, accelerators=False) as handle:
        yield handle


@pytest.fixture
def legacy_file_db(tmp_path):
    """:func:`legacy_db` on disk, for multi-connection tests."""
    from anatid import Anatid

    with Anatid.open(
        tmp_path / "core.anatid", tenant=1, embedding_dim=DIM, accelerators=False
    ) as handle:
        yield handle


@pytest.fixture(scope="session")
def spike_db(tmp_path_factory, spike_common):
    """The spike's ``small`` dataset (100k memories, 196k ABOUT edges, 27.8k RELATES_TO edges)
    bulk-loaded into a real anatid database with the benchmarked clustering."""
    from anatid import Anatid

    path = tmp_path_factory.mktemp("spike") / "oracle.anatid"
    handle = Anatid.open(path, tenant=0, embedding_dim=64)
    counts = handle.load_parquet(SPIKE_SMALL, rebuild_fts=False, build_csr=False)
    assert counts.get("memories") == 100_000, counts
    try:
        yield handle
    finally:
        handle.close()


@pytest.fixture(scope="session")
def spike_queries(spike_common):
    """The 2,000 benchmark queries (index == query_id)."""
    return spike_common.load_queries("small")


def vec(*values: float) -> list[float]:
    """Pad/truncate to :data:`DIM` so tests can write short vectors."""
    out = list(values) + [0.0] * DIM
    return [float(x) for x in out[:DIM]]
