<p><a href="README.md"><img src="assets/brand/anatid-logo.png" alt="anatid" width="140" height="48"></a></p>

# Contributing to anatid

anatid is an embedded graph database for AI agents, built on DuckDB. It is MIT licensed and
developed in the open. Bug reports, benchmarks that contradict ours, and pull requests are all
welcome.

## Getting set up

```bash
git clone https://github.com/thedatasense/anatid
cd anatid
uv venv                                    # or: python -m venv .venv
uv pip install -e ".[dev]"                 # duckdb + pytest + both integrations
pytest                                     # the whole suite, in-memory, seconds
python examples/quickstart.py              # end-to-end smoke test, no API key needed
```

Python 3.10 through 3.13 are supported and all four are exercised in CI on Linux and macOS.
The only required runtime dependency is `duckdb>=1.5`. The optional extras are `agents`
(`openai-agents`) and `mcp` (`mcp`); `dev` installs `pytest` together with both of those, because
`tests/test_mcp.py` and `tests/test_openai_agents.py` `importorskip` their SDK at module level
and would otherwise vanish from the run without saying so. `tests/test_extension.py` still skips
unless you build the C++ extension (`cd ext && GEN=ninja make release`), and the `slow`/`oracle`
tests skip unless `spike/data/small` exists; everything else runs everywhere, CI included. Nothing
in `src/anatid` may import an optional dependency at module import time; integrations import theirs
lazily so that `import anatid` keeps working without them.

Some tests are marked `slow` (they load the 100k-memory spike dataset) or `oracle` (they check
results against the Phase 0 reference implementation). Skip them with
`pytest -m "not slow and not oracle"` while iterating.

## What we care about in a change

- Performance changes come with before/after numbers and the command that produced them. An
  expected speedup without a measurement behind it is not enough.
- Do not claim behavior the engine does not support. DuckDB has no `AS OF SYSTEM TIME`, no row- or
  schema-level access control, and no incremental full-text index. Our docstrings say so, and they
  must keep saying so. Where anatid oversells DuckDB, that is a bug and we want the report.
- A test that cannot pass is reported as failing, with its error text. Do not skip it, weaken it,
  or delete it.
- Correctness comes before speed on the recall path. `spike/bench/common.py` holds `reference_r1`,
  a pure-Python oracle for 2-hop recall, and any change to the traversal must still return
  byte-identical id lists against it.
- Docstrings state the contract, including staleness windows, isolation levels, and ceilings.

## Repository layout

```
src/anatid/            the library (schema, verbs, recall, csr, ids, types, errors)
src/anatid/integrations/  OpenAI Agents SDK session + tools, MCP server
ext/                   the optional C++ DuckDB extension (anatid_build_csr / graph_expand)
tests/                 pytest suite
examples/              runnable examples, no API keys
docs/                  architecture, benchmarks, roadmap
spike/                 Phase 0 evidence: benchmark contract, runners, results (READ ONLY)
```

`ext/` is a DuckDB extension built from DuckDB's own extension template and is compiled against a
pinned DuckDB version. Nothing in `src/anatid` requires it: the SQL graph path returns identical
rows, and the extension is opt-in (`Anatid.open(use_csr_extension=True)`).

`spike/` is frozen. It is the evidence behind the engine decision, and its results are only
meaningful if nobody edits them after the fact. If you want to re-run it, copy it out or add a
new runner; do not change what is there.

## Third-party notices

anatid is MIT licensed (see `LICENSE`), `Copyright (c) 2026 anatid contributors`. anatid stands
on two other MIT-licensed projects and may carry ideas or code from either. When it does, the
original notice travels with the code:

- DuckDB, Copyright Stichting DuckDB Foundation, MIT. anatid is built on DuckDB and links
  against it. `ext/` and `spike/extension/` are built from DuckDB's extension template and keep the
  template's own `LICENSE` file (`Copyright 2018-2025 Stichting DuckDB Foundation`) in place. Any
  other file in anatid that contains code adapted from the DuckDB source tree, such as SQL adapted
  from DuckDB's `fts` extension, carries DuckDB's copyright line and MIT notice at the top of the
  file, in addition to anatid's own.
- Kuzu, Copyright 2022-2025 Kùzu Inc., MIT. Kuzu was archived on 2025-10-10. Where we port an
  idea whose expression comes from Kuzu's source (a query formulation, a data-layout trick, a test
  corpus), the file carries `Copyright 2022-2025 Kùzu Inc.` and Kuzu's MIT notice alongside ours.
  The same applies to code taken from LadybugDB, Kuzu's maintained MIT fork.

Both licenses are MIT, so the combination stays MIT and no relicensing question arises. Attribution
is the part we are strict about: if you copy or closely adapt code from another project into a
pull request, say so in the PR description and put the notice in the file. If you cannot establish
the provenance of code you are contributing, do not contribute it.

By submitting a pull request you agree that your contribution is licensed under the MIT License
and that you have the right to license it that way. There is no CLA.

## Pull requests

- Branch from `main`, one logical change per PR.
- `pytest` passes locally on at least one platform before you open it; CI runs the matrix.
- Say what you measured. If you touched recall, show latency numbers; if you touched storage,
  show file sizes.
- Public API additions come with a docstring that states the contract and its limits, and an
  entry in `docs/roadmap.md` if they change what a milestone means.

## Releasing

The release workflow uses PyPI trusted publishing, so no API token is required in GitHub.

One-time setup at <https://pypi.org/manage/account/publishing/>: add a pending publisher with
owner `thedatasense`, repository `anatid`, workflow `release.yml`, environment `release`.

Then for each release:

1. Bump `version` in `pyproject.toml` and the two `__version__` strings (`src/anatid/__init__.py`,
   `src/anatid/database.py`). Update the manifest version, package version, and runtime pin in
   `mcp-registry/server.json` and `ANATID_EXT_VERSION` in `ext/src/anatid_extension.cpp` to match.
2. Run `python -m pytest tests/`. All of it must pass.
3. Tag and push: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
4. Publish a GitHub Release for that tag. The `release` workflow builds, runs `twine check`,
   smoke-tests the wheel in a clean virtualenv, and only then uploads.

Use the workflow's manual `workflow_dispatch` run with target `testpypi` to rehearse first.

The README is also the PyPI description. Keep its image and file links absolute so they work
on both sites, and include README changes in a new release to update PyPI. See
[brand assets](docs/branding.md) for the shared logo, favicon, and GitHub sharing card.
