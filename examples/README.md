<p><a href="../README.md"><img src="../assets/brand/anatid-logo.png" alt="anatid" width="140" height="48"></a></p>

# Examples

Every script runs from the repository root with the package installed (`pip install -e .`).
Most scripts that talk to a model read `OPEN_ROUTER_KEY` from the environment or `open_router_key=`
from a `.env` file in the repository root; none of them prints the key. The medical-history module
uses explicit embedding endpoint flags and does not read `.env`.

| example | what it shows | needs a key |
| --- | --- | --- |
| [`quickstart.py`](quickstart.py) | The verbs in one file: `remember`, `relate`, `recall`, `supersede`, `as_of`, `provenance`, `forget`. Finishes in under a second. | no |
| [`procedural_graph.py`](procedural_graph.py) | Directed guidance for what to do next, a validated procedure repair, rejection memory and historical replay. Scripted solver and refiner; [design and limits](../docs/procedural-graphs.md). | no |
| [`procedural_studio/`](procedural_studio/README.md) | Fictional Cedar manufacturing lot review: reconcile passing reports with withdrawals, configuration and open nonconformances; inspect repairs and history. Optional live OpenRouter guidance; offline HTML export. | no; `--live` enables OpenRouter |
| [`medical_history/`](medical_history/README.md) | A fully synthetic infusion-pump development history: revisions, configuration applicability, retrospective corrections and evidence gaps. Compare raw retrieval, SQL traceability and anatid with a shared source-only interpreter. | no; semantic embeddings are opt-in |
| [`ingest_notes.py`](ingest_notes.py) | Three project notes ingested as prose. Each becomes a reviewed memory patch applied in one transaction; then `recall_2hop`, `provenance` and `as_of` show the owner changed, why, and what was believed before. Offline by default with a scripted extractor. | no; `--live` uses GLM 5.3 Flash through OpenRouter |
| [`dinner_party.py`](dinner_party.py) | The flagship demonstration. Six months of household facts, a question that names no guest and no ingredient, and an answer two hops away. `--scenario oncall` runs the on-call story instead. Writes go through an approval gate. | yes (OpenRouter) |
| [`scenarios.py`](scenarios.py) | The data behind `dinner_party.py` and the studio: entities, edges, facts, the question, the correction. Import it; it has no `main`. | no |
| [`glm_openrouter_agent.py`](glm_openrouter_agent.py) | An engineering-team assistant with tool calls into anatid. The answer to the question is never in one stored sentence; the 2-hop walk finds it. Shows `supersede`, `as_of` and `provenance` through the tools. | yes (OpenRouter) |
| [`agent_with_memory.py`](agent_with_memory.py) | The OpenAI Agents SDK integration: memory tools, an `AnatidSession` transcript and parked run states in one file, with an approval interruption and a resume. Runs the SDK's `ScriptedModel` without a key. | no; `OPENAI_API_KEY` switches to a real model |
| [`server_demo.py`](server_demo.py) | Two writer processes and a reader against one file: the DuckDB lock in act 1, the server profile in act 2, the same workload embedded in act 3. About twenty seconds. | no |
| [`studio/`](studio/README.md) | A local web UI that runs `dinner_party.py` one step at a time and shows the database at each step. See its README. | optional |

## Notebooks

[`notebooks/`](notebooks/) holds the same material as executed Jupyter notebooks, with outputs
saved, so they read on GitHub without running. Each runs in seconds with no key; the cells that
talk to a model are guarded and say so.

| notebook | what it explains |
| --- | --- |
| [`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Evidence before belief, facts and entities, relations, recall, correction, time travel, provenance, forgetting, and the file as SQL. |
| [`02_ingest_notes.ipynb`](notebooks/02_ingest_notes.ipynb) | The ingestion pipeline step by step: the memory patch, propose and review, a correction that moves an edge, aliases and dedupe, and how to plug in a real extractor. |
| [`03_retrieval.ipynb`](notebooks/03_retrieval.ipynb) | Each recall arm on its own, where the graph arm's seeds come from, the fusion weights and why, the graph arm's ordering, and the knobs. |
| [`04_agents_and_mcp.ipynb`](notebooks/04_agents_and_mcp.ipynb) | The OpenAI Agents SDK tools with an approval interruption and a resume, and the MCP server driven in process, including reviewed ingestion over `ingest` and `apply_patch`. |

To rebuild them after an API change, run `jupyter execute --inplace examples/notebooks/*.ipynb`
in a venv with `anatid[agents,mcp,notebooks]` installed; the `notebooks` extra is the executor.

The medical-history module always creates a fresh run directory and never deletes earlier runs.
The procedural-graph example defaults to an in-memory database; `--db PATH` requires a new file.
The other database-writing scripts put their database beside themselves (`*.anatid`, ignored by git) and
delete it on the next run. That is the only file a script deletes: `ingest_notes.py --db PATH`
refuses a `PATH` that already exists unless `--reset` says to delete it first.
