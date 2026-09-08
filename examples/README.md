# Examples

Every script runs from the repository root with the package installed (`pip install -e .`).
Most scripts that talk to a model read `OPEN_ROUTER_KEY` from the environment or `open_router_key=`
from a `.env` file in the repository root; none of them prints the key. The medical-history module
uses explicit embedding endpoint flags and does not read `.env`.

| example | what it shows | needs a key |
| --- | --- | --- |
| [`quickstart.py`](quickstart.py) | The verbs in one file: `remember`, `relate`, `recall`, `supersede`, `as_of`, `provenance`, `forget`. Finishes in under a second. | no |
| [`medical_history/`](medical_history/README.md) | A fully synthetic infusion-pump development history: revisions, configuration applicability, retrospective corrections and evidence gaps. Compare raw retrieval, SQL traceability and anatid with a shared source-only interpreter. | no; semantic embeddings are opt-in |
| [`ingest_notes.py`](ingest_notes.py) | Three project notes ingested as prose. Each becomes a reviewed memory patch applied in one transaction; then `recall_2hop`, `provenance` and `as_of` show the owner changed, why, and what was believed before. Offline by default with a scripted extractor. | no; `--live` uses GLM 5.3 Flash through OpenRouter |
| [`dinner_party.py`](dinner_party.py) | The flagship demonstration. Six months of household facts, a question that names no guest and no ingredient, and an answer two hops away. `--scenario oncall` runs the on-call story instead. Writes go through an approval gate. | yes (OpenRouter) |
| [`scenarios.py`](scenarios.py) | The data behind `dinner_party.py` and the studio: entities, edges, facts, the question, the correction. Import it; it has no `main`. | no |
| [`glm_openrouter_agent.py`](glm_openrouter_agent.py) | An engineering-team assistant with tool calls into anatid. The answer to the question is never in one stored sentence; the 2-hop walk finds it. Shows `supersede`, `as_of` and `provenance` through the tools. | yes (OpenRouter) |
| [`agent_with_memory.py`](agent_with_memory.py) | The OpenAI Agents SDK integration: memory tools, an `AnatidSession` transcript and parked run states in one file, with an approval interruption and a resume. Runs the SDK's `ScriptedModel` without a key. | no; `OPENAI_API_KEY` switches to a real model |
| [`server_demo.py`](server_demo.py) | Two writer processes and a reader against one file: the DuckDB lock in act 1, the server profile in act 2, the same workload embedded in act 3. About twenty seconds. | no |
| [`studio/`](studio/README.md) | A local web UI that runs `dinner_party.py` one step at a time and shows the database at each step. See its README. | optional |

The medical-history module always creates a fresh run directory and never deletes earlier runs.
The other database-writing scripts put their database beside themselves (`*.anatid`, ignored by git) and
delete it on the next run. That is the only file a script deletes: `ingest_notes.py --db PATH`
refuses a `PATH` that already exists unless `--reset` says to delete it first.
