# anatid studio

A local web UI that runs `examples/glm_openrouter_agent.py` one step at a time and shows what
the database does at each step. It is a teaching aid for the example, not a product. Everything
runs on your machine against a single file, `studio.anatid`, beside the server.

## Run it

```
pip install -r examples/studio/requirements.txt
export OPEN_ROUTER_KEY=sk-or-...        # optional, see below
python examples/studio/server.py
```

Open http://127.0.0.1:8765. The server binds to 127.0.0.1 only. Set `STUDIO_PORT` to use another
port. If you prefer uvicorn's CLI, run it from the repository root with
`uvicorn examples.studio.server:app --host 127.0.0.1 --port 8765 --no-access-log`; the
`--no-access-log` flag keeps query strings out of the terminal.

The first start creates the database and writes the day-one briefing. Step 1 in the UI deletes
and rebuilds it at any time.

## The model key

Steps 2, 3 (the "ask again" part) and 4 call GLM 5.3 Flash through OpenRouter. The key is read
from the environment variable `OPEN_ROUTER_KEY` (or `OPENROUTER_API_KEY`), or from a `.env` file
in this directory or any parent directory containing a line `open_router_key=sk-or-...`. That is
the same lookup the example uses. The key stays in process memory. It is never written to disk,
returned by any endpoint, or printed.

Set `STUDIO_NO_MODEL=1` to start without a model even when a key is available.

Without a key the status strip says so, the model steps return a short message instead of running,
and steps 1, 3 (the handover itself), 5 and 6 work as normal. Those four steps need no model.

## What each step demonstrates

1. Brief the assistant. Resets the database and writes four `RELATES_TO` edges and five facts, each
   attached to the entities it is about, all with writer `onboarding` and one episode. The graph
   panel shows the result: entities as circles, facts as squares.

2. Ask the two-hop question. "Ada's project is paging. Who should I wake up, and why?" No stored
   sentence contains both Ada and paging, and Bo is not in the question. The model calls `recall`
   with `seed_entity="Ada"`. The graph arm walks Ada to Project Kestrel to ingest-service to Bo,
   and the recall arms table shows which arm found each row and at what rank. The graph panel
   animates the walk hop by hop. The answer names Bo.

3. The world changes. `supersede` replaces the Bo fact with one naming Cy, sets `valid_to` on the
   old row, and records a `SUPERSEDES` edge. The old square in the graph turns hollow. Asking again
   returns Cy, because recall only returns current rows.

4. The model proposes a write. The user asks the assistant to remember a deployment rule. The model
   calls `remember`; the server parks the call and shows an approval card. Approve stores the fact
   with writer `glm-5.3-flash` and rebuilds the fts index; decline sends the model a refusal. Either
   way the model then finishes its turn.

5. Time travel. `as_of(t)` filters on both valid time and transaction time. Day one still returns
   Bo as the maintainer; after handover and now return Cy. The buttons re-run `recall_2hop` for
   ingest-service and for Ada at each instant. No model is involved.

6. Provenance. `provenance(id)` walks the `SUPERSEDES` chain from the current maintainer fact back
   to its first version, listing each version with its episode text and writer. No model is involved.

The status strip reports the anatid and schema versions, row counts, whether a key is configured,
and which recall arms are active. The vector arm is inactive because the demo stores no embeddings;
only the text and graph arms run.

## Endpoints

All responses are JSON. Ids are 64-bit integers and are sent as decimal strings. Timestamps are
ISO-8601 UTC with a `Z` suffix.

| Method and path | Purpose |
| --- | --- |
| `GET /` | the UI |
| `GET /api/health` | version, schema, whether a key is configured |
| `POST /api/reset` | delete and rebuild the database; returns the state |
| `GET /api/state` | entities, edges, memories, stats, timestamps, key status |
| `POST /api/ask` `{question}` | run the tool loop; returns steps, answer and any pending write |
| `POST /api/approve` `{id, approved}` | resume the loop after a decision |
| `POST /api/supersede` | apply the scripted handover |
| `GET /api/recall?seed=&q=` | raw arms view plus the 2-hop frontier from the seed |
| `GET /api/asof?t=` | `recall_2hop` for ingest-service and Ada as of `t` |
| `GET /api/provenance/{id}` | version chain, episodes, writers |

`/api/ask` returns 409 with a plain message when no key is configured or a write is still waiting
for a decision.

## Files

- `server.py`: the FastAPI backend, one file
- `index.html`: the UI, one file; it loads d3 v7 from cdnjs for the force layout and falls back to a
  fixed layout when offline
- `requirements.txt`, `.gitignore`
