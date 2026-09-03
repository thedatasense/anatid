# anatid studio

A local web UI that runs `examples/dinner_party.py` one step at a time and shows what the
database does at each step. It is a teaching aid for the example, not a product. Everything runs
on your machine against a single file, `studio.anatid`, beside the server.

The default story is the dinner party. Someone is cooking on Friday and asks whether there is a
problem with the menu. The question names two dishes and a day. It names no guest and no
ingredient, and no stored sentence contains both "pesto" and "Priya". The answer lives in the edges
between sentences, and one guest carries an adrenaline pen.

The database holds twenty-six facts by the time the question is asked. Six of them carry the
question. The rest is household noise: the oven, the boiler, the car, who walks home, which bread is
better. `recall` returns eight. The demonstration is which eight.

The second story is the on-call rotation the studio shipped with first: a page arrives for Ada's
project, and the person to wake is Bo, two hops away. Pick either from the Story menu in the
header. Both come from the same `Scenario` definition in `examples/scenarios.py`, so every
question, every button label and every explanation on screen follows whichever story is loaded.

## Run it

```
pip install -r examples/studio/requirements.txt
export OPEN_ROUTER_KEY=sk-or-...        # optional, see below
python examples/studio/server.py
```

Open http://127.0.0.1:8765. The server binds to 127.0.0.1 only. Set `STUDIO_PORT` to use another
port. It runs the same from the repository root and from `examples/studio` itself; it puts its
parent directory on `sys.path` so that `import scenarios` works either way. If you prefer
uvicorn's CLI, run it from the repository root with
`uvicorn examples.studio.server:app --host 127.0.0.1 --port 8765 --no-access-log`; the
`--no-access-log` flag keeps query strings out of the terminal.

The first start creates the database and writes the default story. Step 1 in the UI rebuilds it at
any time, and the Story menu rebuilds it with the other scenario.

## The model key

Steps 2, 3 (the "ask again" part) and 4 call GLM 5.3 Flash through OpenRouter. The key is read
from the environment variable `OPEN_ROUTER_KEY` (or `OPENROUTER_API_KEY`), or from a `.env` file
in this directory or any parent directory containing a line `open_router_key=sk-or-...`. That is
the same lookup the example uses. The key stays in process memory. It is never written to disk,
returned by any endpoint, or printed.

Set `STUDIO_NO_MODEL=1` to start without a model even when a key is available.

Without a key the status strip says so, the model steps return a short message instead of running,
and steps 1, 3 (the change itself), 5 and 6 work as normal. Those four steps need no model.

## What each step demonstrates

1. Write the memory. Resets the database and writes the whole story: the entities, the
   `RELATES_TO` edges between them, and the facts, each attached to the entities it is about, each
   with the writer who told it and the day they told it. In the dinner story that is twenty-six
   facts spread over eleven months of ordinary household talk, written at the dates it happened, so
   the time travel later is real. The graph panel shows the result: entities as circles, facts as
   squares. Entities with no edge, and the facts that are only about them, are left out of the
   picture so the shape stays legible; the note under the legend says how many.

2. Ask the two-hop question. "I'm making pesto pasta and pavlova for Friday. Any problems?" The
   model calls `recall` with a `seed_entity`. The graph arm walks Friday dinner to Priya to pine
   nuts to pesto pasta and returns the fact that joins them, which shares no words with the
   question. The recall arms table shows which arm found each row and at what rank, how many of the
   stored facts came back, and which rows carry a graph rank with no text rank. The row that carried
   the answer is one of those. The graph panel animates the walk hop by hop.

3. The world changes. `supersede` writes the new fact, sets `valid_to` on the old row, and records
   a `SUPERSEDES` edge between the two versions. Priya's allergist clears her for pine nuts and she
   reacts to prawns instead. The old square in the graph turns hollow. Asking again about a menu
   with a prawn starter gets a different answer, because recall returns only current rows.

4. The model proposes a write. The user asks the assistant to remember something new. The model
   calls `remember`; the server parks the call and shows an approval card. Approve stores the fact
   with writer `glm-5.3-flash` and rebuilds the fts index; decline sends the model a refusal. Either
   way the model then finishes its turn. Every write goes through the gate. Reads run straight
   through.

5. Time travel. `as_of(t)` filters on both valid time and transaction time. A read at 15 March 2026
   still returns the pine nut warning, because that is what the database believed on 12 March when
   the Sunday risotto was planned without pine nuts. That note is still on record today, next to a
   belief that has since closed. The buttons re-run `recall_2hop` for two seeds at each instant, and
   the row that took part in the change is marked. No model is involved.

6. Provenance. `provenance(id)` walks the `SUPERSEDES` chain from a memory back to its first
   version, listing each version with its episode text and writer. The pine nut belief traces back
   to the message Sam sent on 26 December, with his name and the date on it. No model is involved.

The status strip reports the anatid and schema versions, row counts, whether a key is configured,
and which recall arms are active. The vector arm is inactive because the demo stores no embeddings;
only the text and graph arms run.

## Endpoints

All responses are JSON. Ids are 64-bit integers and are sent as decimal strings. Timestamps are
ISO-8601 UTC with a `Z` suffix.

| Method and path | Purpose |
| --- | --- |
| `GET /` | the UI |
| `GET /api/health` | version, schema, whether a key is configured, the active scenario |
| `POST /api/reset` `{scenario}` | delete and rebuild the database; returns the state. The body is optional and keeps the current story when omitted |
| `GET /api/state` | the active scenario, entities, edges, memories, stats, timestamps, key status |
| `POST /api/ask` `{question}` | run the tool loop; returns steps, answer and any pending write. With no body it asks the scenario's own question |
| `POST /api/approve` `{id, approved}` | resume the loop after a decision |
| `POST /api/supersede` | apply the scenario's scripted change |
| `GET /api/recall?seed=&q=&k=` | raw arms view, the corpus size, and the 2-hop frontier from the seed. `k` defaults to the same 8 the model's own tool spends |
| `GET /api/asof?t=` | `recall_2hop` for the scenario's two seeds as of `t` |
| `GET /api/provenance/{id}` | version chain, episodes, writers |

`/api/ask` returns 409 with a plain message when no key is configured or a write is still waiting
for a decision. `/api/supersede` returns 409 once the change has already been applied.
`/api/reset` returns 400 for a scenario name that does not exist.

`/api/state` carries a `scenario` object with the story's key, title, one-liner, question, followup
question, write request, seed entity, the `explain` texts the steps display, the labels for the
three as_of instants, and the list of stories to offer in the picker. Everything the UI says about
the story comes from there. The two instants themselves are `past_ts` and `change_ts`; the older
names `day_one_ts` and `handover_ts` carry the same two values.

## Files

- `server.py`: the FastAPI backend, one file. It imports `scenarios.py` from the parent directory
- `index.html`: the UI, one file; it loads d3 v7 from cdnjs for the force layout and falls back to a
  fixed layout when offline
- `requirements.txt`, `.gitignore`

The stories themselves live in `examples/scenarios.py`, shared with `examples/dinner_party.py`.
Add one there and it appears in the Story menu, with no change here.
