<p><a href="../../README.md"><img src="../../assets/brand/anatid-logo.png" alt="anatid" width="140" height="48"></a></p>

# Cedar manufacturing: procedural studio

A visual demonstration of procedural memory for a **fictional infusion-pump manufacturing
program**. A passing final-test report appears sufficient until a linked correction withdraws
it. The repair adds reconciliation of the lot, configuration, work-instruction revision,
traveler, withdrawals, and open nonconformances before the packet advances.

All records, rules, dates, and outcomes are invented. These are packet-review recommendations
for an authorized human, never device release decisions or safety determinations. The demo
does not implement regulatory requirements or clinical acceptance thresholds.

## Run it

From the repository root with Anatid installed:

```bash
python -m examples.procedural_studio
```

Open **http://127.0.0.1:8766**. Use `--port` for a different port. The offline replay requires
no extra packages, model, key, CDN, or web font.

To enable a real model call through OpenRouter:

```bash
export OPEN_ROUTER_KEY=your-key
python -m examples.procedural_studio --live
# Or choose a specific model instead of OpenRouter's automatic router:
python -m examples.procedural_studio --live --model z-ai/glm-5.3-flash
```

`OPENROUTER_API_KEY` also works. The server can read `open_router_key=` or `openrouter_api_key=`
from the repository's `.env`, following the other examples' convention. The key remains on the
server; it is never embedded in the page, exported, logged, or sent back to the browser.
`--live` fails at startup when no key is available. Offline mode does not load a key.

## The walkthrough

1. **The shortcut:** run the original procedure on `CED-2409`. It advances the packet based
   on `FT-2409`, overlooking withdrawal `COR-2409` and the absence of an executed retest.
2. **The repair:** validate the added reconciliation step. The same scripted solver now routes
   the packet to a hold, with the withdrawal and retest-plan evidence cited.
3. **The guardrail:** propose bypassing that check again. Validation rejects the regression,
   while Anatid retains the failed proposal and the working graph.
4. **Original snapshot:** replay the earlier procedure through Anatid's temporal read.
5. **Ask OpenRouter:** on the repaired graph, advance three steps to **Read**, then ask the
   model for next-step guidance. The live panel shows its recommendation, citations, resolved
   model, latency, token usage, and cost when the provider reports it.

Select graph nodes to inspect their local guidance. Select a different packet to try a complete
packet, a wrong configuration/instruction, a withdrawn pass, an open nonconformance, or a plan
without a completed result. Expand the provenance and model-context panels to inspect the evidence.

## Offline replay and live guidance are distinct

At startup, `data.py` runs the procedure against a fresh in-memory Anatid database. It stores
synthetic manufacturing records with raw episodes and explicit references, reads them back,
commits the validated rule repair, records the rejected shortcut, and reads historical graph
and provenance data. The presentation replays these checkpoints. Clicking a chapter does not
write to an existing database.

The scripted evaluation uses five validation cases and five separate test cases, plus two
training cases. The original procedure passes **1/5** test cases and the repaired procedure
**5/5**. Cases share five constructed patterns; these numbers demonstrate mechanics, not
LLM performance or generalization. The interpreter reads source fields, not expected answers.
A complete-packet case prevents “always hold” from passing.

The optional live call is the paper's **guidance-model slot**, not an autonomous manufacturing
agent or refiner. It receives the task, simulation policy, last completed procedure, recent
completed actions, outgoing two-hop graph, and records observed after Search. It receives no
gold answers, scripted outcomes, validation traces, or scores. Before Search, it receives no
records. Selecting a node in the inspector does not change the agent's active step.

The model may disagree with the graph: guidance remains soft. Output does not execute a tool,
release a lot, alter a rule, or change the scripted scores. Citation validation checks that IDs
were observed, not whether every generated claim is entailed. Changing the replay step or
scenario discards stale live output.

## OpenRouter integration

`POST /api/guide` accepts a known `case_id`, `checkpoint`, and `step`. The server reconstructs
context from its own records and calls the fixed OpenRouter chat-completions endpoint with
`model=openrouter/auto` by default, JSON output, minimal reasoning, a 900-token completion cap,
and a 60-second timeout. There are no automatic retries. One provider request runs at a time.
Requests require the local page's per-server token and matching Origin when present. The server
binds to localhost and rejects other Host headers.

The browser displays an error for transport failures, invalid JSON, missing fields, unknown
actions, or unobserved citations. Raw provider error bodies and credentials are not returned.
Provider usage is displayed when supplied. Only synthetic demo data goes to OpenRouter;
provider charges may apply when the button is used.

The integration follows OpenRouter's [chat API](https://openrouter.ai/docs/api_reference/overview),
[automatic model routing](https://openrouter.ai/docs/guides/routing/routers/auto-router), and
[reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## Export and test

```bash
python -m examples.procedural_studio --export /tmp/cedar-manufacturing.html
python -m pytest tests/test_procedural_studio.py tests/test_manufacturing_router.py tests/test_procedural_graph.py
```

The export embeds all replay data and assets. It is offline: the live button is disabled, no
key or live request token is exported, and `--live --export` is refused. Existing export files
are never overwritten. The UI supports keyboard controls, reduced motion, and horizontally
scrollable graphs on narrow screens.

`manufacturing.py` owns synthetic records and the packet interpreter; `data.py` handles Anatid
writes and replay data; `router.py` handles source-only context and OpenRouter transport;
`__main__.py` serves the demo. The original generic
[`procedural_graph.py`](../procedural_graph.py) remains usable on its own.
