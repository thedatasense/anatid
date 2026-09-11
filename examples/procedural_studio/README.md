<p><a href="../../README.md"><img src="../../assets/brand/anatid-logo.png" alt="anatid" width="140" height="48"></a></p>

# Medical device manufacturing: Cedar lot review

Cedar is a fictional infusion-pump program. Lot `CED-2409` has a passing final-test report, but
a linked correction withdraws it after a firmware error. The procedure must check that correction
before advancing the manufacturing packet to human review.

The records and review rules are invented. The demo produces advisory packet-routing outcomes.
A person retains authority over any device release.

## Run the example

Install anatid from the repository root with `pip install -e .`, then choose a view:

| View | Command |
| --- | --- |
| Interactive graph | `python -m examples.procedural_studio` |
| Terminal walkthrough | `python -m examples.manufacturing_review` |
| Interactive graph with model guidance | `python -m examples.procedural_studio -l` |
| Offline HyperText Markup Language (HTML) export | `python -m examples.procedural_studio -o /tmp/cedar-manufacturing.html` |

Open <http://127.0.0.1:8766> for the interactive graph. Use `-p` to choose another port.
The offline view uses local assets and needs no model credentials.

Live guidance calls a large language model (LLM) through OpenRouter. Set `OPENROUTER_API_KEY`
in the environment before starting the server. `OPEN_ROUTER_KEY` is also accepted. The server
can read `open_router_key` or `openrouter_api_key` from the repository's `.env` file.
The credential stays on the server. Offline mode does not load it.

The default model name is `openrouter/auto`. To request a particular model, add `-m` followed by
its OpenRouter model name. Live mode stops at startup if no key is configured.

## Follow the packet

| Step | Action | What to inspect |
| --- | --- | --- |
| Original procedure | Run the agent on lot CED-2409 | It advances the packet after reading the passing test |
| Repair | Select “Validate a repair” | Reconciliation finds correction COR-2409 and the missing retest result |
| Rejected edit | Select “Test a shortcut” | Validation rejects removal of the reconciliation step |
| Earlier revision | Select “Original snapshot” | The earlier route remains available after the repair |
| Model guidance | On the repaired graph, advance three steps to Read, then select “Ask OpenRouter” | The proposed next action, cited records, and request usage |

Select a node to inspect its guidance. The packet selector also includes a complete record set
and other evidence gaps, such as an open nonconformance or a configuration mismatch.
An assembly traveler records the work performed on the lot. In this example it appears within
the device history record (DHR). Final test (FT), correction (COR), and test plan (TP) prefixes
identify the other records. Additional cases use an engineering change order (ECO) or a
nonconformance report (NCR).

## What runs in the database

At startup, `data.py` creates an in-memory anatid database. It stores the manufacturing records
with their source text and references, then reads them back for the simulation. The accepted
repair changes a rule and its edge in one transaction. The rejected proposal is saved as memory.
The page replays these saved states, including the earlier graph and its evidence.

| Evaluation input | Count | Purpose |
| --- | --- | --- |
| Training cases | 2 | Supply traces for the proposed repair |
| Validation cases | 5 | Decide whether to retain the edit |
| Test cases | 5 | Check the retained graph on separate records |

The original procedure passes 1/5 test cases; the repaired procedure passes 5/5. The sets share
five designed failure patterns. These results describe the scripted evaluator and do not estimate
LLM accuracy. A complete-packet case prevents an “always hold” rule from passing.

## What the model receives

| Request content | Boundary |
| --- | --- |
| Task and example review policy | Reconstructed on the server for the selected packet |
| Last action and recent completed steps | Follows the actual replay position |
| Outgoing graph within two hops | Carries the conditions and guidance for nearby actions |
| Observed records | Available only after Search |
| Expected answers and evaluation scores | Omitted |

The guidance response can disagree with the graph. It cannot execute an action or change a rule.
Citations are checked against the observed record identifiers. That check does not establish that
every generated statement follows from the cited text. Changing the packet or replay position
discards stale model output.

## Request handling

The application programming interface (API) accepts `case_id`, `checkpoint`, and `step` at
`POST /api/guide`. The server reconstructs the request from known records and calls OpenRouter's
chat-completions endpoint. The response uses JavaScript Object Notation (JSON).

| Setting | Value |
| --- | --- |
| Model | `openrouter/auto`, unless a model is named at startup |
| Completion limit | 900 tokens |
| Timeout | 60 seconds |
| Concurrent requests | One |
| Automatic retries | None |
| Local request checks | Page token, local Host header, and matching Origin when present |

The browser reports transport failures or invalid model responses without exposing the raw
provider error. Only the fictional packet data is sent to OpenRouter. Usage and cost appear when
the provider supplies them. A click on “Ask OpenRouter” may incur a provider charge.

See OpenRouter's [chat interface](https://openrouter.ai/docs/api_reference/overview),
[automatic routing](https://openrouter.ai/docs/guides/routing/routers/auto-router), and
[reasoning settings](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## Export and verification

An export embeds the graph and its assets in one HTML file. The live button is disabled and no
credential or request token is included. Existing export files are preserved. Live mode and export
cannot be selected together. Keyboard controls and reduced-motion support are available; narrow
screens can scroll the graph horizontally.

Run the checks from the repository root:

```bash
python -m pytest tests/test_procedural_studio.py tests/test_manufacturing_router.py tests/test_procedural_graph.py
```

| File | Responsibility |
| --- | --- |
| `manufacturing.py` | Fictional records and packet evaluation |
| `data.py` | Database writes and replay data |
| `router.py` | Model context and OpenRouter request |
| `__main__.py` | Local server and HTML export |
| `../procedural_graph.py` | Shared graph store and revision operations |
