<p><a href="../README.md"><img src="../assets/brand/anatid-logo.png" alt="anatid" width="140" height="48"></a></p>

[Documentation](README.md) · [Examples](../examples/README.md)

# Procedural graphs for medical device manufacturing

A procedure tells an assistant which check comes next. In the Cedar example, that check decides
whether a fictional infusion-pump lot has enough evidence to advance its packet to human review.
The first procedure misses a test withdrawal. A stored repair adds the missing reconciliation step.

The design follows [Lu et al., *Procedural Graphs*](https://arxiv.org/abs/2609.09153v1).
A large language model (LLM) can read the outgoing steps near the last action and offer guidance.
Proposed procedure changes are evaluated before they enter the retained graph. Anatid stores the
rules with their evidence and preserves earlier revisions.

## Run the Cedar example

Use either view from a source checkout with anatid installed:

| View | Command | What it shows |
| --- | --- | --- |
| Terminal | `python -m examples.manufacturing_review` | Records, routes, evaluation results, and retained rejection |
| Browser | `python -m examples.procedural_studio` | Interactive graph, record inspection, and revision playback |
| Browser with model guidance | `python -m examples.procedural_studio -l` | A real OpenRouter request after a chosen step |

Open <http://127.0.0.1:8766> for the browser view. Live guidance requires `OPENROUTER_API_KEY` in
the environment. The [visual guide](../examples/procedural_studio/README.md) explains the request
and offline export.

Cedar lot `CED-2409` uses hardware revision B and firmware 1.1, under work instruction 17 revision C.
Its packet contains these fictional records:

| Record | Evidence |
| --- | --- |
| Final test `FT-2409` | A passing functional-test result |
| Device history record `DHR-2409` | An approved assembly traveler |
| Correction `COR-2409` | Withdrawal of the test after the wrong firmware image was discovered |
| Test plan `TP-2409` | A scheduled replacement test, with no completed result |

The simulation requires a current passing report for the exact lot and configuration. It also
requires an approved traveler and no unresolved nonconformance. These are rules for the example.
A person retains authority over any device release.

## What changes

| Stage | Path through the records | Outcome |
| --- | --- | --- |
| Original procedure | Search → Read → Review | Advances the packet after reading the passing report |
| Repaired procedure | Search → Read → Reconcile → Hold | Finds the withdrawal and cites the missing retest result |
| Proposed shortcut | Remove reconciliation | Validation rejects the change and saves the failed proposal |
| Historical read | Load the earlier graph | Replays the original route after the repair has been stored |

Selected output from the terminal example:

```text
Original route: start -> search -> read -> review
Original outcome: ready_for_review
Repaired route: start -> search -> read -> reconcile -> hold
Repaired outcome: hold_for_review
Evidence: FT-2409, DHR-2409, COR-2409, TP-2409

Original procedure: 1/5 scripted test cases passed
Repaired procedure: 5/5 scripted test cases passed
Repair accepted: True
Later shortcut accepted: False
Rejected proposals retained: 1
```

Five validation cases decide whether to keep the edit. Five separate test cases measure the
retained graph. They cover a complete packet, a configuration mismatch, a withdrawn report,
an open nonconformance, and a plan without a result. A procedure that always holds fails the
complete-packet case.

The scores describe a scripted simulation. The case sets share the same small collection of
failure patterns, so the results provide no estimate of model accuracy on other tasks. Live
model advice is displayed separately and does not change these scores.

## How the graph fits anatid

| Stored item | Manufacturing example | Existing operation |
| --- | --- | --- |
| Procedure entity | `cedar_manufacturing/read` or `cedar_manufacturing/check` | `remember()` with a procedure entity kind |
| Directed edge | Read requires Reconcile | `relate()` |
| Rule attributes | Condition, guidance text, and possible mistakes | A memory attached to the edge endpoints |
| Source evidence | Runbook text or validation trace | Raw episode and `provenance()` |
| Accepted repair | Replace the direct review edge with a reconciliation edge | `correct()` within `transaction()` |
| Rejected proposal | Shortcut and failed validation retained together | `remember()` with a rejection kind |
| Earlier procedure | Original graph before the repair | A temporal read through `Visibility.at()` |

Rule attributes are stored as JavaScript Object Notation (JSON) in memories. Each rule has a
unique edge label. The read checks that the stored endpoints match the direction of the edge.
This gives each rule a source record and a correction history without changing the schema.

Direction matters. `recall_2hop()` searches relationships in either direction to find facts.
The procedure adapter follows outgoing edges to preserve action order. It loads this small
graph once per run, with tenant and time filters applied through Structured Query Language (SQL).
The example has no measurement for large procedure graphs.

## What the graph adds

| Property | Evidence in the demo |
| --- | --- |
| A required check stays attached to an action | Read leads to Reconcile before either terminal outcome |
| A changed rule has a reason | `provenance()` links the repair to its runbook and validation traces |
| A failed edit remains inspectable | The rejected shortcut is stored while the working graph stays active |
| Earlier behavior can be reproduced | A historical snapshot replays the direct route to Review |

The text-retrieval panel ranks rules for the query “review summary.” A ranked list can omit a
prerequisite that the outgoing graph includes. That example demonstrates preserved connections;
it supplies no measured accuracy comparison with a tuned text-retrieval system.

Before evaluation, the adapter rejects duplicate rule keys and empty fields. A terminal node
cannot have outgoing edges, and each reachable node must have a path to a terminal. Cycles are
allowed within the simulator's step limit.

## The OpenRouter call

At **Read**, choose **Ask OpenRouter**. The server sends the model the task and observed records,
along with the last completed action and its outgoing two-hop graph. Expected answers and
validation results are omitted. Before Search, the request contains no records.

The response names a suggested action and cites records already observed. The browser shows the
resolved model and request usage. Advice remains optional and never executes a tool or writes a
procedure change. The model can disagree with the stored route.

To evaluate a model-driven procedure, compare the same solver and tool set across these setups:

| Setup | Procedure context |
| --- | --- |
| Baseline | No stored procedure |
| Retrieved rules | A ranked list of relevant rules |
| Full graph | Every stored transition |
| Local graph | Outgoing steps within two hops of the current action |
| Edited graph | Local guidance after a validated change |

Reserve test tasks from refinement. Measure completed tasks and missed prerequisites, then report
request cost and latency. Repeat model runs to show how much the results vary. A refiner should
receive failed proposals as well as successful traces before suggesting another edit.

Run model evaluation outside the write transaction. Before committing an accepted edit, check
that the retained procedure still matches the evaluated revision. The example's cheap scripted
evaluation currently runs inside a transaction. A future procedure application programming
interface (API) needs a revision-conflict contract for concurrent refiners.

The reusable store remains in [`procedural_graph.py`](../examples/procedural_graph.py).
[`manufacturing.py`](../examples/procedural_studio/manufacturing.py) supplies the Cedar records
and evaluator used by the visual and terminal views.
