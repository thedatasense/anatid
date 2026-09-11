<p><a href="../../README.md"><img src="../../assets/brand/anatid-logo.png" alt="anatid" width="140" height="48"></a></p>

# A fictional medical-device development history

A runnable evidence-retrieval experiment for **Cedar**, an entirely invented infusion-pump
program. It asks whether a product-history assistant can find the right revision, configuration,
decision and correction—not just a plausible-sounding test report.

No Medtronic information, patient data, clinical thresholds or proprietary documents are used.
This is a simulation of engineering records, **not a device simulator, validated medical tool,
regulatory assessment or release decision**. Original controlled records and authorized human
review remain authoritative. Missing evidence does not establish that a device is unsafe.

## Run it

From the repository root, with anatid installed:

```sh
python -m examples.medical_history
```

The default generates three seeded histories (`7 11 23`), each with 500 documents and 100
questions. It runs five retrieval strategies without a model, credentials or network access.
It prints the path to `REPORT.md`, including the sensor-change question and cited exclusions.

For a tiny, quick walkthrough:

```sh
python -m examples.medical_history --seeds 7 --documents 30 --questions 10
```

Use `--out /tmp/cedar-run-01` for an explicit output location. The directory must not already
exist; the example never deletes or overwrites previous runs. Without `--out`, it creates a fresh
temporary directory. It does not read your existing anatid database, repository notes or `.env`.

Each run contains:

- `REPORT.md`: aggregate results, question-family breakdown and a worked evidence-gap answer.
- `summary.json`: configuration, source fingerprints, metrics and elapsed time.
- `seed-N/documents.jsonl`: the complete synthetic source exports.
- `seed-N/questions.jsonl`: queries and the independent answer/evidence key.
- `seed-N/results.jsonl`: every system's answer, citations, retrieved record IDs, missing evidence
  and approximate context size. Join retrieved IDs to `documents.jsonl` to reconstruct contexts.
- `seed-N/history.anatid`: the actual graph, document embeddings and source provenance.

## The pressure-sensor change

The invented program requires an approved passing report for the **exact** requirement revision
and hardware/firmware configuration. This is a rule of the simulation, not a general regulatory
rule or a claim that real-world evidence reuse is never permissible.

| Record/event | Effective date | Known from | What it means in this scenario |
| --- | --- | --- | --- |
| `TR-88` | 2024-07-10 | 2024-07-12 | Passing evidence for revision A / HW-A / FW-1.0 only. |
| `ECO-23`, `R-44@B` | 2025-02-20 | 2025-02-20 | Sensor supplier change; new configuration and a recorded decision not to reuse the old report. |
| `TR-103` | 2025-03-10 | 2025-03-12 | Initially claims an approved pass for revision B / HW-B / FW-1.1. |
| `COR-12` | 2025-03-10 | 2025-04-01 | Wrong firmware image discovered; withdraws that claim retroactively. |
| `TP-104` | 2025-04-02 | 2025-04-02 | Replacement test planned. Approval of a plan is not a passing result. |
| `TR-104` | 2025-05-03 | 2025-05-05 | Replacement result becomes available. The first case passes; other seeded cases may fail or remain draft. |

There are also a draft passing report and an approved report for the wrong configuration.
The question on April 15 is:

> Which approved passing reports establish evidence for R-44 revision B on HW-B/FW-1.1?

The full-context answer is **not established by the available evidence**, with explicit reasons:
`TR-88` is for the old revision, `TR-103` was withdrawn, the other candidates are draft or
wrong-configuration, and `TP-104` is only a plan. `TR-104` is not available yet.

Looking at March 20 **with knowledge available on March 20** accepts the initial claim.
Looking at March 20 **with knowledge available on April 15** rejects it. Likewise, a result
effective May 3 cannot be used by a team whose knowledge cutoff is May 4 if it was recorded May 5.

Ten question families cover these date distinctions, evidence gaps, current coverage, change
impact, decision rationale, a simple owner lookup and an unanswerable requirement.

## What is actually compared

Every system gets identical full documents, including their structured fields and explicit
source references. Every document is embedded once and shared across all systems. Each query
uses one shared vector, the same product/date scope, candidate cap and context budget.

| System | Retrieval strategy |
| --- | --- |
| `raw-vector` | Cosine ranking of full source-document vectors, with product/date filtering. |
| `raw-hybrid` | Equal-weight reciprocal-rank fusion of those vectors and BM25; same filters. |
| `trace-sql` | Ordinary SQLite recursive traversal of exported references, two hops from the query's identifier; rank the neighborhood with the raw hybrid ranker. |
| `anatid-hybrid` | Existing `db.recall()` with its default fusion weights, the query's explicit seed and two hops. |
| `anatid-trace` | Application strategy using `db.recall_2hop()` to restrict the evidence neighborhood, then the same hybrid ranker as SQL. This is **not** default hybrid recall. |

The requirement or change identifier is an explicit query input, not secretly selected from
gold. A withdrawal need only reference its test report: it need not repeat the requirement ID.
Both SQL and anatid receive that link. SQL is included specifically to test whether anatid adds
value beyond a conventional traceability database.

Imports use `remember(..., episode=source, now=recorded_date, valid_from=effective_date)` and
`relate(..., valid_from=effective_date)`; historical reads use both axes of `AsOf`.
Source records are immutable. A withdrawal is another linked source record, not destructive
replacement of the original report. Applicability is interpreted explicitly in the example;
the graph engine does not decide what constitutes valid medical-device evidence.

No LLM extracts these links: they represent links already present in a requirements/test-system
export. No LLM answers or judges questions either. One deterministic interpreter reads **only
retrieved records**, applying the same documented rules to every system. The answer key comes
from the simulator's event schedule, before retrieval, and is unavailable to that interpreter.

This separation makes retrieval failures inspectable. It does **not** establish free-text
extraction accuracy, natural-language reasoning quality or independence from the template
author's assumptions.

### Read the scores carefully

`exact_answer` checks the answer value. `evidence_recall` measures retrieval of the scenario's
evidence checklist. `supported_correct` additionally requires the complete checklist to be
retrieved and cited. For coverage questions, that checklist includes the misleading reports
and their exclusions, not merely the smallest possible proof of a pass. An accidental
"not established" caused by retrieving no reports does not earn supported correctness.

`false_support` flags acceptance of any report that gold rejects. A missing withdrawal is a
particularly important failure: retrieving a report alone can produce a false positive. The
interpreter's `supported` status means only "supported by this retrieved context under the
simulation rules," never device safety or release readiness.

The default uses `HashEmbedder`: deterministic **word-overlap hashes, not semantic embeddings**.
Tokens are approximated by characters / 4, with whole-record prefix packing and a default budget
of 2,000. DuckDB runs single-threaded to stabilize floating-point ties; this is not a throughput
benchmark. These metrics must not be presented as a comparison against production semantic RAG.

The verified single-threaded three-seed offline run (500 documents / 100 questions per seed) found:

| System | Exact answer | Complete, cited checklist + correct answer |
| --- | ---: | ---: |
| raw-vector | 50.7% | 19.0% |
| raw-hybrid | 50.3% | 20.0% |
| trace-sql | 100.0% | 100.0% |
| anatid-hybrid | 41.0% | 21.7% |
| anatid-trace | 100.0% | 100.0% |

This shows the value of following explicit evidence links **in this constructed task**, not an
anatid-specific victory. Default hybrid recall can still omit the withdrawal while retrieving
the original passing report. The custom trace strategy succeeds, but SQL matches it. The tiny
30-document walkthrough is easy enough for all five strategies to achieve 100%.

### Use real embeddings, explicitly

An optional compatible embedding endpoint receives the generated synthetic records and queries
only. Configure your own endpoint, model and actual output dimension, then run:

```sh
python -m examples.medical_history \
  --embedding-url "$EMBEDDING_BASE_URL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-dim "$EMBEDDING_DIM" \
  --out /tmp/cedar-semantic-01
```

If authentication is needed, set `ANATID_EMBEDDING_KEY` in your environment, or name a different
variable with `--embedding-key-env`. Never pass a key in the URL. `--request-dimensions` is opt-in
because not every endpoint accepts that field. Your provider may charge for this mode; no live
embedding run is bundled or claimed here. Using semantic embeddings still leaves the shared
answerer deterministic: it does not turn this into an end-to-end LLM benchmark.

## Test whether information volume matters

Keep seeds, questions, embeddings and context budget fixed while changing document count:

```sh
python -m examples.medical_history --documents 500 --out /tmp/cedar-size-500
python -m examples.medical_history --documents 2000 --out /tmp/cedar-size-2000
python -m examples.medical_history --documents 5000 --out /tmp/cedar-size-5000
```

At fixed question count these are nested corpora: the same case records and questions plus more
background notes. This isolates distraction from additional information. It does **not** model
longer revision chains, denser cross-product dependencies or longer documents.

Each group of ten questions creates one 11-record case. Increasing `--questions` adds component
histories; at least `11 * ceil(questions / 10)` documents are required. Cases currently share one
event template, with seeded variations in outcome, owner, phrasing and background. Seeds are not
held-out real product programs, and questions from one case are correlated.

For a credible product-history evaluation, the next experiments should add independently written
histories, incomplete/incorrect links, permitted evidence reuse, conflicting reviews, deeper
revision chains and a reference-following RAG baseline. Freeze cases before tuning retrieval.
Measure citation validity, missed corrections, latency and storage alongside answer accuracy.
Any trial with actual employer records requires the employer's data-access and deployment
approval; this example neither requests nor uploads such data.

## Files and tests

`world.py` defines the independent event schedule, sources and gold; `systems.py` builds the
indexes; `evaluate.py` interprets retrieved evidence and scores it; `__main__.py` exports results.

```sh
python -m pytest tests/test_medical_history.py
```

Tests cover truth/source agreement, reproducible noise growth, withheld future knowledge, native
anatid time filtering, provenance, two-hop correction discovery, SQL/anatid parity, misleading
evidence, missing-support scoring, budgets and refusal to overwrite a run.

For public domain context, the FDA's [Generic Infusion Pump research](https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-software-safety-research-fda)
is a useful starting point for future independent scenarios. This example does not reproduce
that reference specification or claim to implement its safety requirements.
