# Answer-quality corpus

This directory holds the data for the answer-quality benchmark: does an agent answer better
from anatid than from the obvious alternatives, at the same model, the same prompt and the same
context budget? The Phase 0 benchmark (`docs/benchmarks.md`) measures storage speed and
structural correctness. It says nothing about answers. This corpus is what the answer
measurement runs on; the measurement itself, its systems and its results are in
[`docs/quality.md`](../../docs/quality.md), and `python -m bench.quality.run` reproduces them.

Everything here is generated from a seeded simulation, so every gold answer is right by
construction and every number is reproducible from one command:

```
python -m bench.quality.gen_corpus --verify
```

The command writes `data/notes.jsonl`, `data/gold_patches.jsonl`, `data/questions.jsonl`,
`data/counts.json` and `data/world.json`, then (with `--verify`) ingests the gold patches into
an in-memory anatid database through `anatid.ingest.ScriptedExtractor` and checks that the
graph agrees with the simulation. Seed `20260905` is the default; `--seed N` produces a
different organisation with the same shape.

## The world

`world.py` simulates a small engineering organisation from 2025-01-06 to 2026-06-26, with
questions asked as of 2026-06-30.

| what | how many | what happens to it |
| --- | ---: | --- |
| services | 12 | each changes owning team one to three times (22 handovers in all) |
| people | 8 | three move between teams during the period |
| teams | 5 | own services, carry a pager, gain and lose members |
| on-call | 5 pagers | each multi-member team rotates every four to eight weeks (35 rotations) |
| dependencies | 9 | one is rewired to another service, one is added later; four services have none |
| deploy windows | 8 | three change once; four services have no window |
| other constraints | 7 | version pins, approval rules, two deploy freezes |
| decisions | 17 | each with a stated reason; six come out of incident reviews |
| incidents | 6 | each review records a root cause and a decision |
| observations | 30 | a team notices a metric on a service it does not own |
| filler | 18 | planning days, practices, floors, who runs which review |
| reminders | 20 | a note restates something already recorded and still true |
| wrong records | 3 | a handover note names the wrong team; a correction note fixes it one to four weeks later |

Every event has a date and a source. Standup notes are labelled `standup/<team>/<date>`,
handovers `handover/<date>-<service>` (corrections add `-correction`), incident reviews
`incident-review/<INC id>`. `data/world.json` lists the full histories: who owned what when,
who was on call when, who was on which team, and the intervals during which the record was
wrong.

Two ownership histories are kept. `owner` is what was true; `recorded_owner` is what the notes
said. They differ only inside the three wrong-record intervals. Gold patches follow the record,
because an extractor can only extract what a note says. Question gold follows the truth, and
temporal questions never ask about a date inside a wrong-record interval.

## The notes (`data/notes.jsonl`)

178 notes in arrival order, mean 16 words, longest 54. One line per note:

| field | meaning |
| --- | --- |
| `note_id`, `seq` | `n001` onwards, arrival order |
| `date` | ISO date of the note; the day its facts take effect |
| `source`, `source_kind` | the label above, and `standup`, `handover` or `incident_review` |
| `team`, `author` | the team whose note it is and a member who wrote it |
| `text` | the note body as written |
| `rendered` | `[date] source: text`, the string every system should receive |
| `event_ids`, `event_kinds` | the simulation events the note carries (164 notes carry one, 13 carry two, 1 carries three) |

Phrasing varies: a handover is written seven different ways, a service is called by three
names ("the ledger", "the ledger service", "Ledger"; "billing-api", "the billing API", "the
billing service"), a team by three ("Atlas", "the Atlas team", "team Atlas"), a person by first
name or full name. Dates inside the body are relative ("as of today", "this cycle"); the
absolute date is in the header, which is why `rendered` and not `text` is the fair input. The
same string goes to every system: appended to the Markdown file, indexed by BM25, embedded,
and handed to `anatid.ingest.ingest` with `now=` the note's date, `source=` and `writer=` the
note's source label, so that anatid's memories carry a validity interval and a writer that
answer provenance questions the same way a header does for the others.

## The gold patches (`data/gold_patches.jsonl`)

One line per note, in the same order: `{"note_id", "source", "date", "patch"}` where `patch` is
`anatid.ingest.MemoryPatch.to_dict()` with `source_text` equal to the note's `rendered` string,
so the S5 oracle stores the same episode text S4 ingests. In all: 143 facts, 64 corrections, 110 relations opened,
61 closed. Facts use canonical sentences (`Atlas owns the ledger`, `Priya is on call for Atlas`,
`Tomasz is a member of Dune`, `Billing-api depends on the ledger`, `The ledger deploys only on
Tuesdays and Thursdays between 10:00 and 12:00 UTC`) with canonical entity names (`ledger`,
`Atlas`, `Priya`), so a fact that changes later is corrected by `old_text` and the pipeline
resolves it to the current memory. Relations use `owns`, `member_of`, `on_call_for` and
`depends_on`. Decisions have kind `decision`, constraints `constraint`, observations
`observation`.

This is the input of the S5 oracle: `ScriptedExtractor([MemoryPatch.from_dict(p["patch"])
for p in patches])`, applied through `ingest` in note order with the same `now`, `source` and
`writer` as S4. It separates extraction quality from retrieval quality; it is not the product.

`--verify` checks the gold end to end: no correction is downgraded to a new fact, the only
dedupe drops are the facts restated by the 20 reminder notes (which must be dropped), the final graph holds exactly
the world's final owners, dependencies, windows, on-call and memberships, every temporal
question is answered by `db.as_of(date).context(entity)`, and every non-abstention question
names at least one existing supporting note.

## The questions (`data/questions.jsonl`)

150 questions, 25 per category, chosen from 313 candidates with a round-robin over subtypes so
no subtype dominates. One line per question:

| field | meaning |
| --- | --- |
| `qid`, `category`, `subtype` | `q001` onwards; the six categories below; a finer label for analysis |
| `question` | the text every system is asked |
| `gold` | the exact answer; `I don't know` for abstention |
| `aliases` | other strings the judge should accept (team and service forms, full names, date formats, short window forms) |
| `support` | note ids whose facts are needed; empty for abstention |
| `as_of` | for temporal questions that name a date, that date |
| `hops`, `chain` | for multi-hop questions, the number of hops and the intermediate facts |
| `distractors` | values that were once true and are now wrong (an earlier owner, the previous window); a judge or an analysis can count "answered with the stale value" separately |
| `rubric` | what the judge should accept for this category |

| category | count | what it asks | example |
| --- | ---: | --- | --- |
| `single_fact` | 25 | a fact stated once and never changed | What was the root cause of INC-2026-003? |
| `knowledge_update` | 25 | the current value of something that changed | Which team owns search-indexer now? |
| `temporal` | 25 | the value at a date, the value before a change, or when a change happened | Which team owned billing-api on 2025-04-14? |
| `multi_hop` | 25 | two or three facts chained, across notes that share no keywords | Who is on call for the team that owns the service webhooks depends on? |
| `provenance` | 25 | which note said it, and on what date | Which note recorded the handover of search-indexer to Atlas, and on what date? |
| `abstention` | 25 | plausible, unanswerable from the notes | Which service does the event-bus depend on? |

Subtypes, from `data/counts.json`:

- single_fact: approval_rule 2, decision_reason 3, dependency 2, deploy_freeze 2,
  deploy_window 2, filler 2, incident_cause 3, incident_date 2, team_membership 2,
  version_pin 2, wrong_record 3
- knowledge_update: owner_now 12, oncall_now 5, team_now 3, window_now 3, dependency_now 2
- temporal: owner_at 5, oncall_at 5, owner_before 4, handover_date 4, team_before 3,
  window_before 3, dependency_before 1
- multi_hop: oncall_of_owner 5, owner_of_dependency 5, oncall_of_dependency_owner 4,
  window_of_dependency 4, services_of_team 4, oncall_of_persons_team 3
- provenance: handover_note 4, decision_note 4, dependency_first_note 4, review_date 4,
  correction_note 3, move_note 3, window_first_note 3
- abstention: no_manager_recorded 3, no_previous_team 3, no_reason_recorded 3,
  owner_before_first 3, unknown_incident 3, no_dependants 2, no_dependency 2,
  no_secondary_recorded 2, no_slo_recorded 2, no_window 2

Abstention questions are built from gaps the world has on purpose: services with no deploy
window or no dependency, services nothing depends on, handovers whose notes give no reason,
people whose earlier team was never recorded, incident ids that never occurred, and things
an engineering notes stream does not record (managers, availability targets, secondary
on-call).

## Reading the results

The corpus is about 6,200 tokens with headers (24,808 characters at 4 characters per token),
so a 1,200-token context budget holds roughly a fifth of it, or the most recent 35 notes for
the Markdown system. That budget is what makes the comparison a comparison: the unbudgeted
Markdown variant, which receives everything, is the upper bound of "just put it in the
prompt" and is expected to do well here. It would not at 10x the corpus.

Things the corpus is built to expose, and where to look for each:

- Stale answers. Every `knowledge_update` and `temporal` question lists its distractors. A
  recency-ordered Markdown window gets the current owner right when the handover is recent and
  wrong when it is not; BM25 and vectors rank the old and the new handover notes alike.
- Correction handling. Three services had a wrong owner on record for one to four weeks. The
  `wrong_record` questions ask for the wrong team; the `owner_now` questions for those
  services list it as a distractor.
- Reasoning across notes. `multi_hop` chains share no keyword with the final answer: the
  question names a service, the answer is a person, and the notes in between are a handover
  and an on-call rotation written months apart.
- Knowing when to stop. Abstention is a quarter of the set. A system that always answers
  scores zero there; a system that abstains too readily loses everywhere else. Both numbers
  belong in the report.

## Limitations

The corpus is synthetic and generated by us. Its phrasing comes from a few dozen templates,
its entities are few, and its notes are short and clean. Real notes are longer, messier and
less consistent about names. The judge and the answerer are the same model family. n is 150
questions and 178 notes, so differences of a few questions between systems are noise. Nothing
here says how a system behaves at ten thousand notes; that is what `docs/benchmarks.md`
measures, for storage rather than for answers.

## Regenerating

```
python -m bench.quality.gen_corpus            # write data/
python -m bench.quality.gen_corpus --verify   # and check the gold against anatid
python -m bench.quality.gen_corpus --seed 7 --out /tmp/q7   # a different organisation
```

Changing anything in `world.py` (a template pool, a count, a date range) changes the RNG
stream and therefore the whole corpus; the question ids are not stable across such edits.
Commit the regenerated `data/` with the change.
