"""The answer-quality benchmark harness: what makes the comparison fair.

* the budget cut is one function, applied by token count, and identical for every system;
* the judge never learns which system produced an answer;
* the model cache makes a second run free, and an offline run refuses the network;
* abstention is scored as a refusal, and a refusal anywhere else is wrong;
* the runner saves every prompt, context and answer, and the report leads with its limits.

No network: every test drives :class:`bench.quality.llm.LLMClient` through a fake transport.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from pathlib import Path

import pytest

from bench.quality import harness as H
from bench.quality import judge as J
from bench.quality.llm import LLMClient, OfflineCacheMiss, TokenCounter, request_key
from bench.quality.report import render_report
from bench.quality.run import assemble_summary, score_runs

DIM = 16
SECRET = "SECRET-SYSTEM-NAME-9f3a"


# --------------------------------------------------------------------------- fakes


class FakeTransport:
    """Deterministic chat and embedding replies; counts every network call."""

    def __init__(self, reply: str = "I don't know") -> None:
        self.reply = reply
        self.calls = 0
        self.bodies: list[dict] = []

    def post(self, path: str, body: dict) -> dict:
        self.calls += 1
        self.bodies.append({"path": path, "body": json.loads(json.dumps(body))})
        if path == "/embeddings":
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            data = []
            for i, text in enumerate(inputs):
                vec = [0.0] * DIM
                for word in text.lower().split():
                    digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
                    vec[int.from_bytes(digest[:4], "big") % DIM] += 1.0
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                data.append({"index": i, "embedding": [x / norm for x in vec]})
            return {"data": data, "usage": {"prompt_tokens": 4 * len(inputs), "cost": 0.0}}
        reply = self.reply(body) if callable(self.reply) else self.reply
        finish = "stop"
        if isinstance(reply, dict):  # {"content": ..., "finish_reason": ...}
            finish = reply.get("finish_reason", "stop")
            reply = reply.get("content", "")
        return {
            "choices": [{"message": {"content": reply}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "cost": 0.00001},
        }


def make_llm(tmp_path: Path, transport: FakeTransport | None = None) -> LLMClient:
    return LLMClient(
        transport=transport or FakeTransport(),
        cache_dir=tmp_path / "cache",
        embed_dim=DIM,
        counter=TokenCounter(prefer_tiktoken=False),
    )


def note(i: int, text: str, day: int = 1) -> H.Note:
    date = _dt.date(2025, 1, day)
    source = f"standup/atlas/{date.isoformat()}"
    return H.Note.from_dict(
        {
            "note_id": f"n{i:03d}",
            "seq": i,
            "date": date.isoformat(),
            "source": source,
            "source_kind": "standup",
            "team": "Atlas",
            "author": "Priya",
            "text": text,
            "rendered": f"[{date.isoformat()}] {source}: {text}",
        }
    )


def question(qid: str, category: str, text: str, gold: str, **kw) -> H.Question:
    row = {"qid": qid, "category": category, "subtype": "t", "question": text, "gold": gold}
    row.update(kw)
    return H.Question.from_dict(row)


class FixedSystem(H.BaseSystem):
    """A system whose ranking is given; used to prove the cut is by tokens alone."""

    def __init__(self, llm: LLMClient, name: str, ranked: list[str]) -> None:
        super().__init__(llm)
        self.name = name
        self.ranked = ranked

    def ingest(self, notes) -> None:
        self.ingest_report = {"notes": len(notes)}

    def candidates(self, question: str) -> list[str]:
        return list(self.ranked)


# --------------------------------------------------------------------------- the budget


def test_fit_budget_is_a_prefix_cut_by_tokens(tmp_path):
    counter = TokenCounter(prefer_tiktoken=False)  # 4 chars per token, rounded up
    ranked = ["a" * 40, "b" * 40, "c" * 40, "d" * 40]  # 10 tokens each, plus newlines
    kept = H.fit_budget(ranked, 25, counter)
    assert kept == ["a" * 40, "b" * 40]
    assert H.context_tokens(kept, counter) <= 25
    assert H.context_tokens([*kept, "c" * 40], counter) > 25
    assert H.fit_budget(ranked, None, counter) == ranked
    assert H.fit_budget(["x" * 400, "y"], 10, counter) == []
    assert H.fit_budget(["", "  ", "z"], 10, counter) == ["z"]


def test_the_cut_is_identical_for_two_systems_with_the_same_ranking(tmp_path):
    llm = make_llm(tmp_path)
    ranked = [f"- note {i} " + "word " * (5 + i) for i in range(30)]
    one = FixedSystem(llm, "one", ranked)
    two = FixedSystem(llm, "two", ranked)
    for budget in (50, 120, 400):
        a = one.retrieve("q", budget)
        b = two.retrieve("q", budget)
        assert a == b
        assert a == ranked[: len(a)]
        assert H.context_tokens(a, llm.counter) <= budget
        if len(a) < len(ranked):
            assert H.context_tokens(ranked[: len(a) + 1], llm.counter) > budget


def test_markdown_keeps_the_most_recent_notes_in_reading_order(tmp_path):
    llm = make_llm(tmp_path)
    notes = [
        note(i, f"Note number {i} says something about the ledger service.", 1 + i)
        for i in range(1, 21)
    ]
    system = H.MarkdownSystem(llm)
    system.workdir = tmp_path / "md"
    system.ingest(notes)
    assert (tmp_path / "md" / "notes.md").read_text().count("\n- [") == 20
    kept = system.retrieve("anything", 120)
    assert 0 < len(kept) < 20
    assert kept == [f"- {n.rendered}" for n in notes[-len(kept) :]]  # the tail, oldest first
    whole = H.MarkdownSystem(llm, budgeted=False)
    whole.ingest(notes)
    assert whole.unbudgeted and len(whole.retrieve("anything", None)) == 20


def test_bm25_ranks_the_matching_note_first_and_keeps_digits(tmp_path):
    llm = make_llm(tmp_path)
    notes = [
        note(1, "The ledger service is owned by the Atlas team."),
        note(2, "INC-2025-011 was caused by a stuck consumer offset.", 2),
        note(3, "Boreal holds its planning meeting on Thursdays.", 3),
    ]
    system = H.Bm25System(llm)
    system.ingest(notes)
    top = system.retrieve("What was the root cause of INC-2025-011?", 1200)
    assert top and "INC-2025-011" in top[0]
    assert "ledger" in system.retrieve("Which team owns the ledger?", 1200)[0]
    system.close()


def test_vector_system_ranks_by_cosine_through_the_cached_embedder(tmp_path):
    transport = FakeTransport()
    llm = make_llm(tmp_path, transport)
    notes = [
        note(1, "The ledger service is owned by the Atlas team."),
        note(2, "Boreal holds its planning meeting on Thursdays.", 2),
    ]
    system = H.VectorSystem(llm)
    system.ingest(notes)
    assert "ledger" in system.retrieve("ledger service owned team", 1200)[0]
    assert "planning" in system.retrieve("planning meeting Thursdays", 1200)[0]
    first = transport.calls
    system.retrieve("ledger service owned team", 1200)
    assert transport.calls == first  # the question's vector came from the cache


# --------------------------------------------------------------------------- fused baselines


def test_fuse_rankings_is_reciprocal_rank_fusion_with_a_named_tie_break():
    a = ["x", "y", "z"]
    b = ["y", "w", "x"]
    # y scores 1/61 plus 1/62, x scores 1/61 plus 1/63, w 1/62 alone and z 1/63 alone
    assert H.fuse_rankings([a, b], k=60) == ["y", "x", "w", "z"]
    # an exact tie keeps the order of the tie-break ranking
    assert H.fuse_rankings([["p", "q"], ["q", "p"]], tie_break=["q", "p"]) == ["q", "p"]
    assert H.fuse_rankings([["p", "q"], ["q", "p"]], tie_break=["p", "q"]) == ["p", "q"]
    assert H.fuse_rankings([["p", "q"], ["q", "p"]]) == ["p", "q"]  # default: the first ranking
    # a line missing from the tie-break ranking sorts after those it holds
    assert H.fuse_rankings([["m"], ["n"]], tie_break=["n"]) == ["n", "m"]


def test_hybrid_fuses_the_two_raw_note_rankings_and_shares_the_cut(tmp_path):
    transport = FakeTransport()
    llm = make_llm(tmp_path, transport)
    notes = [
        note(1, "The ledger service is owned by the Atlas team.", 3),
        note(2, "INC-2025-011 was caused by a stuck consumer offset.", 4),
        note(3, "Boreal holds its planning meeting on Thursdays.", 5),
    ]
    system = H.HybridSystem(llm)
    system.ingest(notes)
    assert system.ingest_report["arms"] == ["bm25", "vector"]
    question = "Which team owns the ledger service?"
    fused = system.candidates(question)
    by_text = system.bm25.candidates(question)
    by_vector = system.vector.candidates(question)
    assert fused == H.fuse_rankings([by_text, by_vector], k=H.FUSION_K, tie_break=by_vector)
    assert "ledger" in fused[0]
    assert sorted(fused) == sorted(f"- {n.rendered}" for n in notes)  # cosine ranks every note
    assert system.last_retrieval["bm25_candidates"] < system.last_retrieval["vector_candidates"]
    kept = system.retrieve(question, 40)
    assert kept == fused[: len(kept)] and 0 < len(kept) < 3
    assert H.context_tokens(kept, llm.counter) <= 40
    system.close()


def test_vector_prf_ranks_a_second_time_with_the_top_notes_appended(tmp_path):
    transport = FakeTransport()
    llm = make_llm(tmp_path, transport)
    notes = [
        note(1, "The ledger service is owned by the Atlas team.", 3),
        note(2, "Atlas pager is held by Priya this cycle.", 4),
        note(3, "Boreal holds its planning meeting on Thursdays in the service room.", 5),
    ]
    system = H.VectorPrfSystem(llm, feedback=1)
    system.ingest(notes)
    plain = H.VectorSystem(llm)
    plain.ingest(notes)
    question = "ledger service owned team"
    before = transport.calls
    fused = system.candidates(question)
    assert transport.calls == before + 2  # the question, then the question with feedback
    embeds = [b["body"]["input"] for b in transport.bodies if b["path"] == "/embeddings"]
    expanded = embeds[-1] if isinstance(embeds[-1], str) else embeds[-1][0]
    assert expanded.startswith(question) and notes[0].rendered in expanded
    first = plain.candidates(question)
    second = plain.rank(expanded)
    assert fused == H.fuse_rankings([first, second], k=H.FUSION_K, tie_break=first)
    assert fused[0] == first[0] and "ledger" in fused[0]
    assert sorted(fused) == sorted(f"- {n.rendered}" for n in notes)
    assert system.last_retrieval == {"feedback_notes": 1}
    again = transport.calls
    assert system.candidates(question) == fused and transport.calls == again  # both cached


def test_external_fused_baselines_agree_with_the_built_in_rule(tmp_path):
    from bench.quality import systems as ext
    from bench.quality.systems import hybrid

    llm = make_llm(tmp_path)
    ext.set_token_counter(llm.counter.count, name=llm.counter.name)
    notes = [
        note(1, "The ledger service is owned by the Atlas team.", 3),
        note(2, "Atlas pager is held by Priya this cycle.", 4),
        note(3, "Boreal holds its planning meeting on Thursdays in the service room.", 5),
    ]
    converted = [ext.Note.from_dict(n.to_dict()) for n in notes]
    embedder = llm.embedder()
    question = "Which team owns the ledger service?"

    fused = hybrid.HybridSystem(embedder)
    fused.build(converted)
    builtin = H.HybridSystem(llm)
    builtin.ingest(notes)
    assert [h.line for h in fused.rank(question)] == builtin.candidates(question)
    result = fused.retrieve(question, budget=1200)
    assert result.meta["arms"] == ["bm25", "vector"] and result.tokens <= 1200

    prf = hybrid.VectorPrfSystem(embedder, feedback=1)
    prf.build(converted)
    builtin_prf = H.VectorPrfSystem(llm, feedback=1)
    builtin_prf.ingest(notes)
    assert [h.line for h in prf.rank(question)] == builtin_prf.candidates(question)
    assert prf.retrieve(question, budget=1200).meta["rounds"] == 2
    fused.close()
    builtin.close()


# --------------------------------------------------------------------------- anatid


def test_anatid_gold_store_renders_validity_writer_and_history(tmp_path):
    from anatid.ingest import AddFact, Correction, MemoryPatch, Relation

    llm = make_llm(tmp_path)
    notes = [
        note(1, "For the record, the ledger service is owned by the Atlas team.", 8),
        note(2, "Ownership of the ledger moves from Atlas to Boreal as of today.", 20),
    ]
    patches = [
        {
            "note_id": "n001",
            "patch": MemoryPatch(
                add_facts=(AddFact("Atlas owns the ledger", ("Atlas", "ledger")),),
                add_relations=(Relation("Atlas", "ledger", rel_kind="owns"),),
            ).to_dict(),
        },
        {
            "note_id": "n002",
            "patch": MemoryPatch(
                corrections=(
                    Correction(
                        "Boreal owns the ledger",
                        old_text="Atlas owns the ledger",
                        entities=("Boreal", "ledger"),
                    ),
                ),
                add_relations=(Relation("Boreal", "ledger", rel_kind="owns"),),
                remove_relations=(Relation("Atlas", "ledger", rel_kind="owns"),),
            ).to_dict(),
        },
    ]
    system = H.AnatidSystem(llm, extractor="gold", gold_patches=patches)
    system.ingest(notes)
    assert system.ingest_report["memories_created"] == 2
    assert system.ingest_report["memories_closed"] == 1
    contexts = system.retrieve("Which team owns the ledger?", 1200)
    assert contexts, "recall found nothing"
    block = contexts[0]
    assert block.startswith(
        "- Boreal owns the ledger [valid 2025-01-20 to now; recorded by standup/atlas/2025-01-20]"
    )
    assert (
        "earlier: Atlas owns the ledger [valid 2025-01-08 to 2025-01-20; recorded by standup/atlas/2025-01-08]"
        in block
    )
    text_only = system.ablation("anatid-text", ["text"])
    text_only.ingest(notes)
    assert text_only.retrieve("Which team owns the ledger?", 1200)
    assert text_only.last_retrieval["arms_ran"] == ["text"]
    graph_only = system.ablation("anatid-graph", ["graph"])
    graph_only.ingest(notes)
    graph_only.retrieve("Which team owns the ledger?", 1200)
    assert graph_only.last_retrieval["arms_ran"] == ["graph"]
    assert "ledger" in graph_only.last_retrieval["seeds"]
    system.close()


def test_pin_ties_orders_tied_scores_by_memory_id_and_nothing_else():
    rows = [(9, 3.0272204179158297), (7, 3.0272204179158293), (5, 9.0), (4, 0.5)]
    assert H.pin_ties(rows) == [
        (5, 9.0),
        (7, 3.0272204179158293),
        (9, 3.0272204179158297),
        (4, 0.5),
    ]
    assert H.pin_ties(list(reversed(rows))) == H.pin_ties(rows)


def test_fused_recall_and_existing_facts_are_the_same_on_every_call(tmp_path):
    from anatid.ingest import AddFact, MemoryPatch, Relation

    llm = make_llm(tmp_path)
    notes = [
        note(i, f"Reminder that the {svc} is owned by the Atlas team.", 2 + i)
        for i, svc in enumerate(("ledger", "catalog", "notifier", "indexer", "scheduler"), 1)
    ]
    patches = [
        {
            "note_id": n.note_id,
            "patch": MemoryPatch(
                add_facts=(
                    AddFact(
                        f"Atlas owns the {n.text.split('the ')[1].split(' is')[0]}",
                        ("Atlas", n.text.split("the ")[1].split(" is")[0]),
                    ),
                ),
                add_relations=(
                    Relation("Atlas", n.text.split("the ")[1].split(" is")[0], rel_kind="owns"),
                ),
            ).to_dict(),
        }
        for n in notes
    ]
    system = H.AnatidSystem(llm, extractor="gold", gold_patches=patches)
    system.ingest(notes)
    # every memory ties on BM25 for a query that names only the team; the order must not move
    orders = {
        tuple(
            m.memory_id
            for m in H.fused_recall(system.db, "Which services does Atlas own?", k=10)[0]
        )
        for _ in range(5)
    }
    assert len(orders) == 1
    facts = {
        tuple(m.memory_id for m in H.existing_facts(system.db, "Atlas owns everything now."))
        for _ in range(5)
    }
    assert len(facts) == 1 and len(next(iter(facts))) == 5
    _memories, info = H.fused_recall(
        system.db, "Which services does Atlas own?", arms=["graph"], k=10
    )
    assert info["arms_ran"] == ["graph"] and info["seeds"] == ["Atlas"]
    system.close()


def test_the_extractor_sees_existing_facts_newest_first_by_id():
    from anatid import Memory

    class Recording:
        def __init__(self) -> None:
            self.seen: list[list[int]] = []

        def extract(self, text, *, existing):
            self.seen.append([m.memory_id for m in existing])
            return text

    inner = Recording()
    wrapped = H.SortedExistingExtractor(inner)
    facts = [Memory(memory_id=i, tenant_id=1, content=str(i)) for i in (5, 9, 2)]
    wrapped.extract("note", existing=facts)
    wrapped.extract("note", existing=list(reversed(facts)))
    assert inner.seen == [[9, 5, 2], [9, 5, 2]]


def test_builtin_systems_are_the_eleven_registered_names(tmp_path):
    llm = make_llm(tmp_path)
    systems = H.builtin_systems(llm)
    assert tuple(s.name for s in systems) == H.SYSTEM_NAMES
    assert len(H.SYSTEM_NAMES) == 11
    assert [s.name for s in systems if s.unbudgeted] == ["markdown-full"]
    assert {s.shares_store_with for s in systems if s.shares_store_with} == {"anatid"}
    chosen, family = H.default_systems(llm, family="builtin")
    assert family["family"] == "builtin" and len(chosen) == 11
    external, family = H.default_systems(llm, family="external")
    assert family["family"] == "external"
    assert tuple(s.name for s in external) == H.SYSTEM_NAMES


# --------------------------------------------------------------------------- the judge


def test_the_lexical_scorer_accepts_aliases_and_rejects_stale_values():
    q = question(
        "q1",
        "knowledge_update",
        "Which team owns the ledger now?",
        "Boreal",
        aliases=["the Boreal team", "team Boreal"],
        distractors=["Atlas"],
    )
    assert J.lexical_score(q, "Boreal").correct
    assert J.lexical_score(q, "The Boreal team owns it.").correct
    assert J.lexical_score(q, "Boreal (it was Atlas before).").correct
    stale = J.lexical_score(q, "Atlas, and later Boreal")
    assert not stale.correct and stale.stale
    assert not J.lexical_score(q, "Atlas").correct
    assert not J.lexical_score(q, "Borealis").correct  # whole words only
    assert not J.lexical_score(q, "I don't know").correct
    assert J.lexical_score(q, "I don't know").refused


def test_the_lexical_scorer_handles_lists_dates_and_provenance():
    services = question(
        "q2", "multi_hop", "Which services does Atlas own?", "catalog, image-resizer and event-bus"
    )
    assert J.lexical_score(
        services, "Atlas owns the catalog, the event-bus and image-resizer."
    ).correct
    assert not J.lexical_score(services, "catalog and event-bus").correct
    when = question(
        "q3",
        "temporal",
        "When did Atlas take over?",
        "2026-02-11",
        aliases=["11 February 2026", "February 11, 2026"],
    )
    assert J.lexical_score(when, "On February 11, 2026.").correct
    assert J.lexical_score(when, "2026-02-11").correct
    assert not J.lexical_score(when, "2026-02-12").correct
    prov = question(
        "q4",
        "provenance",
        "Which note recorded it?",
        "handover/2025-10-13-ledger (2025-10-13)",
        aliases=["handover/2025-10-13-ledger", "2025-10-13", "the handover message of 2025-10-13"],
    )
    assert J.lexical_score(prov, "handover/2025-10-13-ledger").correct
    assert J.lexical_score(prov, "The handover message of 2025-10-13.").correct


def test_abstention_is_scored_as_a_refusal_and_only_there():
    abstain = question("q5", "abstention", "Who is the manager of Atlas?", "I don't know")
    for answer in (
        "I don't know",
        "I don't know.",
        "I do not know",
        "The memory does not contain the answer.",
    ):
        verdict = J.lexical_score(abstain, answer)
        assert verdict.correct and verdict.refused, answer
    assert not J.lexical_score(abstain, "Priya").correct
    answerable = question("q6", "single_fact", "Which team is Farah on?", "Dune")
    assert not J.lexical_score(answerable, "I don't know").correct
    scored = [
        J.ScoredAnswer(
            "q5", "abstention", "t", "I don't know", J.lexical_score(abstain, "I don't know"), None
        ),
        J.ScoredAnswer(
            "q6",
            "single_fact",
            "t",
            "I don't know",
            J.lexical_score(answerable, "I don't know"),
            None,
        ),
        J.ScoredAnswer("q7", "single_fact", "t", "Dune", J.lexical_score(answerable, "Dune"), None),
    ]
    summary = J.summarise(scored)
    assert summary["abstention"]["precision"] == 0.5
    assert summary["abstention"]["recall"] == 1.0
    assert summary["abstention"]["false_refusals"] == 1
    assert summary["by_category"]["all"]["lexical"] == pytest.approx(2 / 3)
    assert summary["by_category"]["all"]["llm"] is None


def test_the_judge_never_sees_the_system(tmp_path):
    transport = FakeTransport(reply='{"correct": true, "reason": "matches the gold"}')
    llm = make_llm(tmp_path, transport)
    q = question("q8", "single_fact", "Which team owns the ledger?", "Atlas")
    answer = "Atlas"
    messages = J.judge_messages(q, answer)
    assert all(SECRET not in m["content"] for m in messages)
    assert all(
        "markdown" not in m["content"].lower() and "anatid" not in m["content"].lower()
        for m in messages
    )
    run = H.SystemRun(
        name=SECRET,
        description=SECRET,
        budget=1200,
        ingest_s=0.0,
        ingest_stats={},
        ingest_report={},
        answers=[
            H.QuestionRun(
                qid=q.qid,
                category=q.category,
                question=q.question,
                contexts=[],
                n_contexts=0,
                context_tokens=0,
                answer=answer,
                answer_cached=False,
                retrieve_s=0.0,
                answer_s=0.0,
                answer_network_s=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=None,
            )
        ],
        answer_stats={},
    )
    scored = score_runs([run], [q], llm=llm, workers=2, progress=False)
    assert scored[SECRET][0].judge is not None and scored[SECRET][0].judge.correct
    assert transport.calls == 1
    assert SECRET not in json.dumps(transport.bodies)
    # score_answers takes answers keyed by question id and nothing about their origin
    blind = J.score_answers([q], {q.qid: answer}, llm=llm)
    assert blind[0].judge is not None and blind[0].agree is True


def test_the_judge_reply_is_parsed_tolerantly(tmp_path):
    for case, (reply, expected) in enumerate(
        (
            ('{"correct": false, "reason": "stale"}', False),
            ('```json\n{"correct": true, "reason": "ok"}\n```', True),
            ('Sure: {"correct": "yes", "reason": "ok"} done', True),
        )
    ):
        # a directory named from the reply text carried quotes and colons, which Windows rejects
        llm = make_llm(tmp_path / f"case{case}", FakeTransport(reply=reply))
        q = question("q9", "single_fact", "Q?", "A")
        verdict = J.llm_judge(llm, q, "A")
        assert verdict.parsed and verdict.correct is expected
    llm = make_llm(tmp_path / "garbage", FakeTransport(reply="no idea"))
    verdict = J.llm_judge(llm, question("q9", "single_fact", "Q?", "A"), "A")
    assert not verdict.parsed and not verdict.correct


def test_an_answer_emptied_by_the_completion_cap_is_asked_once_more(tmp_path):
    def reply(body: dict) -> dict:
        if body.get("max_tokens") == H.ANSWER_MAX_TOKENS:
            return {"content": "", "finish_reason": "length"}
        return {"content": "Atlas", "finish_reason": "stop"}

    transport = FakeTransport(reply=reply)
    llm = make_llm(tmp_path, transport)
    system = FixedSystem(llm, "fixed", ["- note"])
    assert system.answer("Which team owns the ledger?", ["- note"]) == "Atlas"
    assert transport.calls == 2
    assert transport.bodies[1]["body"]["max_tokens"] == H.ANSWER_MAX_TOKENS_RETRY
    assert system.last_answer == {"finish_reason": "stop", "escalated": True, "truncated": False}

    always_empty = make_llm(
        tmp_path / "e", FakeTransport(reply={"content": "", "finish_reason": "length"})
    )
    system = FixedSystem(always_empty, "fixed", ["- note"])
    assert system.answer("q", ["- note"]) == ""
    assert system.last_answer["truncated"] and system.last_answer["escalated"]
    q = question("q1", "single_fact", "q", "Atlas")
    assert not J.lexical_score(q, "").correct and not J.lexical_score(q, "").refused


# --------------------------------------------------------------------------- the cache


def test_the_cache_makes_a_second_run_free_and_offline_refuses_misses(tmp_path):
    transport = FakeTransport(reply="Atlas")
    llm = make_llm(tmp_path, transport)
    messages = H.build_messages("Which team owns the ledger?", ["- [2025-01-08] s: Atlas owns it."])
    first = llm.chat(messages, max_tokens=50)
    assert first.content == "Atlas" and not first.cached and transport.calls == 1
    second = llm.chat(messages, max_tokens=50)
    assert second.cached and second.content == "Atlas" and transport.calls == 1
    assert llm.embed(["a", "b"]) == llm.embed(["a", "b"]) and transport.calls == 2
    # embeddings are counted per text: 2 chat calls + 4 texts, 3 of them from the cache
    assert llm.stats.calls == 6 and llm.stats.cache_hits == 3 and llm.stats.network_calls == 2
    assert llm.stats.cost_usd_charged == pytest.approx(0.00001)
    assert llm.stats.cost_usd == pytest.approx(0.00002)

    replay_transport = FakeTransport(reply="different")
    replay = LLMClient(
        transport=replay_transport,
        cache_dir=tmp_path / "cache",
        embed_dim=DIM,
        offline=True,
        counter=TokenCounter(prefer_tiktoken=False),
    )
    assert replay.chat(messages, max_tokens=50).content == "Atlas"
    assert replay.embed(["b", "a"])[0] == llm.embed(["b"])[0]
    assert replay_transport.calls == 0
    assert replay.stats.cost_usd_charged == 0.0 and replay.stats.hit_rate == 1.0
    with pytest.raises(OfflineCacheMiss):
        replay.chat(messages, max_tokens=51)
    with pytest.raises(OfflineCacheMiss):
        replay.embed(["never seen"])


def test_the_cache_key_covers_model_messages_and_parameters():
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    assert request_key("/chat/completions", base) == request_key("/chat/completions", dict(base))
    assert request_key("/chat/completions", base) != request_key("/embeddings", base)
    assert request_key("/chat/completions", base) != request_key(
        "/chat/completions", {**base, "temperature": 0.5}
    )
    assert request_key("/chat/completions", base) != request_key(
        "/chat/completions", {**base, "model": "n"}
    )


def test_the_api_key_is_never_in_the_cache_or_the_request_body(tmp_path):
    transport = FakeTransport()
    llm = make_llm(tmp_path, transport)
    llm.chat([{"role": "user", "content": "hello"}])
    files = list((tmp_path / "cache").rglob("*.json"))
    assert files
    for path in files:
        assert "Authorization" not in path.read_text() and "sk-" not in path.read_text()
    assert "api_key" not in json.dumps(transport.bodies)


# --------------------------------------------------------------------------- the runner


def test_run_system_saves_prompts_contexts_and_answers(tmp_path):
    def reply(body: dict) -> str:
        user = body["messages"][-1]["content"]
        return "Atlas" if "ledger" in user.split("Question:")[-1] else "I don't know"

    transport = FakeTransport(reply=reply)
    llm = make_llm(tmp_path, transport)
    notes = [
        note(1, "The ledger service is owned by the Atlas team.", 8),
        note(2, "Boreal holds its planning meeting on Thursdays.", 9),
    ]
    questions = [
        question("q1", "single_fact", "Which team owns the ledger?", "Atlas"),
        question("q2", "abstention", "Who is the manager of Atlas?", "I don't know"),
    ]
    system = H.MarkdownSystem(llm)
    out = tmp_path / "results" / "markdown"
    run = H.run_system(system, notes, questions, budget=1200, out_dir=out, llm=llm)
    assert run.name == "markdown" and run.budget == 1200
    assert [a.answer for a in run.answers] == ["Atlas", "I don't know"]
    assert all(a.context_tokens <= 1200 and a.n_contexts == 2 for a in run.answers)
    saved = json.loads((out / "prompts" / "q1.json").read_text())
    assert saved["messages"] == H.build_messages(questions[0].question, run.answers[0].contexts)
    assert saved["messages"][0]["content"] == H.SYSTEM_PROMPT
    assert H.IDK in H.SYSTEM_PROMPT
    assert (out / "answers.jsonl").read_text().count("\n") == 2
    assert (out / "notes.md").exists() and (out / "ingest.json").exists()
    assert run.answer_stats["network_calls"] == 2

    scored = score_runs([run], questions, llm=None)
    assert [s.lexical.correct for s in scored["markdown"]] == [True, True]
    summary = assemble_summary(
        [run],
        scored,
        questions,
        run_id="t",
        seed=1,
        budget=1200,
        llm=llm,
        family={"family": "builtin", "note": "test"},
        judge_stats=None,
        judge_workers=1,
        started_at="now",
        wall_s=0.1,
        notes_count=2,
        offline=False,
        limit=None,
    )
    entry = summary["systems"][0]
    assert entry["scores"]["by_category"]["all"]["lexical"] == 1.0
    assert entry["scores"]["abstention"]["precision"] == 1.0
    assert entry["mean_context_tokens"] > 0
    report = render_report(summary)
    limits = report.split("## Limitations, first", 1)[1]
    first_limit = limits.strip().splitlines()[0]
    assert "synthetic" in first_limit
    assert "same model family" in limits and "n is small" in limits
    assert "markdown (S1)" in report
    for forbidden in ("—", "not X", "blazing", "seamless"):
        assert forbidden not in report


def test_the_seed_selects_the_corpus(tmp_path):
    """``--seed`` names the world: the committed data for its own seed, a generated corpus for
    any other, written under the run directory with the seed recorded, and the report says so."""
    import subprocess
    import sys

    from bench.quality import DATA_DIR
    from bench.quality.run import corpus_for_seed

    committed = json.loads((DATA_DIR / "counts.json").read_text())["seed"]
    path, generated = corpus_for_seed(committed, tmp_path / "run")
    assert path == DATA_DIR and generated is False

    run_dir = tmp_path / "s11"
    path, generated = corpus_for_seed(11, run_dir, verify=False)
    assert path == run_dir / "data" and generated is True
    counts = json.loads((path / "counts.json").read_text())
    assert counts["seed"] == 11 and counts["questions"] == 150
    notes = H.load_notes(path / "notes.jsonl")
    questions = H.load_questions(path / "questions.jsonl")
    gold = H.load_gold_patches(path / "gold_patches.jsonl")
    assert len(notes) == counts["notes"] and len(gold) == len(notes) and len(questions) == 150
    assert all(p["patch"]["source_text"] == n.rendered for p, n in zip(gold, notes, strict=True))
    # generation is deterministic: a second call writes the same bytes
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    corpus_for_seed(11, run_dir, verify=False)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before

    summary = {
        "run_id": "s11-b1200",
        "seed": 11,
        "budget": 1200,
        "family": {"family": "builtin", "note": "test"},
        "corpus": {
            "seed": 11,
            "data_dir": "bench/quality/results/s11-b1200/data",
            "generated": True,
        },
        "systems": [],
        "questions": [],
    }
    report = render_report(summary)
    assert "--seed 11" in report and "bench/quality/results/s11-b1200/data" in report
    assert "python -m bench.quality.gen_corpus --verify" not in report.split("## Reproduce", 1)[1]

    # the runner also works as a file, which is how the plan invokes it
    root = Path(H.__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(root / "bench" / "quality" / "run.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=False,
    )
    assert proc.returncode == 0 and "--seed" in proc.stdout
