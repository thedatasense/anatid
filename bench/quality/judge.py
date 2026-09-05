"""Scoring that never learns which system produced an answer.

Two scorers, reported side by side with their agreement rate:

* :func:`lexical_score` is a normalised match of the gold answer or one of its aliases inside
  the answer, whole words only.  For a question whose value changed over time it also checks
  the distractors (values that were once true): an answer that names a stale value before the
  gold is wrong.  An abstention question is correct only when the answer is a refusal
  (:func:`is_refusal`), and a refusal to any other question is wrong.  It is cheap, exact and
  transparent, and it under-scores paraphrase.
* :func:`llm_judge` asks the same chat model, at temperature 0, whether the answer is correct
  given the question, the gold, its aliases and the category's rubric.  It returns one verdict
  and one sentence, and it is cached like every other call.

Neither scorer is given the system's name, its contexts or its prompt.  :func:`score_answers`
takes a mapping from question id to answer text and nothing else, which is how the runner
keeps it blind.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .harness import IDK, NOW, Question
from .llm import LLMClient

__all__ = [
    "JUDGE_MAX_TOKENS",
    "JUDGE_MAX_TOKENS_RETRY",
    "REFUSAL_PHRASES",
    "JudgeVerdict",
    "LexicalVerdict",
    "ScoredAnswer",
    "gold_items",
    "is_refusal",
    "judge_messages",
    "lexical_score",
    "llm_judge",
    "normalise",
    "phrase_in",
    "score_answers",
    "summarise",
]

#: Completion cap for one judge call; the model reasons before it answers.  A verdict that
#: comes back empty because the reasoning used the whole cap is asked once more with the
#: larger cap.
JUDGE_MAX_TOKENS = 2000
JUDGE_MAX_TOKENS_RETRY = 8000

#: Normalised phrases that make an answer a refusal.
REFUSAL_PHRASES = (
    "i don't know",
    "i do not know",
    "i dont know",
    "don't know",
    "does not contain",
    "doesn't contain",
    "not contain the answer",
    "not in the memory",
    "no information",
    "not recorded",
    "no record of",
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "not enough information",
    "insufficient information",
    "unknown",
)

#: Leading words a gold phrase may carry that an answer may drop ("on Tuesdays" / "Tuesdays").
_LEADING = ("the ", "a ", "an ", "on ", "in ", "at ")


def normalise(text: str) -> str:
    """Lower case, curly quotes straightened, punctuation (other than ``-``, ``/``, ``:`` and
    ``'``) turned to spaces, whitespace collapsed."""
    t = str(text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[^a-z0-9/:'\-]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _strip_leading(phrase: str) -> str:
    for word in _LEADING:
        if phrase.startswith(word):
            return phrase[len(word) :]
    return phrase


def phrase_in(phrase: str, text: str) -> int | None:
    """Position of ``phrase`` in ``text`` as whole words, both normalised; None when absent.

    A leading article or preposition on the phrase is optional ("the ledger" matches
    "ledger", "on Tuesdays" matches "Tuesdays").
    """
    p = normalise(phrase)
    t = normalise(text)
    if not p or not t:
        return None
    for candidate in dict.fromkeys((p, _strip_leading(p))):
        if not candidate:
            continue
        m = re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", t)
        if m:
            return m.start()
    return None


def is_refusal(answer: str) -> bool:
    """True when the answer says the memory does not hold the answer."""
    n = normalise(answer)
    if not n:
        return False
    return any(p in n for p in REFUSAL_PHRASES)


def gold_items(gold: str) -> list[str]:
    """A list-valued gold split into its items: ``"a, b and c"`` -> ``["a", "b", "c"]``."""
    parts = re.split(r",\s*|\s+and\s+", gold)
    return [p.strip() for p in parts if p.strip()]


@dataclass(frozen=True)
class LexicalVerdict:
    correct: bool
    refused: bool
    matched: str | None
    stale: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lexical_score(question: Question, answer: str) -> LexicalVerdict:
    """The exact/alias scorer.  See the module docstring for the rules."""
    refused = is_refusal(answer)
    if question.category == "abstention" or question.gold.strip().lower() == IDK.lower():
        return LexicalVerdict(
            correct=refused, refused=refused, matched=IDK if refused else None, stale=False
        )
    if refused:
        return LexicalVerdict(correct=False, refused=True, matched=None, stale=False)

    best: tuple[int, str] | None = None
    for phrase in (question.gold, *question.aliases):
        at = phrase_in(phrase, answer)
        if at is not None and (best is None or at < best[0]):
            best = (at, phrase)
    if best is None:
        items = gold_items(question.gold)
        if len(items) > 1:
            positions = [phrase_in(item, answer) for item in items]
            if all(p is not None for p in positions):
                best = (min(p for p in positions if p is not None), question.gold)

    stale_at: int | None = None
    for d in question.distractors:
        at = phrase_in(d, answer)
        if at is not None and (stale_at is None or at < stale_at):
            stale_at = at
    stale = stale_at is not None and (best is None or stale_at < best[0])
    correct = best is not None and not stale
    return LexicalVerdict(
        correct=correct, refused=False, matched=best[1] if best else None, stale=stale
    )


# --------------------------------------------------------------------------- the LLM judge

JUDGE_SYSTEM = (
    "You grade one answer to one question against a gold answer. You see the question, the "
    "gold answer, accepted alternative phrasings, a grading rule and the answer to grade. You "
    "do not know where the answer came from and you must not guess. Apply the rule literally. "
    f"Today is {NOW}. Reply with one JSON object and nothing else: "
    '{"correct": true or false, "reason": "one sentence"}.'
)

JUDGE_USER = (
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Also accepted: {aliases}\n"
    "Grading rule: {rubric}\n"
    "\n"
    "Answer to grade: {answer}"
)


def judge_messages(question: Question, answer: str) -> list[dict[str, str]]:
    """The judge's prompt.  It carries the question, the gold, the aliases, the rubric and the
    answer.  Nothing identifies the system."""
    aliases = "; ".join(question.aliases) if question.aliases else "(none)"
    rubric = (
        question.rubric or "Correct if the answer states the gold fact; 'I don't know' is wrong."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                question=question.question,
                gold=question.gold,
                aliases=aliases,
                rubric=rubric,
                answer=answer.strip() or "(empty answer)",
            ),
        },
    ]


@dataclass(frozen=True)
class JudgeVerdict:
    correct: bool
    reason: str
    cached: bool
    parsed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_verdict(text: str) -> tuple[bool | None, str]:
    body = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", body, re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        data = None
        if 0 <= start < end:
            try:
                data = json.loads(body[start : end + 1])
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict) and isinstance(data.get("correct"), bool):
        return bool(data["correct"]), str(data.get("reason") or "")
    if isinstance(data, dict) and isinstance(data.get("correct"), str):
        value = data["correct"].strip().lower()
        if value in ("true", "yes", "correct"):
            return True, str(data.get("reason") or "")
        if value in ("false", "no", "incorrect"):
            return False, str(data.get("reason") or "")
    low = body.lower()
    if '"correct": true' in low or low.startswith("correct"):
        return True, body[:200]
    if '"correct": false' in low or low.startswith("incorrect"):
        return False, body[:200]
    return None, body[:200]


def llm_judge(llm: LLMClient, question: Question, answer: str) -> JudgeVerdict:
    """One cached judge call.  An unparseable reply counts as incorrect and is flagged."""
    messages = judge_messages(question, answer)
    result = llm.chat(
        messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=JUDGE_MAX_TOKENS,
    )
    if not result.content and result.finish_reason == "length":
        result = llm.chat(
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=JUDGE_MAX_TOKENS_RETRY,
        )
    verdict, reason = _parse_verdict(result.content)
    if verdict is None:
        return JudgeVerdict(
            correct=False,
            reason=f"unparseable judge reply: {reason}",
            cached=result.cached,
            parsed=False,
        )
    return JudgeVerdict(correct=verdict, reason=reason, cached=result.cached, parsed=True)


# --------------------------------------------------------------------------- scoring a run


@dataclass(frozen=True)
class ScoredAnswer:
    qid: str
    category: str
    subtype: str
    answer: str
    lexical: LexicalVerdict
    judge: JudgeVerdict | None

    @property
    def agree(self) -> bool | None:
        return None if self.judge is None else self.lexical.correct == self.judge.correct

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "category": self.category,
            "subtype": self.subtype,
            "answer": self.answer,
            "lexical": self.lexical.to_dict(),
            "judge": None if self.judge is None else self.judge.to_dict(),
            "agree": self.agree,
        }


def score_answers(
    questions: Sequence[Question],
    answers: Mapping[str, str],
    *,
    llm: LLMClient | None = None,
) -> list[ScoredAnswer]:
    """Score ``answers`` (question id -> answer text) with both scorers.

    This function is the blind boundary: it receives answers keyed by question id and
    nothing about their origin.  ``llm=None`` skips the LLM judge.
    """
    out: list[ScoredAnswer] = []
    for q in questions:
        answer = str(answers.get(q.qid, ""))
        lexical = lexical_score(q, answer)
        judge = llm_judge(llm, q, answer) if llm is not None else None
        out.append(ScoredAnswer(q.qid, q.category, q.subtype, answer, lexical, judge))
    return out


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def summarise(scored: Sequence[ScoredAnswer]) -> dict[str, Any]:
    """Accuracy per category for both scorers, agreement, abstention and staleness rates."""
    categories = sorted({s.category for s in scored})
    per_cat: dict[str, dict[str, Any]] = {}
    for cat in [*categories, "all"]:
        rows = [s for s in scored if cat == "all" or s.category == cat]
        judged = [s for s in rows if s.judge is not None]
        per_cat[cat] = {
            "n": len(rows),
            "lexical": _rate(sum(s.lexical.correct for s in rows), len(rows)),
            "llm": _rate(sum(bool(s.judge and s.judge.correct) for s in judged), len(judged)),
            "agreement": _rate(sum(bool(s.agree) for s in judged), len(judged)),
            "refusals": sum(s.lexical.refused for s in rows),
        }
    abst = [s for s in scored if s.category == "abstention"]
    answerable = [s for s in scored if s.category != "abstention"]
    refused_all = [s for s in scored if s.lexical.refused]
    stale_pool = [s for s in answerable if s.category in ("knowledge_update", "temporal")]
    return {
        "by_category": per_cat,
        "abstention": {
            "precision": _rate(
                sum(s.category == "abstention" for s in refused_all), len(refused_all)
            ),
            "recall": _rate(sum(s.lexical.refused for s in abst), len(abst)),
            "false_refusals": sum(s.lexical.refused for s in answerable),
            "false_refusal_rate": _rate(
                sum(s.lexical.refused for s in answerable), len(answerable)
            ),
            "refusals_total": len(refused_all),
        },
        "stale": {
            "n": len(stale_pool),
            "stale_answers": sum(s.lexical.stale for s in stale_pool),
            "stale_rate": _rate(sum(s.lexical.stale for s in stale_pool), len(stale_pool)),
        },
        "judge_unparsed": sum(1 for s in scored if s.judge is not None and not s.judge.parsed),
    }
