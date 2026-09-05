"""S2: BM25 over raw notes with DuckDB's full-text-search extension.

One row per note in a ``notes`` table, ``PRAGMA create_fts_index`` over the note's ``rendered``
string, ``match_bm25`` at query time with the question as the query.  The extension applies
the same tokenizer to documents and queries: lower-case, strip accents, drop English stop
words, Porter-stem, and split on every run of characters outside the kept alphabet.

Two choices make this a baseline worth beating rather than a straw man:

* Digits are kept.  DuckDB's default ``ignore`` pattern drops everything outside ``a-z``, which
  turns ``INC-2025-011`` into ``inc`` and ``11:00 UTC`` into ``utc``.  With ``[^a-z0-9]`` as the
  separator instead, incident ids, times and the dates in every header are searchable terms.
  ``keep_digits=False`` restores the default for an ablation.
* Ties break by recency.  Two notes with the same score are ordered newest first, so when the
  question does not distinguish an old handover from the new one, the new one comes first.

Hyphens and slashes are separators, so ``billing-api`` matches ``billing API`` and the source
label ``handover/2025-10-13-ledger`` contributes ``handover``, the date parts and ``ledger``.

Only notes with a non-null score (at least one query term present) are candidates; the
context is never padded with unrelated notes.  The kept hits are shown oldest first.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import duckdb

from bench.quality.systems import (
    DEFAULT_BUDGET,
    Hit,
    Note,
    Retrieval,
    count_tokens,
    pack,
)

__all__ = ["Bm25System", "BM25_K1", "BM25_B"]

#: Okapi BM25 parameters, the usual ones and the ones anatid's own text arm uses.
BM25_K1, BM25_B = 1.2, 0.75

#: Separator patterns for the fts tokenizer.  Every run of a match is a token boundary.
_IGNORE_WITH_DIGITS = r"(\.|[^a-z0-9])+"
_IGNORE_DEFAULT = r"(\.|[^a-z])+"


def _load_fts(con: duckdb.DuckDBPyConnection) -> None:
    try:
        con.execute("LOAD fts")
    except duckdb.Error:
        con.execute("INSTALL fts")
        con.execute("LOAD fts")


class Bm25System:
    """DuckDB fts BM25 over one row per note.  See the module docstring."""

    name: str

    def __init__(
        self,
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        stemmer: str = "porter",
        stopwords: str = "english",
        keep_digits: bool = True,
        order: str = "chronological",
        name: str | None = None,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.keep_digits = bool(keep_digits)
        self.order = order
        self.name = name or "bm25"
        self._con: duckdb.DuckDBPyConnection | None = None
        self._notes: dict[str, Note] = {}
        self.build_seconds = 0.0
        self.vocabulary = 0

    def __repr__(self) -> str:
        return (
            f"Bm25System(name={self.name!r}, k1={self.k1}, b={self.b}, stemmer={self.stemmer!r}, "
            f"keep_digits={self.keep_digits}, notes={len(self._notes)})"
        )

    @property
    def ignore_pattern(self) -> str:
        return _IGNORE_WITH_DIGITS if self.keep_digits else _IGNORE_DEFAULT

    # ------------------------------------------------------------------------------- build

    def build(self, notes: Sequence[Note]) -> None:
        started = time.perf_counter()
        self.close()
        con = duckdb.connect(":memory:")
        _load_fts(con)
        con.execute(
            "CREATE TABLE notes (note_id VARCHAR PRIMARY KEY, seq INTEGER, date VARCHAR, "
            "rendered VARCHAR)"
        )
        self._notes = {note.note_id: note for note in notes}
        con.executemany(
            "INSERT INTO notes VALUES (?, ?, ?, ?)",
            [(n.note_id, n.seq, n.date, n.rendered) for n in notes],
        )
        # The tokenizer settings are the PRAGMA's own keywords; the ignore pattern is a SQL
        # string literal, and neither pattern above contains a quote.
        con.execute(
            "PRAGMA create_fts_index('notes', 'note_id', 'rendered', "
            f"stemmer='{self.stemmer}', stopwords='{self.stopwords}', "
            f"ignore='{self.ignore_pattern}', strip_accents=1, lower=1, overwrite=1)"
        )
        row = con.execute("SELECT count(*) FROM fts_main_notes.dict").fetchone()
        self.vocabulary = int(row[0]) if row else 0
        self._con = con
        self.build_seconds = time.perf_counter() - started

    # ---------------------------------------------------------------------------- retrieve

    def rank(self, question: str) -> list[Hit]:
        """Every note with a BM25 score for ``question``, best first, newest first on ties."""
        if self._con is None:
            raise RuntimeError("Bm25System.build() has not run")
        rows = self._con.execute(
            "SELECT note_id, score FROM ("
            "  SELECT note_id, seq, fts_main_notes.match_bm25("
            "    note_id, ?, k := ?, b := ?, conjunctive := 0) AS score"
            "  FROM notes"
            ") WHERE score IS NOT NULL ORDER BY score DESC, seq DESC",
            [question, self.k1, self.b],
        ).fetchall()
        return [
            Hit.of_note(self._notes[note_id], rank=position, score=float(score))
            for position, (note_id, score) in enumerate(rows, start=1)
        ]

    def retrieve(self, question: str, *, budget: int | None = DEFAULT_BUDGET) -> Retrieval:
        started = time.perf_counter()
        ranked = self.rank(question)
        text, kept, skipped = pack(ranked, budget=budget, order=self.order)
        return Retrieval(
            system=self.name,
            question=question,
            budget=budget,
            text=text,
            hits=kept,
            tokens=count_tokens(text),
            considered=len(ranked),
            skipped=skipped,
            seconds=time.perf_counter() - started,
            meta={
                "k1": self.k1,
                "b": self.b,
                "stemmer": self.stemmer,
                "stopwords": self.stopwords,
                "keep_digits": self.keep_digits,
                "ignore": self.ignore_pattern,
                "tie_break": "newest first",
                "order": self.order,
                "vocabulary": self.vocabulary,
                "build_seconds": self.build_seconds,
            },
        )

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
