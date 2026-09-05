"""Answer-quality benchmark corpus for anatid.

``world`` simulates a small engineering organisation over eighteen months and is the single
source of truth; ``gen_corpus`` renders its events into notes, gold memory patches and a
question set with exact gold answers.  Nothing here is imported by ``import anatid``.

Regenerate with ``python -m bench.quality.gen_corpus --verify`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
NOTES_PATH = DATA_DIR / "notes.jsonl"
GOLD_PATCHES_PATH = DATA_DIR / "gold_patches.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"

__all__ = ["DATA_DIR", "NOTES_PATH", "GOLD_PATCHES_PATH", "QUESTIONS_PATH"]
