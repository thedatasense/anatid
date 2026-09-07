"""Three project notes, ingested over six months, and what the graph knows afterwards.

The core verbs take one fact or one edge at a time. This example feeds anatid prose instead:
a note on who owns what, a note that changes the owner, and a note that adds a constraint.
For each note the pipeline proposes a memory patch, prints it as a diff, and applies it in
one transaction with the raw note stored as an episode first. Then three reads show the
result: recall_2hop from Ada reaches the constraint and the new owner across two hops,
provenance on the current owner fact walks back through the superseded one to both notes,
and as_of in April answers with the owner the database believed at the time.

Run it:

    python examples/ingest_notes.py                    # offline: a scripted extractor, no key
    python examples/ingest_notes.py --live             # GLM 5.3 Flash through OpenRouter proposes the patches
    python examples/ingest_notes.py --db PATH          # write the database somewhere else; PATH must not exist
    python examples/ingest_notes.py --db PATH --reset  # delete PATH first

Live mode reads OPEN_ROUTER_KEY from the environment or open_router_key= from a .env file.
The script writes ./ingest_demo.anatid beside itself and recreates it on every run. A path
given with --db is a database of yours as far as the script knows: it refuses one that exists
unless --reset says to delete it, so a demo cannot empty a memory you meant to keep.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pathlib
import sys
from collections.abc import Sequence

from anatid import Anatid
from anatid.ingest import (
    AddFact,
    Alias,
    Correction,
    MemoryPatch,
    OpenAICompatibleExtractor,
    PatchReceipt,
    Relation,
    ScriptedExtractor,
    Span,
    ingest,
)

MODEL = "z-ai/glm-5.3-flash"
BASE_URL = "https://openrouter.ai/api/v1"
DB_PATH = pathlib.Path(__file__).with_name("ingest_demo.anatid")
WRITER = "notes-bot"

# --------------------------------------------------------------------------------------
# The notes, in the order they were written.
# --------------------------------------------------------------------------------------

NOTES: list[tuple[_dt.datetime, str, str]] = [
    (
        _dt.datetime(2026, 3, 2, 9, 0),
        "notes/2026-03-02.md",
        (
            "Ada leads the Kestrel team. Kestrel owns the ingest service, and Bo maintains it "
            "day to day."
        ),
    ),
    (
        _dt.datetime(2026, 6, 15, 9, 0),
        "notes/2026-06-15.md",
        "Bo moved to the platform group. Cy took over the ingest service from Bo this week.",
    ),
    (
        _dt.datetime(2026, 8, 20, 9, 0),
        "notes/2026-08-20.md",
        (
            "The ingest service must stay on Python 3.10 until the Kestrel team finishes the "
            "migration. Ada leads Kestrel."
        ),
    ),
]

#: A day between the first two notes, for the as_of read.
APRIL = _dt.datetime(2026, 4, 1, 12, 0)


def scripted_patches() -> list[MemoryPatch]:
    """What a good extractor says about each note, written down so the run needs no model."""
    n1, n2, n3 = (text for _, _, text in NOTES)
    return [
        MemoryPatch(
            add_facts=(
                AddFact(
                    "Ada leads Kestrel",
                    ("Ada", "Kestrel"),
                    span=Span.locate("Ada leads the Kestrel team", n1),
                ),
                AddFact(
                    "Kestrel owns the ingest service",
                    ("Kestrel", "ingest service"),
                    span=Span.locate("Kestrel owns the ingest service", n1),
                ),
                AddFact(
                    "Bo maintains the ingest service",
                    ("Bo", "ingest service"),
                    span=Span.locate("Bo maintains it day to day", n1),
                ),
            ),
            add_relations=(
                Relation("Ada", "Kestrel", "leads"),
                Relation("Kestrel", "ingest service", "owns"),
                Relation("Bo", "ingest service", "maintains"),
            ),
        ),
        MemoryPatch(
            add_facts=(
                AddFact(
                    "Bo works in the platform group",
                    ("Bo", "platform group"),
                    span=Span.locate("Bo moved to the platform group", n2),
                ),
            ),
            corrections=(
                Correction(
                    "Cy maintains the ingest service",
                    old_text="Bo maintains the ingest service",
                    entities=("Cy", "ingest service"),
                    span=Span.locate("Cy took over the ingest service from Bo", n2),
                ),
            ),
            remove_relations=(Relation("Bo", "ingest service", "maintains"),),
            add_relations=(
                Relation("Cy", "ingest service", "maintains"),
                Relation("Bo", "platform group", "member_of"),
            ),
        ),
        MemoryPatch(
            add_facts=(
                AddFact(
                    "The ingest service must stay on Python 3.10 until the Kestrel migration "
                    "finishes",
                    ("ingest service", "Kestrel"),
                    kind="constraint",
                    span=Span.locate("The ingest service must stay on Python 3.10", n3),
                ),
                # The note repeats something the graph already holds. The pipeline's dedupe
                # step drops it and says so in the patch notes.
                AddFact("Ada leads Kestrel", ("Ada", "the Kestrel team")),
            ),
            entity_aliases=(
                Alias("the Kestrel team", "Kestrel", span=Span.locate("the Kestrel team", n3)),
            ),
        ),
    ]


# --------------------------------------------------------------------------------------
# Live mode: the key, and the extractor that talks to the model.
# --------------------------------------------------------------------------------------


def load_key() -> str:
    for var in ("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    for parent in (pathlib.Path(__file__).parent, *pathlib.Path(__file__).parents):
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                key, _, value = line.partition("=")
                if key.strip().lower() in ("open_router_key", "openrouter_api_key"):
                    return value.strip().strip("\"'")
    sys.exit("Set OPEN_ROUTER_KEY, or put open_router_key= in a .env file, or drop --live.")


def make_extractor(live: bool) -> ScriptedExtractor | OpenAICompatibleExtractor:
    if not live:
        return ScriptedExtractor(scripted_patches())
    return OpenAICompatibleExtractor(
        model=MODEL,
        base_url=BASE_URL,
        api_key=load_key(),
        extra_body={"reasoning": {"enabled": True}},
    )


# --------------------------------------------------------------------------------------
# Printing.
# --------------------------------------------------------------------------------------


def rule(title: str = "") -> None:
    line = "-" * 88
    print(f"\n{line}\n{title}\n{line}" if title else line)


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def day(when: _dt.datetime | None) -> str:
    return when.strftime("%Y-%m-%d") if when else "?"


def review(patch: MemoryPatch) -> MemoryPatch:
    """The review hook: show the proposed patch, then let it through.

    A real reviewer edits here (``patch.replace(add_facts=...)``) or returns ``None`` to
    decline, and nothing is written in that case.
    """
    print(indent(patch.describe()))
    return patch


def show_receipt(receipt: PatchReceipt | None) -> None:
    if receipt is None:
        print("    declined; nothing written")
    else:
        print(f"    applied: {receipt.describe()}")


def show_memories(memories: Sequence, *, prefix: str = "    ") -> None:
    if not memories:
        print(f"{prefix}(nothing)")
    for m in memories:
        kind = f" [{m.kind}]" if m.kind and m.kind != "fact" else ""
        print(f"{prefix}{day(m.valid_from)}  {m.content}{kind}")


# --------------------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------------------


def fresh_database(path: pathlib.Path | None, *, reset: bool = False) -> pathlib.Path:
    """Where this run writes, empty, without deleting a database the script did not make.

    The script's own file beside it (``path`` None) is recreated on every run. Any other path
    was chosen by the person running the demo and may hold memory they want: it must not exist
    yet, unless ``reset`` says to delete it. Refusing is a ``SystemExit`` with the reason.
    """
    own = path is None
    db_path = DB_PATH if path is None else path
    if db_path.exists() and not (own or reset):
        raise SystemExit(
            f"{db_path} exists. This demo writes a fresh database and will not delete one it "
            f"did not make: pass --reset to delete it first, or choose another --db path."
        )
    for stale in (db_path, pathlib.Path(f"{db_path}.wal")):
        if stale.exists():
            stale.unlink()
    return db_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live", action="store_true", help="use GLM through OpenRouter")
    parser.add_argument(
        "--db",
        type=pathlib.Path,
        default=None,
        help=(
            f"database path; must not exist yet (default: {DB_PATH.name} beside this script, "
            f"recreated on every run)"
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the database at --db before the run instead of refusing it",
    )
    args = parser.parse_args(argv)

    db_path = fresh_database(args.db, reset=args.reset)

    extractor = make_extractor(args.live)
    mode = f"live, {MODEL} via OpenRouter" if args.live else "offline, scripted extractor"

    with Anatid.open(db_path, tenant=1, embedding_dim=8) as db:
        rule(f"1. Ingest three notes ({mode})")
        for when, source, text in NOTES:
            print(f"\n  {day(when)}  {source}")
            print(indent(text, "    > "))
            print()
            receipt = ingest(
                db,
                text,
                extractor=extractor,
                writer=WRITER,
                source=source,
                review=review,
                now=when,
            )
            show_receipt(receipt)

        rule("2. recall_2hop('Ada'): what is reachable from Ada across two hops")
        print("  Ada -leads-> Kestrel -owns-> ingest service. No fact about the ingest service")
        print("  mentions Ada; the graph walk is what reaches them.")
        show_memories(db.recall_2hop("Ada", limit=10))

        rule("3. Who maintains the ingest service now, and why")
        about_service = db.context("ingest service", limit=20)
        corrected = [m for m in about_service if db.provenance(m.memory_id).depth > 0]
        if not corrected:
            print("  No current fact about the ingest service supersedes an older one. The")
            print("  extractor proposed no correction for the second note this run.")
            show_memories(about_service)
        for memory in corrected:
            prov = db.provenance(memory.memory_id)
            print(f"  now:     {memory.content}")
            for older in prov.chain[1:]:
                print(f"  before:  {older.content}  (valid until {day(older.valid_to)})")
            print("  chain of evidence, newest first:")
            for link, episode in zip(prov.chain, prov.episodes, strict=False):
                print(f"    {day(link.valid_from)}  {episode.source or '-'}")
                print(indent(episode.content, "      > "))
            print(f"  writers: {', '.join(prov.writers)}")

        rule(
            f"4. as_of({day(APRIL)}): what the database believed about the ingest service in April"
        )
        show_memories(db.as_of(APRIL).context("ingest service", limit=20))

        rule("5. The same read today")
        show_memories(db.context("ingest service", limit=20))

        stats = db.stats()
        rule("6. What the file holds")
        print(
            f"  {stats.get('episodes', 0)} episodes, {stats.get('memories', 0)} memory rows "
            f"(versions included), {stats.get('entities', 0)} entities, "
            f"{stats.get('edges_relates', 0)} relates edge rows"
        )
        print(f"  database: {db_path}")


if __name__ == "__main__":
    main()
