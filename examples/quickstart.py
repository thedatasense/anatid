"""anatid quickstart -- remember, recall, supersede, as-of, provenance, forget.

Runs on a laptop with no API key, no server and no network: `python examples/quickstart.py`.
The embeddings here are a deterministic bag-of-words hash so the script is reproducible; in a
real agent you would pass vectors from whatever embedding model you already use, and open the
database with `embedding_dim=` set to that model's dimension.

Everything below is one DuckDB file (here, an in-memory one).
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta

from anatid import Anatid

DIM = 64


def embed(text: str) -> list[float]:
    """A deterministic stand-in for a real embedding model: hashed bag of words, L2-normalized."""
    vec = [0.0] * DIM
    for word in text.lower().split():
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        vec[digest[0] % DIM] += 1.0
        vec[digest[1] % DIM] += 0.5
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def main() -> None:
    # A fixed clock so every run of this script prints the same timestamps.
    t0 = datetime(2026, 3, 1, 9, 0, 0)

    with Anatid.open(":memory:", tenant=1, embedding_dim=DIM) as db:
        print(f"anatid schema v{db.info().schema_version} on duckdb {db.info().duckdb_version}, "
              f"tenant {db.namespace.tenant_id}, expand path: {db.expand_path}")

        # 1. Evidence before belief. The raw source text is written first; the memories derived
        #    from it carry its episode_id, which is what provenance() walks back to.
        note = db.episode(
            "Standup 2026-03-01: Ada is leading Project Kestrel; she takes her coffee dark roast. "
            "Kestrel depends on the ingest service that Bo maintains.",
            source="standup-notes", writer="agent-1", now=t0)

        # 2. Facts. `entities=` creates the entities on demand and wires up the ABOUT edges that
        #    make each memory reachable by graph traversal.
        pref = db.remember("Ada prefers dark roast coffee",
                           entities=["Ada", "coffee"], kind="preference",
                           embedding=embed("Ada prefers dark roast coffee"),
                           writer="agent-1", episode_id=note.episode_id, now=t0)
        db.remember("Ada leads Project Kestrel",
                    entities=["Ada", "Kestrel"], kind="fact",
                    embedding=embed("Ada leads Project Kestrel"),
                    writer="agent-1", episode_id=note.episode_id, now=t0)
        db.remember("The ingest service is maintained by Bo",
                    entities=["ingest service", "Bo"], kind="fact",
                    embedding=embed("The ingest service is maintained by Bo"),
                    writer="agent-1", episode_id=note.episode_id,
                    now=t0 + timedelta(minutes=1))

        # 3. Entity-to-entity edges. These are the hops recall traverses; ABOUT edges alone
        #    would only ever find memories filed directly under the seed.
        db.relate("Ada", "Kestrel", rel_kind="leads", writer="agent-1", now=t0)
        db.relate("Kestrel", "ingest service", rel_kind="depends_on", writer="agent-1", now=t0)

        # 4. BM25 rides DuckDB's fts index, which is NOT incremental: rows written since the last
        #    rebuild are invisible to the text arm. anatid never hides that -- you decide when to
        #    pay for a rebuild, and every recall() result reports its own staleness.
        status = db.fts_status()
        print(f"before rebuild: bm25 stale={status.stale}, rows waiting={status.pending_rows}")
        db.rebuild_fts_index(now=t0)

        # 5. Hybrid recall: cosine + BM25 + 2-hop graph expansion, fused with RRF (k=60).
        hits = db.recall("coffee roast", embedding=embed("coffee roast"),
                         seed_entity="Ada", k=3)
        print(f"\nrecall(query + embedding + seed): arms={hits.arms} stale={hits.bm25_stale}")
        for h in hits:
            print(f"  [{h.rank}] {h.score:.4f} {h.content!r} via {'+'.join(h.sources)} "
                  f"about={list(h.about)}")

        # 6. Pure graph recall: 2 hops out from Ada over RELATES_TO, in both directions.
        #    Bo's memory is 2 hops away (Ada -> Kestrel -> ingest service).
        print("\nrecall_2hop('Ada'):")
        for m in db.recall_2hop("Ada", limit=5):
            print(f"  {m.content!r}")

        # 7. Supersede: the old belief is not deleted, it is closed. entities= and kind= are
        #    inherited from the memory being replaced unless you override them.
        t1 = t0 + timedelta(days=30)
        newer = db.supersede(pref.memory_id, "Ada switched to decaf",
                             writer="agent-2", embedding=embed("Ada switched to decaf"), now=t1)
        old = db.get(pref.memory_id)
        print(f"\nsupersede: old is_current={old.is_current} valid_to={old.valid_to} "
              f"-> new {newer.content!r}")

        # 8. Time travel -- anatid's own filter over valid_from/valid_to and tx_from/tx_to.
        #    DuckDB has no AS OF SYSTEM TIME clause; nothing rewinds, we just filter.
        before = db.as_of(t0 + timedelta(days=1)).recall_2hop("Ada", limit=5)
        after = db.recall_2hop("Ada", limit=5)
        print(f"\nas_of(day 1)  : {[m.content for m in before]}")
        print(f"current       : {[m.content for m in after]}")

        # 9. Provenance: walk the SUPERSEDES chain back to the first assertion and the raw
        #    evidence behind it.
        prov = db.provenance(newer.memory_id)
        print(f"\nprovenance(depth={prov.depth}, writers={list(prov.writers)}):")
        for m in prov.chain:
            print(f"  {'current ' if m.is_current else 'closed  '} {m.content!r} (by {m.writer})")
        print(f"  source: {prov.source_text[:60]!r}...")

        # 10. Erasure. hard=True really removes the row, its edges, its embedding and its
        #     provenance -- there is no soft tombstone left to un-delete.
        receipt = db.forget(newer.memory_id, hard=True, reason="right-to-erasure request",
                            writer="agent-2", now=t1)
        print(f"\nforget(hard=True): rows_removed={receipt.rows_removed} "
              f"about_edges={receipt.about_edges_deleted} "
              f"supersedes_edges={receipt.supersedes_edges_deleted} "
              f"audit_rows_deleted={receipt.audit_rows_deleted}")

        s = db.stats()
        print(f"\nstats: memories={s['memories']} current={s['current_memories']} "
              f"entities={s['entities']} about={s['edges_about']} relates={s['edges_relates']}")


if __name__ == "__main__":
    main()
