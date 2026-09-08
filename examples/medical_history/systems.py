"""Same-source retrieval baselines. No gold imports or access to the simulator's Case objects."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from anatid import Anatid, AsOf
from anatid.embed import Embedder

from .world import Query, Record

SYSTEMS = ("raw-vector", "raw-hybrid", "trace-sql", "anatid-hybrid", "anatid-trace")


def timestamp(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def token_estimate(record: Record) -> int:
    """Shared approximate context accounting, NOT a particular model's tokenizer."""
    return (len(record.render()) + 3) // 4


def pack(records: list[Record], budget: int) -> list[Record]:
    """A whole-record ranked prefix: never read fields beyond the returned context."""
    out: list[Record] = []
    seen: set[str] = set()
    used = 0
    for record in records:
        if record.record_id in seen:
            continue
        size = token_estimate(record)
        if used + size > budget:
            break
        seen.add(record.record_id)
        used += size
        out.append(record)
    return out


def fuse(*rankings: list[str]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, rid in enumerate(ranking, 1):
            scores[rid] += 1 / (60 + rank)
    return sorted(scores, key=lambda rid: (-scores[rid], rid))


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("embedding must be finite and nonzero")
    return [v / norm for v in vector]


class RetrievalSystems:
    """Build all indexes from the identical source exports and one embedding per document.

    anatid-hybrid is the library's existing weighted recall. anatid-trace is an explicit
    application strategy: expand the query's source identifier, then rank that neighborhood
    with the same raw hybrid ranker as SQL. Neither gets privileged links or a gold answer.
    """

    def __init__(self, records: tuple[Record, ...], embedder: Embedder, database_path: Path):
        if database_path.exists() or database_path.is_symlink():
            raise FileExistsError(f"refusing to open an existing example database: {database_path}")
        self.records = {r.record_id: r for r in records}
        if len(self.records) != len(records):
            raise ValueError("duplicate source identifiers")
        self.embedder = embedder
        texts = [r.render() for r in records]
        vectors = embedder.embed(texts)
        if len(vectors) != len(records) or any(len(v) != embedder.dim for v in vectors):
            raise ValueError("wrong embedding count or dimension")
        self.vectors = {r.record_id: normalize(v) for r, v in zip(records, vectors, strict=True)}
        self.words = {
            r.record_id: Counter(re.findall(r"\w+", text.lower()))
            for r, text in zip(records, texts, strict=True)
        }
        self.sql = sqlite3.connect(":memory:")
        self.sql.executescript("""
            CREATE TABLE links(src TEXT, dst TEXT, effective TEXT, recorded TEXT);
            CREATE INDEX links_src ON links(src);
            CREATE INDEX links_dst ON links(dst);
        """)
        try:
            # Single-threaded aggregation avoids parallel floating-point tie drift in this
            # small, heavily templated corpus. This is not a throughput benchmark.
            self.db = Anatid.open(database_path, embedding_dim=embedder.dim, threads=1)
        except BaseException:
            self.sql.close()
            raise
        self.memory_to_record: dict[int, str] = {}
        try:
            with self.db.transaction(), self.sql:
                # Stable IDs also make rank ties reproducible across fresh databases.
                for mid, record in enumerate(
                    sorted(records, key=lambda r: (r.recorded_on, r.record_id)), 1
                ):
                    node = f"doc:{record.record_id}"
                    now = timestamp(record.recorded_on)
                    effective = timestamp(record.effective_on)
                    memory = self.db.remember(
                        record.render(),
                        entities=[node],
                        embedding=self.vectors[record.record_id],
                        episode=record.render(),
                        episode_source=f"synthetic://{record.record_id}",
                        writer="synthetic-source-import",
                        memory_id=mid,
                        now=now,
                        valid_from=effective,
                    )
                    self.memory_to_record[memory.memory_id] = record.record_id
                    for ref in record.references:
                        self.db.relate(
                            node,
                            ref,
                            rel_kind="source_reference",
                            now=now,
                            valid_from=effective,
                            episode_id=memory.episode_id,
                        )
                        self.sql.execute(
                            "INSERT INTO links VALUES (?, ?, ?, ?)",
                            (node, ref, record.effective_on, record.recorded_on),
                        )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.db.close()
        self.sql.close()

    def raw_rankings(self, query: Query, vector: list[float]) -> tuple[list[str], list[str]]:
        visible = [
            rid
            for rid, r in self.records.items()
            if r.product == query.product and r.visible(query.effective_on, query.known_on)
        ]
        cosine = {
            rid: sum(a * b for a, b in zip(vector, self.vectors[rid], strict=True))
            for rid in visible
        }
        vector_rank = sorted(visible, key=lambda rid: (-cosine[rid], rid))
        # Compute statistics only on the visible corpus: even future document-frequency
        # statistics must not influence a historical reconstruction.
        # Stable summation order across Python hash seeds and separate CLI invocations.
        terms = sorted(set(re.findall(r"\w+", query.question.lower())))
        lengths = {rid: sum(self.words[rid].values()) for rid in visible}
        avg = sum(lengths.values()) / max(1, len(visible)) or 1
        df = Counter(term for rid in visible for term in terms if term in self.words[rid])
        bm25: dict[str, float] = {}
        for rid in visible:
            score = 0.0
            for term in terms:
                tf = self.words[rid][term]
                if tf:
                    idf = math.log(1 + (len(visible) - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[rid] / avg))
            bm25[rid] = score
        text_rank = sorted(visible, key=lambda rid: (-bm25[rid], rid))
        return vector_rank, fuse(vector_rank, text_rank)

    def sql_neighborhood(self, query: Query, hops: int = 2) -> set[str]:
        rows = self.sql.execute(
            """
            WITH RECURSIVE walk(node, depth) AS (
                SELECT ?, 0
                UNION
                SELECT CASE WHEN l.src = w.node THEN l.dst ELSE l.src END, w.depth + 1
                FROM walk w JOIN links l ON l.src = w.node OR l.dst = w.node
                WHERE w.depth < ? AND l.effective <= ? AND l.recorded <= ?
            ) SELECT DISTINCT node FROM walk
        """,
            (query.anchor, hops, query.effective_on, query.known_on),
        ).fetchall()
        return {node[4:] for (node,) in rows if node.startswith("doc:")}

    def retrieve(
        self, query: Query, *, budget: int = 2000, candidates: int = 100
    ) -> dict[str, list[Record]]:
        if budget < 1 or candidates < 1:
            raise ValueError("budget and candidates must be positive")
        vector = normalize(self.embedder.embed_one(query.question))
        raw_vector, raw_hybrid = self.raw_rankings(query, vector)
        scope = AsOf(timestamp(query.effective_on), timestamp(query.known_on))
        seed = query.anchor if self.db.get_entity(query.anchor) is not None else None
        hits = self.db.recall(
            query.question,
            embedding=vector,
            seed_entity=seed,
            hops=2,
            as_of=scope,
            k=candidates,
            candidates=candidates,
        )
        if hits.bm25_stale:
            raise RuntimeError(
                "anatid text index unavailable; refusing a silently weakened baseline"
            )
        anatid_rank = [self.memory_to_record[h.memory_id] for h in hits]
        graph_ids = set()
        if seed is not None:
            graph_ids = {
                self.memory_to_record[m.memory_id]
                for m in self.db.recall_2hop(seed, hops=2, limit=len(self.records), as_of=scope)
            }
        sql_ids = self.sql_neighborhood(query)
        rankings = {
            "raw-vector": raw_vector,
            "raw-hybrid": raw_hybrid,
            "trace-sql": [rid for rid in raw_hybrid if rid in sql_ids],
            "anatid-hybrid": anatid_rank,
            "anatid-trace": [rid for rid in raw_hybrid if rid in graph_ids],
        }
        return {
            name: pack(
                [
                    self.records[rid]
                    for rid in ranking
                    if self.records[rid].product == query.product
                    and self.records[rid].visible(query.effective_on, query.known_on)
                ][:candidates],
                budget,
            )
            for name, ranking in rankings.items()
        }
