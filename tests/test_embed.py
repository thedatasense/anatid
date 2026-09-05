"""Embeddings as a protocol (:mod:`anatid.embed`) and the MCP server's embedder configuration
(:mod:`anatid.integrations.mcp.embedding`).

The OpenAI-compatible client is exercised against a real HTTP server on a loopback port, with
the request it sent inspected and the response shaped the way the endpoints shape it (out of
order, with ``index``), so nothing about the wire format is assumed by both sides at once.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from anatid import Anatid, Embedder, EmbedderError, EmbeddingDimensionError, HashEmbedder
from anatid import OpenAICompatibleEmbedder
from anatid.integrations.mcp import embedding as mcp_embedding
from conftest import DIM, T0, vec


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


# ============================================================================ HashEmbedder


def test_hash_embedder_is_deterministic_unit_length_and_word_based():
    h = HashEmbedder(DIM)
    assert isinstance(h, Embedder)
    assert h.dim == DIM
    a = h.embed_one("Bo maintains the ingest service")
    assert a == h.embed_one("Bo maintains the ingest service")
    assert a == HashEmbedder(DIM).embed_one("bo MAINTAINS the ingest service")
    assert len(a) == DIM and math.isclose(math.sqrt(sum(x * x for x in a)), 1.0)
    assert h.embed(["x", "y"]) == [h.embed_one("x"), h.embed_one("y")]
    # Shared words score higher than disjoint ones; that is the whole model.
    same = h.embed_one("who maintains the ingest service")
    other = h.embed_one("nightly vacuum window")
    assert cosine(a, same) > cosine(a, other)
    assert all(math.isfinite(x) for x in a)


def test_hash_embedder_never_returns_the_zero_vector():
    h = HashEmbedder(4)
    assert h.embed_one("") == [1.0, 0.0, 0.0, 0.0]
    assert h.embed_one("... !!!") == [1.0, 0.0, 0.0, 0.0]


def test_hash_embedder_seed_and_dimension_are_checked():
    assert HashEmbedder(DIM, seed=1).embed_one("x") != HashEmbedder(DIM, seed=2).embed_one("x")
    with pytest.raises(ValueError):
        HashEmbedder(0)
    assert "HashEmbedder(dim=8" in repr(HashEmbedder(8))


# ============================================================================ OpenAI-compatible


class _Endpoint:
    """A loopback ``/embeddings`` endpoint that records requests and serves a scripted reply."""

    def __init__(self, dim: int):
        self.dim = dim
        self.requests: list[dict] = []
        self.status = 200
        self.body = None  # None: echo one deterministic vector per input, out of order
        self.raw = None  # bytes to send verbatim instead of JSON

        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                endpoint.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "content_type": self.headers.get("Content-Type"),
                        "body": payload,
                    }
                )
                if endpoint.raw is not None:
                    data = endpoint.raw
                elif endpoint.body is not None:
                    data = json.dumps(endpoint.body).encode()
                else:
                    items = [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [float(i + 1)] + [0.0] * (endpoint.dim - 1),
                        }
                        for i in range(len(payload["input"]))
                    ]
                    items.reverse()  # endpoints may answer in any order; index is the truth
                    data = json.dumps(
                        {"object": "list", "data": items, "model": payload["model"]}
                    ).encode()
                self.send_response(endpoint.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # silence the test log
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def endpoint():
    ep = _Endpoint(DIM)
    try:
        yield ep
    finally:
        ep.close()


def test_openai_compatible_embedder_speaks_the_embeddings_shape(endpoint):
    emb = OpenAICompatibleEmbedder(endpoint.base_url + "/", "sk-test", "text-embedding-x", DIM)
    assert isinstance(emb, Embedder)
    assert emb.url == endpoint.base_url + "/embeddings"
    vectors = emb.embed(["first", "second", "third"])
    assert vectors == [
        [1.0] + [0.0] * (DIM - 1),
        [2.0] + [0.0] * (DIM - 1),
        [3.0] + [0.0] * (DIM - 1),
    ]  # restored to input order by ``index``
    (req,) = endpoint.requests
    assert req["path"] == "/v1/embeddings"
    assert req["auth"] == "Bearer sk-test"
    assert req["content_type"] == "application/json"
    assert req["body"] == {"model": "text-embedding-x", "input": ["first", "second", "third"]}
    assert emb.embed_one("solo") == [1.0] + [0.0] * (DIM - 1)


def test_openai_compatible_embedder_options(endpoint):
    no_key = OpenAICompatibleEmbedder(
        endpoint.base_url + "/embeddings", None, "m", DIM, request_dimensions=True, batch_size=2
    )
    assert no_key.url == endpoint.base_url + "/embeddings"
    out = no_key.embed(["a", "b", "c"])
    assert len(out) == 3 and out[2][0] == 1.0  # the second batch starts numbering again
    assert len(endpoint.requests) == 2
    assert endpoint.requests[0]["auth"] is None
    assert endpoint.requests[0]["body"]["dimensions"] == DIM
    assert endpoint.requests[0]["body"]["input"] == ["a", "b"]
    assert endpoint.requests[1]["body"]["input"] == ["c"]
    with pytest.raises(ValueError):
        OpenAICompatibleEmbedder("", None, "m", DIM)
    with pytest.raises(ValueError):
        OpenAICompatibleEmbedder(endpoint.base_url, None, "m", 0)
    assert "model='m'" in repr(no_key)


def test_openai_compatible_embedder_refuses_the_wrong_dimension(endpoint):
    emb = OpenAICompatibleEmbedder(endpoint.base_url, None, "m", DIM + 1)
    with pytest.raises(EmbeddingDimensionError) as exc:
        emb.embed_one("x")
    assert exc.value.expected == DIM + 1 and exc.value.got == DIM


def test_openai_compatible_embedder_reports_transport_and_shape_failures(endpoint):
    emb = OpenAICompatibleEmbedder(endpoint.base_url, "k", "m", DIM)
    endpoint.status = 500
    endpoint.body = {"error": {"message": "boom"}}
    with pytest.raises(EmbedderError) as exc:
        emb.embed_one("x")
    assert exc.value.status == 500 and "HTTP 500" in str(exc.value) and "boom" in str(exc.value)

    endpoint.status = 200
    endpoint.body = {"object": "list"}  # no data
    with pytest.raises(EmbedderError, match="no 'data' list"):
        emb.embed_one("x")

    endpoint.body = {"data": [{"index": 0, "embedding": [0.0] * DIM}]}
    with pytest.raises(EmbedderError, match="returned 1 embeddings for 2 texts"):
        emb.embed(["x", "y"])

    endpoint.body = {"data": [{"index": 5, "embedding": [0.0] * DIM}]}
    with pytest.raises(EmbedderError, match="has index 5"):
        emb.embed_one("x")

    endpoint.body = {"data": [{"index": 0}]}
    with pytest.raises(EmbedderError, match="no 'embedding' list"):
        emb.embed_one("x")

    endpoint.body = None
    endpoint.raw = b"<html>not json</html>"
    with pytest.raises(EmbedderError, match="not JSON"):
        emb.embed_one("x")

    endpoint.raw = b"[1, 2, 3]"
    with pytest.raises(EmbedderError, match="expected an object"):
        emb.embed_one("x")

    endpoint.close()
    with pytest.raises(EmbedderError, match="could not reach"):
        emb.embed_one("x")


# ============================================================================ on the handle


def test_the_handle_embeds_what_it_is_not_given_and_keeps_what_it_is(tmp_path):
    h = HashEmbedder(DIM)
    with Anatid.open(tmp_path / "e.anatid", tenant=1, embedding_dim=DIM, embedder=h) as db:
        assert db.embedder is h
        implicit = db.remember("Bo maintains the ingest service", entities=["Bo"], now=T0)
        explicit = db.remember("given a vector", embedding=vec(0, 1), now=T0)
        stored = db.get(implicit.memory_id, with_embedding=True)
        assert stored is not None and stored.embedding is not None
        assert list(stored.embedding) == pytest.approx(
            h.embed_one("Bo maintains the ingest service"), abs=1e-6
        )
        given = db.get(explicit.memory_id, with_embedding=True)
        assert given is not None and list(given.embedding or ()) == pytest.approx(vec(0, 1))

        # supersede embeds the corrected fact too, so the vector arm can reach it.
        newer = db.supersede(implicit.memory_id, "Cy maintains the ingest service", now=T0)
        fixed = db.get(newer.memory_id, with_embedding=True)
        assert fixed is not None and list(fixed.embedding or ()) == pytest.approx(
            h.embed_one("Cy maintains the ingest service"), abs=1e-6
        )

        # recall embeds the query; an embedding passed by the caller is used as given.
        hits = db.recall("who maintains the ingest service", seed_entity=None)
        assert hits.arms == ("vector", "text")
        assert hits[0].content == "Cy maintains the ingest service"
        assert hits[0].vector_rank == 1
        by_vector = db.recall(embedding=vec(0, 1), seed_entity=None)
        assert by_vector.arms == ("vector",) and by_vector[0].content == "given a vector"

    # No embedder: nothing changes.
    with Anatid.open(tmp_path / "e.anatid", tenant=1, embedding_dim=DIM) as db:
        assert db.embedder is None
        plain = db.remember("no vector here", now=T0)
        row = db.get(plain.memory_id, with_embedding=True)
        assert row is not None and row.embedding is None
        assert db.recall("vector", seed_entity=None).arms == ("text",)


def test_an_embedder_of_the_wrong_dimension_is_refused_at_open(tmp_path):
    path = tmp_path / "dim.anatid"
    with Anatid.open(path, tenant=1, embedding_dim=DIM):
        pass
    with pytest.raises(EmbeddingDimensionError) as exc:
        Anatid.open(path, tenant=1, embedding_dim=DIM, embedder=HashEmbedder(DIM + 1))
    assert exc.value.expected == DIM and exc.value.got == DIM + 1
    # The refused open released the file: the next one works.
    with Anatid.open(path, tenant=1, embedding_dim=DIM, embedder=HashEmbedder(DIM)) as db:
        assert db.embedder is not None


def test_an_embedder_without_dim_is_accepted_and_checked_per_vector():
    class Bare:
        def embed(self, texts):
            return [[1.0] * 3 for _ in texts]

        def embed_one(self, text):
            return [1.0] * 3

    with Anatid.open(":memory:", tenant=1, embedding_dim=DIM, embedder=Bare()) as db:
        with pytest.raises(EmbeddingDimensionError):
            db.remember("three is not eight")


# ============================================================================ MCP configuration


def test_embedder_from_config_reads_the_environment():
    cfg = mcp_embedding.embedder_from_config
    assert cfg(embedding_dim=DIM, env={}) is None
    hashed = cfg(embedding_dim=DIM, env={}, embed_hash=True)
    assert isinstance(hashed, HashEmbedder) and hashed.dim == DIM
    by_env = cfg(embedding_dim=DIM, env={mcp_embedding.ENV_HASH: "1"})
    assert isinstance(by_env, HashEmbedder)
    assert cfg(embedding_dim=DIM, env={mcp_embedding.ENV_HASH: "1"}, embed_hash=False) is None
    assert cfg(embedding_dim=DIM, env={mcp_embedding.ENV_HASH: "off"}) is None

    remote = cfg(
        embedding_dim=16,
        env={
            mcp_embedding.ENV_BASE_URL: "https://api.example.test/v1/",
            mcp_embedding.ENV_MODEL: "text-embedding-3-small",
            mcp_embedding.ENV_API_KEY: "sk-secret",
        },
    )
    assert isinstance(remote, OpenAICompatibleEmbedder)
    assert remote.url == "https://api.example.test/v1/embeddings"
    assert remote.model == "text-embedding-3-small" and remote.dim == 16
    assert remote.api_key == "sk-secret"
    local = cfg(
        embedding_dim=16,
        env={
            mcp_embedding.ENV_BASE_URL: "http://localhost:11434/v1",
            mcp_embedding.ENV_MODEL: "nomic-embed-text",
        },
    )
    assert isinstance(local, OpenAICompatibleEmbedder) and local.api_key is None


def test_half_a_configuration_is_refused():
    cfg = mcp_embedding.embedder_from_config
    with pytest.raises(mcp_embedding.EmbedderConfigError, match=mcp_embedding.ENV_MODEL):
        cfg(embedding_dim=DIM, env={mcp_embedding.ENV_BASE_URL: "http://x/v1"})
    with pytest.raises(mcp_embedding.EmbedderConfigError, match=mcp_embedding.ENV_BASE_URL):
        cfg(embedding_dim=DIM, env={mcp_embedding.ENV_MODEL: "m"})
    with pytest.raises(mcp_embedding.EmbedderConfigError, match="pick one"):
        cfg(
            embedding_dim=DIM,
            env={mcp_embedding.ENV_MODEL: "m", mcp_embedding.ENV_BASE_URL: "http://x/v1"},
            embed_hash=True,
        )
    with pytest.raises(mcp_embedding.EmbedderConfigError, match="not a boolean"):
        cfg(embedding_dim=DIM, env={mcp_embedding.ENV_HASH: "maybe"})
    assert issubclass(mcp_embedding.EmbedderConfigError, ValueError)


def test_describe_embedder_says_kind_model_and_dim_and_never_the_key():
    d = mcp_embedding.describe_embedder
    assert d(None) is None
    assert d(HashEmbedder(DIM)) == {
        "kind": "hash",
        "model": None,
        "dim": DIM,
        "note": "deterministic offline stand-in; similarity means shared words",
    }
    remote = OpenAICompatibleEmbedder("https://api.example.test/v1", "sk-secret", "m", 16)
    described = d(remote)
    assert described == {
        "kind": "openai_compatible",
        "model": "m",
        "dim": 16,
        "url": "https://api.example.test/v1/embeddings",
    }
    assert "sk-secret" not in json.dumps(described)


def test_add_cli_arguments_registers_embed_hash():
    import argparse

    parser = argparse.ArgumentParser()
    mcp_embedding.add_cli_arguments(parser)
    assert parser.parse_args([]).embed_hash is None
    assert parser.parse_args(["--embed-hash"]).embed_hash is True


def test_an_mcp_server_on_an_embedding_handle_stores_vectors_and_runs_the_vector_arm(tmp_path):
    """No tool changes: the handle embeds, so ``remember`` stores a vector and ``recall`` runs
    all three arms and says so in ``arms``."""
    pytest.importorskip("mcp.client", reason="the MCP server needs 'mcp>=2.1'")
    import asyncio

    from mcp.client import Client

    from anatid.integrations.mcp.server import build_server

    embedder = mcp_embedding.embedder_from_config(embedding_dim=DIM, env={}, embed_hash=True)
    with Anatid.open(tmp_path / "mcp.anatid", tenant=3, embedding_dim=DIM, embedder=embedder) as db:
        server = build_server(db)

        async def go():
            async with Client(server) as client:
                await client.call_tool(
                    "remember",
                    {
                        "content": "Bo maintains the ingest service",
                        "entities": ["Bo", "ingest service"],
                    },
                )
                await client.call_tool(
                    "remember",
                    {
                        "content": "Kestrel owns the ingest service",
                        "entities": ["Kestrel", "ingest service"],
                    },
                )
                return await client.call_tool(
                    "recall",
                    {
                        "query": "who maintains the ingest service",
                        "seed_entity": "ingest service",
                    },
                )

        result = asyncio.run(go())
        assert result.is_error is False
        out = result.structured_content
        assert out is not None
        assert set(out["arms"]) >= {"vector", "text", "graph"}
        assert out["hits"][0]["content"] == "Bo maintains the ingest service"
        assert "vector" in out["hits"][0]["sources"]
        stored = db.get(int(out["hits"][0]["memory_id"]), with_embedding=True)
        assert stored is not None and stored.embedding is not None
