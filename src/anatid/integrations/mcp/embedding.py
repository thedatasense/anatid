"""The MCP server's embedder, from configuration.

The MCP tools take no vectors from the model that calls them: ``remember`` has no ``embedding``
argument on purpose (a model must not be asked to invent one) and ``recall`` accepts one only
for clients that already have it.  So on a default server the vector arm never runs and every
recall is text plus graph.  This module is the switch that changes that, and the operator
throws it in configuration: name an embedding endpoint and every memory the server writes is
embedded, and every query is embedded before it is searched.

Configuration, environment first because that is what a client's JSON config block can set:

``ANATID_EMBED_BASE_URL``   an OpenAI-compatible API root, e.g. ``https://api.openai.com/v1``
                           or ``http://localhost:11434/v1``
``ANATID_EMBED_MODEL``      the embedding model that endpoint serves
``ANATID_EMBED_API_KEY``    sent as a bearer token when set; local servers usually need none
``ANATID_EMBED_HASH``       ``1`` selects :class:`~anatid.embed.HashEmbedder`, the deterministic
                           offline stand-in, which is for demos and tests and not for quality.
                           The ``--embed-hash`` flag says the same thing.

:func:`embedder_from_config` turns that into an :class:`~anatid.embed.Embedder`, or None when
nothing is configured, and the caller hands it to ``Anatid.open(embedder=...)``::

    embedder = embedder_from_config(embedding_dim=cfg.embedding_dim, embed_hash=args.embed_hash)
    db = Anatid.open(
        cfg.resolved_db(), tenant=cfg.tenant, embedding_dim=cfg.embedding_dim, embedder=embedder
    )

From then on the handle does the work: ``db.remember(content)`` stores a vector and
``db.recall(query)`` runs the vector arm, and the ``arms`` field of every recall result says so
(``["vector", "text", "graph"]`` rather than ``["text", "graph"]``).  Nothing in the tool list
changes.  The embedder belongs to the process that owns the file: with the server profile
(``anatid-server``) it has to be configured there, because an
:class:`~anatid.server.client.AnatidClient` forwards verbs and does not embed.

Half a configuration is refused rather than guessed: a base URL with no model, or a model with
no base URL, raises :class:`EmbedderConfigError`, and so does asking for the hash stand-in and
an endpoint at the same time.  The dimension is the database's: the embedder is built for
``embedding_dim`` and :meth:`anatid.Anatid.open` checks the two agree.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from anatid.embed import Embedder, HashEmbedder, OpenAICompatibleEmbedder

__all__ = [
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_HASH",
    "ENV_MODEL",
    "EmbedderConfigError",
    "add_cli_arguments",
    "describe_embedder",
    "embedder_from_config",
]

ENV_BASE_URL = "ANATID_EMBED_BASE_URL"
ENV_MODEL = "ANATID_EMBED_MODEL"
ENV_API_KEY = "ANATID_EMBED_API_KEY"
ENV_HASH = "ANATID_EMBED_HASH"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


class EmbedderConfigError(ValueError):
    """The embedding configuration names part of an endpoint, or two embedders at once."""


def _flag(value: str | None) -> bool:
    if value is None:
        return False
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise EmbedderConfigError(f"{ENV_HASH}={value!r} is not a boolean (use 1 or 0)")


def embedder_from_config(
    *,
    embedding_dim: int,
    env: Mapping[str, str] | None = None,
    embed_hash: bool | None = None,
) -> Embedder | None:
    """The embedder the configuration asks for, or None when it asks for none.

    ``embedding_dim``
        The database's ``FLOAT[N]`` width; the embedder is built to produce vectors of it.
    ``env``
        The environment to read; ``os.environ`` when omitted.
    ``embed_hash``
        The ``--embed-hash`` flag.  ``True`` selects the hash stand-in, ``False`` refuses it
        even if ``ANATID_EMBED_HASH`` is set, ``None`` lets the environment decide.

    Returns :class:`~anatid.embed.HashEmbedder` for the stand-in,
    :class:`~anatid.embed.OpenAICompatibleEmbedder` for an endpoint, or None.  Raises
    :class:`EmbedderConfigError` for half an endpoint or for both kinds at once.
    """
    if env is None:
        import os

        env = os.environ
    base_url = (env.get(ENV_BASE_URL) or "").strip()
    model = (env.get(ENV_MODEL) or "").strip()
    api_key = (env.get(ENV_API_KEY) or "").strip() or None
    use_hash = _flag(env.get(ENV_HASH)) if embed_hash is None else bool(embed_hash)

    if use_hash:
        if base_url or model:
            raise EmbedderConfigError(
                f"both the hash stand-in ({ENV_HASH} / --embed-hash) and an embedding endpoint "
                f"({ENV_BASE_URL} / {ENV_MODEL}) are configured; pick one"
            )
        return HashEmbedder(int(embedding_dim))
    if not base_url and not model:
        return None
    if not base_url or not model:
        missing = ENV_MODEL if not model else ENV_BASE_URL
        raise EmbedderConfigError(
            f"an embedding endpoint needs both {ENV_BASE_URL} and {ENV_MODEL}; {missing} is not set"
        )
    return OpenAICompatibleEmbedder(base_url, api_key, model, int(embedding_dim))


def describe_embedder(embedder: Embedder | None) -> dict[str, Any] | None:
    """What the ``stats`` tool can say about the server's embedder: kind, model, dimension.
    Never the key."""
    if embedder is None:
        return None
    if isinstance(embedder, HashEmbedder):
        return {
            "kind": "hash",
            "model": None,
            "dim": embedder.dim,
            "note": "deterministic offline stand-in; similarity means shared words",
        }
    if isinstance(embedder, OpenAICompatibleEmbedder):
        return {
            "kind": "openai_compatible",
            "model": embedder.model,
            "dim": embedder.dim,
            "url": embedder.url,
        }
    return {"kind": type(embedder).__name__, "model": None, "dim": getattr(embedder, "dim", None)}


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``--embed-hash`` on the ``anatid-mcp`` argument parser.

    The endpoint settings stay environment-only: a base URL, a model and a key are exactly what
    an MCP client's config block carries, and a key on a command line is visible in ``ps``.
    """
    parser.add_argument(
        "--embed-hash",
        action="store_true",
        default=None,
        help=(
            f"embed with the deterministic offline stand-in (env {ENV_HASH}=1): every remember "
            f"stores a vector and every recall runs the vector arm, with similarity meaning "
            f"shared words. For demos and tests. For real embeddings set {ENV_BASE_URL}, "
            f"{ENV_MODEL} and optionally {ENV_API_KEY} instead."
        ),
    )
