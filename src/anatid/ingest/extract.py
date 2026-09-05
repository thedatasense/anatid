"""Extractors: turn a piece of text into a proposed :class:`~anatid.ingest.patch.MemoryPatch`.

An :class:`Extractor` is anything with ``extract(text, *, existing) -> MemoryPatch``.
``existing`` is the current facts the pipeline found about the entities the text mentions, so
the extractor can propose a correction of one of them instead of a duplicate; each is a
:class:`KnownFact`, a :class:`~anatid.Memory` that also carries the names it is about.

Two implementations ship:

* :class:`OpenAICompatibleExtractor` asks a chat model at any OpenAI-compatible endpoint
  (OpenAI, OpenRouter, a local server) for one JSON object against
  :data:`~anatid.ingest.patch.PATCH_JSON_SCHEMA`, and parses the reply tolerantly: code fences
  and prose around the object are stripped, an entry the parser cannot read is dropped with a
  note rather than failing the batch.  The ``openai`` package is imported only when the class
  is instantiated without a ``client``.
* :class:`ScriptedExtractor` returns patches you wrote ahead of time, in order.  It is what the
  tests and the offline example use, and it records every call so a test can check what the
  pipeline handed it.

Neither one writes anything.  The pipeline (:mod:`anatid.ingest.pipeline`) resolves the
proposal against the database and applies it.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol, runtime_checkable

from ..errors import AnatidError
from ..integrations.wire import wire_id
from ..types import Memory
from .patch import PATCH_JSON_SCHEMA, MemoryPatch

__all__ = [
    "Extractor",
    "ExtractionError",
    "KnownFact",
    "ExtractCall",
    "ScriptedExtractor",
    "OpenAICompatibleExtractor",
    "SYSTEM_PROMPT",
    "parse_json_object",
    "render_existing",
]


class ExtractionError(AnatidError):
    """The model's reply could not be read as a patch.  ``raw`` is what it said."""

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True, slots=True)
class KnownFact(Memory):
    """A current :class:`~anatid.Memory` plus the entity names it is ABOUT.

    A ``Memory`` on its own does not carry its entities; an extractor needs them to reuse the
    names the graph already has and to say which fact a correction replaces.  It is a
    ``Memory``, so any code that takes one takes this.
    """

    about: tuple[str, ...] = ()

    @classmethod
    def of(cls, memory: Memory, about: Iterable[str]) -> KnownFact:
        values = {f.name: getattr(memory, f.name) for f in fields(Memory)}
        return cls(**values, about=tuple(about))


@runtime_checkable
class Extractor(Protocol):
    """What the pipeline needs from an extractor."""

    def extract(self, text: str, *, existing: Sequence[Memory]) -> MemoryPatch: ...


# --------------------------------------------------------------------------- scripted


@dataclass(frozen=True, slots=True)
class ExtractCall:
    """One call a :class:`ScriptedExtractor` received."""

    text: str
    existing: tuple[Memory, ...]


class ScriptedExtractor:
    """Return prepared patches in order; for tests and offline demonstrations.

    Each item is a :class:`~anatid.ingest.patch.MemoryPatch` or a callable
    ``(text, existing) -> MemoryPatch``.  A patch whose ``source_text`` is empty gets the text
    it was asked about.  ``calls`` records every request, newest last.
    """

    def __init__(
        self,
        patches: Iterable[MemoryPatch | Callable[[str, Sequence[Memory]], MemoryPatch]],
    ) -> None:
        self._queue: deque[Any] = deque(patches)
        self.calls: list[ExtractCall] = []

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def extract(self, text: str, *, existing: Sequence[Memory]) -> MemoryPatch:
        self.calls.append(ExtractCall(text, tuple(existing)))
        if not self._queue:
            raise LookupError(
                f"ScriptedExtractor has no patch left for text {text[:60]!r}; "
                f"it was given {len(self.calls) - 1}"
            )
        item = self._queue.popleft()
        patch = item(text, existing) if callable(item) else item
        if not isinstance(patch, MemoryPatch):
            raise TypeError(f"ScriptedExtractor items must yield a MemoryPatch, got {patch!r}")
        if not patch.source_text:
            patch = patch.replace(source_text=text)
        return patch


# --------------------------------------------------------------------------- OpenAI-compatible

SYSTEM_PROMPT = """\
You maintain a graph memory for an assistant. You read one note and answer with a memory patch:
one JSON object and nothing else, no prose, no code fence, matching this JSON schema:

{schema}

How to fill it in:
- A fact is one durable statement the note makes, written as a standalone declarative sentence
  that names its subjects, so it can be read on its own later. List every entity the fact is
  about (people, teams, systems, places, things) by name.
- Use the existing entity names given with the note whenever the note refers to the same
  thing. When the note uses a different name for an existing entity, add an entity_aliases
  entry mapping the note's name to the existing name.
- When the note changes something an existing fact states, do not add a contradicting fact.
  Add a correction that copies the existing fact's id into old_memory_id exactly as given (it
  is a string), gives the new content, and lists the entities of the corrected fact. Move the
  relationships that change: put the old one in remove_relations and the new one in
  add_relations.
- A relation joins two entities with a short rel_kind such as leads, maintains, owns,
  depends_on, member_of, reports_to. Add one for each relationship the note states.
- Quote the shortest span of the note that supports each entry in its evidence field.
- Do not invent anything the note does not say. Skip greetings, questions and speculation.
- If the note contains nothing durable, return an object whose arrays are empty.
"""


def render_existing(existing: Sequence[Memory], *, limit: int) -> str:
    """The existing facts as the model sees them: id (a decimal string), content, entities."""
    if not existing:
        return "No existing facts about the entities in this note."
    lines = ["Existing current facts about entities this note mentions:"]
    for memory in list(existing)[:limit]:
        about = getattr(memory, "about", ())
        tail = f"  (about: {', '.join(about)})" if about else ""
        kind = f" [{memory.kind}]" if memory.kind and memory.kind != "fact" else ""
        lines.append(f'- memory_id "{wire_id(memory.memory_id)}"{kind}: {memory.content}{tail}')
    if len(existing) > limit:
        lines.append(f"- ... and {len(existing) - limit} more not shown")
    names: list[str] = []
    for memory in existing:
        for name in getattr(memory, "about", ()):
            if name not in names:
                names.append(name)
    if names:
        lines.append("Existing entity names: " + ", ".join(names))
    return "\n".join(lines)


_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Read the one JSON object a model was asked for, tolerating what models add around it.

    Strips a code fence, then takes the outermost ``{ ... }`` and parses it.  Raises
    :class:`ExtractionError` carrying the raw text when no object can be read.
    """
    raw = text
    if not isinstance(raw, str) or not raw.strip():
        raise ExtractionError("the model returned no content", raw=raw)
    body = raw.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise ExtractionError("the model's reply contains no JSON object", raw=raw) from None
        try:
            data = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"the model's JSON could not be parsed: {exc}", raw=raw) from exc
    if not isinstance(data, dict):
        raise ExtractionError(
            f"the model returned a JSON {type(data).__name__}, not an object", raw=raw
        )
    return data


class OpenAICompatibleExtractor:
    """Extract a patch with a chat model at any OpenAI-compatible endpoint.

    ``base_url`` and ``api_key`` go to ``openai.OpenAI``; pass ``client`` instead to supply
    your own (anything with ``chat.completions.create``), which is how the tests run this class
    with no network.  ``extra_body`` is forwarded on every request, for provider options such
    as OpenRouter's ``{"reasoning": {"enabled": True}}``.  ``response_format`` is
    ``"json_object"`` (the default, widely supported), ``"json_schema"`` (strict, where the
    provider supports it) or ``None`` to rely on the prompt alone.

    ``existing`` is rendered into the prompt, at most ``max_existing`` facts, with ids as
    decimal strings.  ``last_reply`` keeps the model's most recent raw reply for debugging.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        extra_body: Mapping[str, Any] | None = None,
        temperature: float | None = 0.0,
        response_format: str | None = "json_object",
        max_existing: int = 40,
        timeout: float = 120.0,
        system_prompt: str | None = None,
    ) -> None:
        if response_format not in (None, "json_object", "json_schema"):
            raise ValueError(
                f"response_format must be 'json_object', 'json_schema' or None, "
                f"got {response_format!r}"
            )
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "OpenAICompatibleExtractor needs the openai package: pip install openai, "
                    "or pass client="
                ) from exc
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.client = client
        self.model = model
        self.extra_body = dict(extra_body) if extra_body else None
        self.temperature = temperature
        self.response_format = response_format
        self.max_existing = int(max_existing)
        self.system_prompt = system_prompt or SYSTEM_PROMPT.format(
            schema=json.dumps(PATCH_JSON_SCHEMA, indent=1)
        )
        self.last_reply: str | None = None

    def messages(self, text: str, existing: Sequence[Memory]) -> list[dict[str, str]]:
        """The chat messages one call sends, for inspection and tests."""
        user = render_existing(existing, limit=self.max_existing) + "\n\nThe note:\n\n" + text
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def request(self, text: str, existing: Sequence[Memory]) -> dict[str, Any]:
        """The keyword arguments handed to ``chat.completions.create``."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": self.messages(text, existing)}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        elif self.response_format == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "memory_patch", "schema": PATCH_JSON_SCHEMA},
            }
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        return kwargs

    def extract(self, text: str, *, existing: Sequence[Memory]) -> MemoryPatch:
        response = self.client.chat.completions.create(**self.request(text, existing))
        content = _reply_text(response)
        self.last_reply = content
        data = parse_json_object(content)
        return MemoryPatch.from_dict(data, source_text=text)


def _reply_text(response: Any) -> str:
    """The assistant text of a chat completion, from the SDK object or a plain dict."""
    if isinstance(response, Mapping):
        choices = response.get("choices") or []
        if not choices:
            raise ExtractionError("the model returned no choices", raw=json.dumps(response))
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ExtractionError("the model returned no choices", raw=repr(response))
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, list):  # content parts
        content = "".join(str(getattr(p, "text", None) or p.get("text", "")) for p in content)
    return str(content or "")
