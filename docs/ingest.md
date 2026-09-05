# Ingestion: from text to reviewed changes in the graph

The core verbs take one fact or one edge at a time. `anatid.ingest` takes text. A note, a chat
message or a document goes in; a reviewed set of changes to the memory graph comes out, applied
in one transaction with the raw text stored as an episode first. The package is additive: it
calls the same verbs you would call by hand, and every step says what it did.

```python
from anatid import Anatid
from anatid.ingest import OpenAICompatibleExtractor, ingest

extractor = OpenAICompatibleExtractor(model="gpt-4o-mini", api_key=key)

with Anatid.open("team.anatid", tenant=1) as db:
    receipt = ingest(
        db,
        "Cy took over the ingest service from Bo this week.",
        extractor=extractor,
        writer="notes-bot",
        source="standup/2026-06-15",
    )
    print(receipt.describe())
```

`examples/ingest_notes.py` runs the whole flow on three notes without a key.

## The pipeline

`ingest(db, text, *, extractor, writer, source=None, review=None, now=None, tenant=None,
embedder=None)` runs four steps. Each is a public function you can call on its own.

1. **Context.** `existing_context(db, text)` finds the current facts about the entities the
   text names. An entity counts as named when its canonical key occurs in the text as a whole
   word, so `Bo` does not match `Bob`. The text-search arm of `recall` adds what it finds. Each
   fact is a `KnownFact`: a `Memory` that also carries the entity names it is about, so the
   extractor can reuse the graph's names and say which fact a correction replaces.
2. **Extraction.** `extractor.extract(text, existing=...)` returns a `MemoryPatch`. The
   extractor proposes; it writes nothing.
3. **Preparation.** `prepare(patch, db)` resolves the proposal against the database:
   - `resolve_entities` applies the patch's aliases and spells every name the way the graph
     already spells it. The database folds case and whitespace on its own (`entity_key`), so
     this changes what the reviewer reads rather than which entity is written.
   - `resolve_corrections` gives every correction the id of the memory it replaces. A
     correction that names its memory by text becomes one by id when exactly one current
     memory reads that way. A correction whose memory cannot be found, or is no longer
     current, is downgraded to a new fact and noted.
   - `dedupe` drops a fact the graph already holds (same folded content, same entities), a
     fact proposed twice, a relation a current edge already holds, and a relation removal
     that has no current edge to close. Every drop is a note.
4. **Review and apply.** The optional `review` callable receives the prepared patch and
   returns the patch to apply, edited or not, or `None` to decline. Then
   `MemoryPatch.apply(db, writer=...)` commits it.

`propose(db, text, extractor=...)` is steps 1 to 3 without the apply. Use it when approval
happens somewhere else: serialise the patch with `to_json()`, show it, and later apply
`MemoryPatch.from_json(text).apply(db, writer=...)`.

## The patch

A `MemoryPatch` is a frozen dataclass. It is data; nothing happens until `apply`.

| field | type | meaning |
| --- | --- | --- |
| `source_text` | `str` | the text the patch was derived from; stored as the episode |
| `add_facts` | `tuple[AddFact, ...]` | new memories: `content`, `entities`, `kind` (default `"fact"`), `confidence`, `span` |
| `corrections` | `tuple[Correction, ...]` | replace one current memory: `new_content`, `old_id` or `old_text`, `entities` (`None` inherits), `kind` (`None` inherits), `confidence`, `span` |
| `add_relations` | `tuple[Relation, ...]` | `RELATES_TO` edges to open: `src`, `dst`, `rel_kind`, `span` |
| `remove_relations` | `tuple[Relation, ...]` | `RELATES_TO` edges to close, in either direction, narrowed by `rel_kind` when given |
| `entity_aliases` | `tuple[Alias, ...]` | `name` in the text means the entity `canonical` |
| `notes` | `tuple[str, ...]` | what shaped the patch: a dropped duplicate, a downgraded correction, an entry the parser skipped |

Every operation may carry a `Span`, the half-open character range of `source_text` it came
from. `Span.locate(quote, text)` builds one from a quoted sentence.

`describe()` renders the patch as a diff for a person: `+` adds, `~` corrects, `-` closes, `=`
declares an alias, each followed by its evidence when it has one, then the notes.

```
memory patch: 1 fact, 1 correction, 2 relation(s) added, 1 relation(s) removed
source: "Bo moved to the platform group. Cy took over the ingest service from Bo this week." (82 chars)
  + fact        "Bo works in the platform group"  about: Bo, platform group  [0:30] "Bo moved to the platform group"
  ~ correction  memory 883924049843532800 "Bo maintains the ingest service"
                -> "Cy maintains the ingest service"  about: Cy, ingest service
  - relation    Bo -maintains-> ingest service
  + relation    Cy -maintains-> ingest service
  + relation    Bo -member_of-> platform group
```

`replace(**changes)` returns an edited copy, which is how a review hook edits.

### JSON

`to_json()` and `from_json()` carry a patch across a process boundary. Every id is a decimal
string, for the reason `anatid.integrations.wire` gives: a 63-bit id does not survive a
JavaScript JSON parser as a number. `from_dict` is tolerant. It reads what `to_dict` writes
and what a model writes against `PATCH_JSON_SCHEMA`: `facts` for `add_facts`, `aliases` for
`entity_aliases`, `old_id` for `old_memory_id`, an `evidence` quote in place of a `span`, a
single string where a list of entities is expected, an id as a string or an integer. An entry
it cannot read is skipped and the reason goes into `notes`.

### Applying

`patch.apply(db, *, writer, episode=None, source=None, now=None, tenant=None, embedder=None)`
runs in one transaction:

1. the text (`episode`, default `source_text`) is stored as an episode of kind `ingest`, with
   `source` as its label;
2. aliases are applied to every entity reference; when an alias is already an entity of its
   own, an `alias_of` edge is written between it and the canonical entity so graph recall
   crosses between the two spellings;
3. each `AddFact` becomes `remember(...)`;
4. each `Correction` becomes `supersede(...)`, which closes the old memory's version and
   records a `SUPERSEDES` edge;
5. each removal closes the current edges between its two entities (`unrelate`);
6. each addition opens an edge (`relate`).

Every memory and edge carries the new episode's id. An error anywhere rolls back everything,
the episode included. `embedder`, a callable from text to a vector, gives each new memory an
embedding so the vector arm of `recall` can find it. Without one the verbs decide: a handle
opened with `Anatid.open(embedder=...)` embeds every memory it is not given a vector for, and a
handle without one writes no embedding.

The `PatchReceipt` says what landed: `episode_id`, `memories_created`, `memories_closed`,
`corrections` as `(old_id, new_id)` pairs, `relations_opened`, `relations_closed`, `aliases`,
and the patch it came from. `to_dict()` writes ids as strings.

A correction applied without `prepare` must name its memory: by `old_id`, or by an `old_text`
that exactly one current memory matches. None raises `NotFoundError`; several raise
`ValidationError`. The pipeline resolves these ahead of time so the reviewer sees the id.

## The review hook

`review` has the same shape as the approval gate in the tool integrations: a callable that
sees the proposed change and decides. It receives the prepared patch and returns one of three
things.

```python
def review(patch: MemoryPatch) -> MemoryPatch | None:
    print(patch.describe())
    answer = input("apply [y], edit out facts [e], decline [n]? ")
    if answer == "n":
        return None                                    # nothing is written
    if answer == "e":
        return patch.replace(add_facts=())             # an edited copy is applied
    return patch                                       # applied as proposed
```

Declining returns `None` from `ingest` and writes nothing, not even the episode. A patch with
no operations is still applied when the hook lets it through: the episode is stored, so the
record shows the note was read even when it changed nothing.

## Extractors

An extractor is anything with `extract(text, *, existing) -> MemoryPatch`.

`OpenAICompatibleExtractor(model=..., base_url=None, api_key=None, client=None,
extra_body=None, temperature=0.0, response_format="json_object", max_existing=40)` asks a
chat model at any OpenAI-compatible endpoint for one JSON object against `PATCH_JSON_SCHEMA`.
The prompt carries the existing facts with their ids as decimal strings and their entity
names, and asks for a correction rather than a duplicate when the text changes one of them.
The reply is parsed tolerantly: a code fence or prose around the object is stripped; an entry
that cannot be read is dropped with a note. `extra_body` is forwarded on every request, which
is where OpenRouter's `{"reasoning": {"enabled": True}}` goes. Pass `client` to supply your own
object with `chat.completions.create`; the tests use a fake. `last_reply` holds the most recent
raw reply.

`ScriptedExtractor(patches)` returns prepared patches in order and records every call in
`calls`, with the `existing` facts it was handed. It is what the tests and the offline example
use.

## Through the tools

The same pipeline is reachable from both integrations, and both keep the review step.

The OpenAI Agents SDK gets a tenth tool, `anatid_ingest`, when `create_memory_tools` is given
an `extractor`. It is a write and waits for approval like the other writes. Called with
`dry_run=true` it proposes the patch and returns `diff` and `patch_json` without writing, and a
dry run never waits, because it changes nothing. Called again with that `patch_json` (edited or
not) as `patch`, it applies exactly that patch without asking the model a second time. An
optional `ingest_review` hook sees the prepared patch before it is applied and can edit or
decline it; a declined patch is reported as `"declined": true` and writes nothing.

```python
from anatid.ingest import OpenAICompatibleExtractor
from anatid.integrations.openai_agents import create_memory_tools

tools = create_memory_tools(db, extractor=OpenAICompatibleExtractor(model=..., api_key=...))
```

The MCP server registers `ingest` and `apply_patch` when `ANATID_EXTRACT_BASE_URL` and
`ANATID_EXTRACT_MODEL` are set (`ANATID_EXTRACT_API_KEY` and `ANATID_EXTRACT_REASONING=1` are
optional), or when `build_server(db, extractor=...)` is given one. `ingest(text)` proposes and
returns `patch_id`, `diff` and `patch`; `apply_patch(patch_id)` commits, with an edited `patch`
to change it first; declining is not calling `apply_patch`. A failed apply keeps the proposal
pending so it can be edited and tried again. Both tools need the embedded handle, because
`MemoryPatch.apply` runs several verbs in one transaction on the file's own connection, so an
`anatid-mcp` that talks to a server over a socket refuses the extraction settings at startup.
[`mcp.md`](mcp.md) has the details.

## What the pipeline does not do

It does not decide what is true. The extractor proposes, the preparation step reconciles the
proposal with what the graph holds and writes down every change it made, and the review hook
decides. It does not merge two entities: an alias links them with an edge and rewrites the
references in the patch, and both entity rows remain. It does not split a long document into
episodes; pass one unit of evidence per call. It does not write embeddings unless given an
`embedder`, or unless the handle was opened with one.
