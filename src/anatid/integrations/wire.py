"""Ids on the wire: decimal strings in JSON, ``int`` inside Python.

anatid mints 63-bit ids (:mod:`anatid.ids`), and a 63-bit integer does not survive a JSON round
trip through a client that parses numbers as IEEE-754 doubles.  JavaScript is the case that
decides it: ``Number.MAX_SAFE_INTEGER`` is ``2**53 - 1``, so an id sent as a JSON *number* comes
back changed.

.. code-block:: console

    $ node -e 'console.log(JSON.parse(String.raw`{"memory_id": 883768514279557120}`).memory_id)'
    883768514279557100

Nothing raises.  The client now holds an id that refers to no row, and every ``get``, ``forget``
and ``supersede`` it makes with that id is wrong.  Every consumer of an MCP server or of an
OpenAI Agents SDK tool is either JavaScript or something that hands its output to JavaScript, so
the boundary is where this has to be fixed.

The contract, at every boundary in :mod:`anatid.integrations`:

* ids **out** are decimal strings -- ``"883768514279557120"``, never ``883768514279557120``;
* ids **in** are a decimal string *or* an integer, because a client written against the older
  shape has to keep working, and are an ``int`` before any anatid verb sees them;
* a tool's JSON schema declares an id parameter as ``"type": "string"`` and says why, so a model
  reading the schema emits a string in the first place.

:class:`WireId` and :class:`WireEntityRef` are that contract as type annotations.  Both
integration boundaries build their tool schemas with pydantic -- the ``mcp`` server generates
one from the tool signature, and so does the Agents SDK -- and pydantic asks a type for its own
schema, so annotating a parameter is all it takes: string in the schema, string or integer
accepted, ``int`` in the function body.

It lives here, next to its two consumers, because the contract is the same one at both
boundaries and neither integration package may own the other's code.  Importing it costs
nothing: pydantic is reached only from inside the two ``__get_pydantic_*`` hooks, which
pydantic itself calls, so ``anatid.integrations.openai_agents.AnatidSession`` still imports
with neither the Agents SDK nor pydantic installed, exactly as it did before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema

__all__ = [
    "JS_MAX_SAFE_INTEGER",
    "WIRE_ID_DESCRIPTION",
    "WIRE_ENTITY_DESCRIPTION",
    "WireId",
    "WireEntityRef",
    "coerce_id",
    "coerce_entity_ref",
    "wire_id",
    "wire_ids",
    "wire_unsafe_ints",
    "is_wire_safe_int",
]

#: The largest integer a JSON number survives in a client that parses numbers as doubles --
#: JavaScript's ``Number.MAX_SAFE_INTEGER``.  anatid ids are far above it: the time bits alone
#: put a 2026 id near ``8.8e17``.
JS_MAX_SAFE_INTEGER = 2**53 - 1

WIRE_ID_DESCRIPTION = (
    'An anatid id as a decimal string, e.g. "883768514279557120". anatid ids are 63-bit '
    "integers and JSON numbers lose precision above 2**53 (JavaScript's "
    "Number.MAX_SAFE_INTEGER), so ids travel as strings and come back unchanged. A plain "
    "integer is still accepted."
)

WIRE_ENTITY_DESCRIPTION = (
    'An entity by name, e.g. "Ada Lovelace", or an existing entity_id as a decimal string, '
    'e.g. "883768514279557120". Ids are strings because JSON numbers lose precision above '
    "2**53 (JavaScript's Number.MAX_SAFE_INTEGER). A plain integer id is still accepted."
)


def is_wire_safe_int(value: Any) -> bool:
    """True when ``value`` is not an integer that a JSON number would corrupt.

    Booleans are integers to Python and are never ids, so they are safe by definition.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return True
    return -JS_MAX_SAFE_INTEGER <= value <= JS_MAX_SAFE_INTEGER


def coerce_id(value: Any) -> int:
    """Decode one id argument.  Accepts a decimal string or an integer; returns an ``int``.

    ``True`` is an ``int`` to Python and is never an id, so a boolean is rejected rather than
    quietly read as 1.  A float is rejected too: ``8.8377183962144358e+17`` is already the
    corruption this module exists to prevent, and silently accepting it would hide it.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected an id, got the boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 10)
        except ValueError:
            raise ValueError(
                f"expected an anatid id as a decimal string such as "
                f'"883768514279557120"; got {value!r}'
            ) from None
    raise ValueError(
        f'expected an anatid id as a decimal string such as "883768514279557120"; '
        f"got {type(value).__name__} {value!r}"
    )


def coerce_entity_ref(value: Any) -> str:
    """Decode one entity argument: a name, or an id as a decimal string or an integer.

    The result is always a string, which is what the callers' name-or-id resolvers take.  An
    integer becomes its decimal string so that ``entity=883768514279557120`` and
    ``entity="883768514279557120"`` mean the same entity rather than a row and a name.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected an entity name or id, got the boolean {value!r}")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    raise ValueError(
        f"expected an entity name, or an entity_id as a decimal string; "
        f"got {type(value).__name__} {value!r}"
    )


def wire_id(value: int | None) -> str | None:
    """Encode one id for JSON.  ``None`` stays ``None``; anything else becomes a decimal string."""
    return None if value is None else str(int(value))


def wire_ids(values: Iterable[int]) -> list[str]:
    """Encode a sequence of ids for JSON."""
    return [str(int(v)) for v in values]


def wire_unsafe_ints(value: Any) -> Any:
    """Recursively replace integers a JSON number would corrupt with decimal strings.

    For payloads whose shape is not known ahead of time -- the rows an arbitrary ``SELECT``
    returns -- where the alternative is handing back a silently wrong id.  Integers inside the
    safe range are left as numbers, so a ``count(*)`` is still a number; only the values that
    could not survive change type.  Where the shape *is* known, use :func:`wire_id` instead, so
    a field's type does not depend on the magnitude of the value in it.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if is_wire_safe_int(value) else str(value)
    if isinstance(value, dict):
        return {k: wire_unsafe_ints(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire_unsafe_ints(v) for v in value]
    return value


class WireId(int):
    """Annotation for a tool parameter that carries an anatid id.

    Declares ``{"type": "string"}`` in the generated JSON schema, accepts a decimal string or an
    integer from the client, and hands the tool body a plain ``int``.  It subclasses ``int`` so
    that a checker sees an ``int`` where the parameter is passed on to an anatid verb.
    """

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # Imported here, not at module scope: pydantic is only ever present when something is
        # asking for a schema, and this module is imported on paths that have neither
        # integration's dependencies.
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            coerce_id,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema()
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string", "description": WIRE_ID_DESCRIPTION}


class WireEntityRef(str):
    """Annotation for a tool parameter that carries an entity by name or by id.

    Same schema story as :class:`WireId` -- ``{"type": "string"}``, with an integer still
    accepted -- but the tool body gets a ``str``, because a name and an id share this argument
    and the callers' resolvers take the string form of both.
    """

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            coerce_entity_ref,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema()
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string", "description": WIRE_ENTITY_DESCRIPTION}
