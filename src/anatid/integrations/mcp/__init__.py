"""anatid's Model Context Protocol server.

``anatid.integrations.mcp`` exposes anatid's memory verbs as MCP tools so that any MCP client
-- Claude Code, Claude Desktop, Cursor -- gets a persistent, bitemporal, graph-shaped memory
backed by one embedded DuckDB file.

Built against **mcp 2.x** (``mcp.server.mcpserver.MCPServer``, the class mcp 1.x called
``FastMCP``).  Run it with the ``anatid-mcp`` console script, or::

    from anatid import Anatid
    from anatid.integrations.mcp import build_server

    db = Anatid.open("memory.anatid", tenant=0)
    build_server(db).run("stdio")

The third-party ``mcp`` distribution is an optional dependency: ``pip install anatid[mcp]``.
Importing this subpackage without it raises :class:`ImportError` with that instruction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # The names __getattr__ below hands out lazily, spelled out for type checkers (which do not
    # run __getattr__) and for `from anatid.integrations.mcp import build_server` completions.
    from .server import (
        InsecureTransport,
        ServerConfig,
        build_server,
        check_transport_security,
        main,
    )
    from .sqlgate import ENFORCEMENT, SqlGateway, SqlNotAllowed, SqlTimeout

__all__ = [
    "build_server", "main", "ServerConfig", "SqlGateway", "SqlNotAllowed", "SqlTimeout",
    "ENFORCEMENT", "InsecureTransport", "check_transport_security",
]


def __getattr__(name: str):
    # Lazy so that `import anatid.integrations.mcp` gives a readable error, rather than a bare
    # ModuleNotFoundError, when the optional `mcp` distribution is not installed.
    if name in __all__:
        try:
            from . import server as _server
            from . import sqlgate as _sqlgate
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
            if (exc.name or "").split(".")[0] in ("mcp", "mcp_types"):
                raise ImportError(
                    "anatid's MCP server needs the 'mcp' package (>=2.1): "
                    "pip install 'anatid[mcp]'  (or: pip install 'mcp>=2.1')"
                ) from exc
            raise
        ns = {
            "build_server": getattr(_server, "build_server", None),
            "main": getattr(_server, "main", None),
            "ServerConfig": getattr(_server, "ServerConfig", None),
            "InsecureTransport": getattr(_server, "InsecureTransport", None),
            "check_transport_security": getattr(_server, "check_transport_security", None),
            "SqlGateway": getattr(_sqlgate, "SqlGateway", None),
            "SqlNotAllowed": getattr(_sqlgate, "SqlNotAllowed", None),
            "SqlTimeout": getattr(_sqlgate, "SqlTimeout", None),
            "ENFORCEMENT": getattr(_sqlgate, "ENFORCEMENT", None),
        }
        return ns[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
