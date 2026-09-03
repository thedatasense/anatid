"""Integrations with agent frameworks and protocols.

* :mod:`anatid.integrations.openai_agents` -- a ``Session`` implementation, approval-gated
  memory tools and durable human-in-the-loop run-state storage for the OpenAI Agents SDK
  (``pip install "anatid[agents]"``).
* :mod:`anatid.integrations.mcp` -- the MCP server (``pip install "anatid[mcp]"``; the console
  script ``anatid-mcp`` points at ``anatid.integrations.mcp.server:main``).  Its raw-SQL escape
  hatch is opt-in and fails closed; see :mod:`anatid.integrations.mcp.sqlgate`.
* :mod:`anatid.integrations.erasure` -- the erasure hooks that make ``forget(hard=True)`` reach
  the tables the integrations above write (transcripts, parked run states).  It imports nothing
  optional, so it is safe to import without either framework installed.

Nothing here is imported by ``import anatid``: every integration pulls in an optional dependency,
so you import the sub-package you actually want.
"""
