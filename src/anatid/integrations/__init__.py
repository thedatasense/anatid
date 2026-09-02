"""Integrations with agent frameworks and protocols.

* :mod:`anatid.integrations.openai_agents` -- a ``Session`` implementation, approval-gated
  memory tools and durable human-in-the-loop run-state storage for the OpenAI Agents SDK
  (``pip install "anatid[agents]"``).
* :mod:`anatid.integrations.mcp` -- the MCP server (owned by another agent; the console script
  ``anatid-mcp`` points at ``anatid.integrations.mcp.server:main``).

Nothing here is imported by ``import anatid``: every integration pulls in an optional dependency,
so you import the sub-package you actually want.
"""
