<p><a href="../README.md"><img src="../assets/brand/anatid-logo.png" alt="anatid" width="280" height="96"></a></p>

# Documentation

Local memory for artificial intelligence (AI) agents. Keep source records and earlier claims in one DuckDB file.

The [Cedar manufacturing walkthrough](../README.md#medical-device-manufacturing-a-withdrawn-test)
follows a withdrawn test through a lot review. The [visual demo](../examples/procedural_studio/README.md)
shows the procedure repair and its earlier revision. For direct Python calls, use the
[quickstart](../README.md#five-minutes-to-a-working-memory).

## Build with anatid

| Guide | What it covers |
| --- | --- |
| [Examples](../examples/README.md) | Manufacturing walkthroughs and integration examples |
| [Ingest notes](ingest.md) | Propose, review, and apply facts, relations, and corrections |
| [Procedural graphs](procedural-graphs.md) | Store directed procedures, validate changes, and replay history |
| [Model Context Protocol (MCP) integration](mcp.md) | Give an assistant access to memory with approval-gated writes |
| [Shared memory server](server.md) | Run several clients against one memory; operate and monitor it |
| [MCP Registry](mcp-registry.md) | Manifest, installation, and publishing |

## Understand the system

| Guide | What it covers |
| --- | --- |
| [Architecture](architecture.md) | Storage, time, provenance, retrieval, and transaction contracts |
| [Quality evaluation](quality.md) | Answer-quality experiments, methods, and limitations |
| [Benchmarks](benchmarks.md) | Performance measurements and reproduction commands |
| [Optional extension](extension.md) | Build and use the DuckDB traversal accelerator |
| [Derived indexes](design/derived-index-framework.md) | Index generations, journals, and consistency |
| [Roadmap](roadmap.md) | Shipped capabilities and planned work |

## Contribute and share

[Contributing and releases](../CONTRIBUTING.md) · [Brand assets](branding.md) ·
[GitHub releases](https://github.com/thedatasense/anatid/releases) ·
[PyPI](https://pypi.org/project/anatid/)
