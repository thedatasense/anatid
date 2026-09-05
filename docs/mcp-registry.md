# Publishing to the MCP Registry

The Model Context Protocol (MCP) Registry hosts metadata, not artifacts. The package itself stays on
PyPI. Publishing takes a manifest, a marker in the README that proves ownership, and one login.

## What is already in the repository

`mcp-registry/server.json` is the manifest. Its `name` is `io.github.thedatasense/anatid`, which is
the namespace GitHub authentication grants to the `thedatasense` account. The package entry points at
the `anatid` distribution on PyPI with the stdio transport and documents `--db`, `ANATID_DB`,
`ANATID_TENANT` and `ANATID_SOCKET`.

The README ends with `<!-- mcp-name: io.github.thedatasense/anatid -->`. The registry checks the
PyPI long description for that string before it accepts the manifest, so the marker has to be in the
version of the README that is on PyPI, which means it ships with the next release.

## Publishing

The version in `server.json` must match a version that exists on PyPI. Bump both together.

```bash
brew install mcp-publisher                # or the tarball from the registry's GitHub releases
cd mcp-registry
mcp-publisher login github                # opens a device-code flow in the browser
mcp-publisher publish
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.thedatasense/anatid"
```

`login` needs a person at the keyboard because it authenticates through GitHub's device flow. The
publish step itself takes a few seconds.

## After publishing

Clients that read the registry can install the server by name. A user who prefers the manual route
still uses the config block in `docs/mcp.md`. Bump `version` in `server.json` with every PyPI
release that changes the MCP surface and publish again; the registry keeps the history.
