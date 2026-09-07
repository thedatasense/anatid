# Publishing to the MCP Registry

The Model Context Protocol (MCP) Registry hosts metadata, not artifacts. The package itself stays on
PyPI. Publishing takes a manifest, a marker in the README that proves ownership, and one login.

## What is already in the repository

`mcp-registry/server.json` is the manifest. Its `name` is `io.github.thedatasense/anatid`, which is
the namespace GitHub authentication grants to the `thedatasense` account. The package entry points at
the `anatid` distribution on PyPI with the stdio transport and documents `--db`, `ANATID_DB`,
`ANATID_TENANT` and `ANATID_SOCKET`.

A client that installs from the registry does not run `anatid-mcp`. It assembles
`uvx <runtimeArguments> anatid@<version> <packageArguments>` (VS Code's MCP management service does
exactly that), and `uvx` runs the executable named like the package. Two things in the manifest make
that command work, and both are held to the release by `tests/test_mcp_registry.py`:

- `runtimeHint` is `uvx` and `runtimeArguments` carries `--with anatid[mcp]==<version>`, because the
  base wheel does not require `mcp` and the server cannot import without it. Without the argument
  the install succeeds and the launch fails with `No module named 'mcp'`.
- `[project.scripts]` in `pyproject.toml` has an `anatid` entry, the MCP server under the package's
  own name, next to `anatid-mcp`. Without it `uvx anatid` fails with "an executable named `anatid`
  is not provided by package `anatid`".

`registryBaseUrl` is deliberately absent: VS Code passes it to `uvx` as `--index-url` verbatim, and
`https://pypi.org` is not a simple-index URL. The same command works by hand, with or without the
registry, and needs no absolute path in a client's config:

```bash
uvx --with "anatid[mcp]" anatid --db ~/.anatid/memory.anatid
```

The README ends with `<!-- mcp-name: io.github.thedatasense/anatid -->`. The registry checks the
PyPI long description for that string before it accepts the manifest, so the marker has to be in the
version of the README that is on PyPI; the 0.4.1 long description carries it.

## Publishing

The version in `server.json` must match a version that exists on PyPI, in three places: the top-level
`version`, the package `version`, and the `anatid[mcp]==` pin in `runtimeArguments`. The registry
test fails when any of them drifts from `anatid.__version__`, so run `pytest tests/test_mcp_registry.py`
after a bump. The `anatid` console script first ships with the release after 0.4.1; a manifest that
names an older release launches nothing, whatever it says.

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
