"""The MCP Registry manifest, the console script it launches, and the version they share.

``mcp-registry/server.json`` is what a registry client installs from.  VS Code's
``mcpManagementService``, the widest-deployed consumer, turns a pypi package entry into
``uvx <runtimeArguments> <identifier>@<version> <packageArguments>``, and uvx runs the executable
NAMED LIKE THE PACKAGE.  So the manifest works only while three things agree:

* ``[project.scripts]`` in pyproject.toml has an ``anatid`` entry that starts the MCP server;
* the manifest's ``--with anatid[mcp]==X`` runtime argument names the release being published,
  because the base wheel does not require ``mcp`` and the server cannot import without it;
* the manifest's versions are ``anatid.__version__`` (CI already holds pyproject to it).

A product review found the manifest launching a package that had no executable of its own name
and no path to the extra, which a clean ``uvx anatid`` reproduced.  These tests keep that from
coming back, and hold the README to the ownership marker the registry checks on PyPI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anatid

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "mcp-registry" / "server.json"
PYPROJECT = REPO / "pyproject.toml"
README = REPO / "README.md"

SERVER_NAME = "io.github.thedatasense/anatid"
MCP_MAIN = "anatid.integrations.mcp.server:main"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def console_scripts() -> dict[str, str]:
    """``[project.scripts]`` as pyproject.toml spells it; a regex so Python 3.10 needs no tomllib."""
    text = PYPROJECT.read_text(encoding="utf-8")
    section = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[)", text, re.MULTILINE | re.DOTALL)
    assert section is not None, "pyproject.toml has no [project.scripts] table"
    return dict(re.findall(r'^([\w.-]+)\s*=\s*"([^"]+)"', section.group(1), re.MULTILINE))


def test_manifest_versions_are_the_package_version():
    m = manifest()
    (package,) = m["packages"]
    assert m["version"] == anatid.__version__
    assert package["version"] == anatid.__version__
    assert m["name"] == SERVER_NAME
    assert package["registryType"] == "pypi" and package["identifier"] == "anatid"
    assert package["transport"] == {"type": "stdio"}


def test_manifest_launches_the_package_with_the_mcp_extra_of_the_same_release():
    (package,) = manifest()["packages"]
    assert package["runtimeHint"] == "uvx", "the registry's runner for pypi packages"
    withs = [
        a for a in package["runtimeArguments"] if a["type"] == "named" and a["name"] == "--with"
    ]
    assert len(withs) == 1, "exactly one --with, carrying the extra"
    assert withs[0]["value"] == f"anatid[mcp]=={anatid.__version__}", (
        "the extra must be pinned to the release the manifest publishes"
    )
    # VS Code passes registryBaseUrl to uvx as --index-url verbatim, and https://pypi.org is
    # not a simple-index URL: leaving it out is what makes the install resolve.
    assert "registryBaseUrl" not in package


def test_the_package_provides_an_executable_named_like_itself():
    scripts = console_scripts()
    assert scripts["anatid-mcp"] == MCP_MAIN
    assert scripts.get("anatid") == MCP_MAIN, (
        "uvx anatid@<version> runs the executable called `anatid`; the registry manifest needs it"
    )


def test_the_readme_carries_the_ownership_marker_the_registry_checks():
    text = README.read_text(encoding="utf-8")
    # The registry looks for `mcp-name: <server name>` in the PyPI long description, followed
    # by a boundary (whitespace, a tag or `-->`), never glued to punctuation.
    found = re.findall(rf"mcp-name:\s*{re.escape(SERVER_NAME)}(?=\s|-->|<)", text)
    assert found, "README.md must carry `<!-- mcp-name: io.github.thedatasense/anatid -->`"
    assert f"mcp-name: {SERVER_NAME}." not in text
