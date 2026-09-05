"""``python -m anatid.server`` runs the operator CLI.

The same entry point as the ``anatid-server`` console script, for the cases where the script is
not on PATH: a virtual environment that was not activated, a container that runs the interpreter
directly, and a systemd unit that would rather name an absolute interpreter than a wrapper.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
