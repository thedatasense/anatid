"""Erasure hooks for the tables anatid's bundled integrations write.

Since 0.1.1 the machinery lives in :mod:`anatid.erasure`, because ``forget(hard=True)`` itself
has to reach the bundled integration tables on every handle -- not only the handle that
constructed an ``AnatidSession`` or ``RunStateStore`` and registered hooks on it.  This module
re-exports the same names so existing imports keep working; see :mod:`anatid.erasure` for how a
row is matched and what the bundled-table pass covers.
"""

from __future__ import annotations

from ..erasure import (
    BUNDLED_INTEGRATION_TABLES,
    TEXT_TYPE_PREFIXES,
    TableErasureHook,
    memory_needles,
    purge_bundled_tables,
    purge_rows_containing,
    register_table_erasure_hooks,
    text_columns,
)

__all__ = [
    "BUNDLED_INTEGRATION_TABLES",
    "TEXT_TYPE_PREFIXES",
    "TableErasureHook",
    "memory_needles",
    "purge_bundled_tables",
    "purge_rows_containing",
    "register_table_erasure_hooks",
    "text_columns",
]
