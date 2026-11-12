"""
src/memory/paths.py

Where the memory files live. By default: entities.json, episodic_memory.jsonl and core_memory.json in the working
directory, as always. With ASH_MEMORY_DIR set, in that folder instead -- which is how the test runner gives every test
its own copy, so no test (not even one that is killed half-way) can write to the user's own memory.
"""

from __future__ import annotations

import os
from typing import Optional


def memory_file(name: str, path: Optional[str] = None) -> str:
    """`path` if one was given; else `name` inside ASH_MEMORY_DIR if that is set; else `name` (the working directory)."""
    if path:
        return path
    folder = os.environ.get("ASH_MEMORY_DIR", "").strip()
    return os.path.join(folder, name) if folder else name
