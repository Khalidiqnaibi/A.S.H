"""
Per-tool "shape" files: one JSON file per tool, in tools/shapes/, that is
the single source of truth for BOTH:

  1. the classifier's intent (tag + description + example patterns --
     same fields classify_and_route() already reads from intents.json)
  2. which function actually runs when that intent fires (the registry
     entry)

Adding a tool = adding one file to tools/shapes/. No other file needs
editing, and the classifier automatically learns to recognize it
because classification.py merges these shapes into its intent catalog
alongside the legacy intents.json entries.

Shape file format (tools/shapes/<tool_name>.json):

{
  "tag": "system_stats",                 // intent label; must be unique
  "description": "...",                 // used to embed the intent
  "patterns": ["...", "..."],            // example phrases (also embedded)
  "tool": {
    "module": "tools.extra_tools",       // python module path
    "function": "system_stats_tool"      // function inside that module
  }
}

`description` is optional if `patterns` is present (classification.py
already falls back to building a description from patterns -- see
_intent_description_from_block). Provide at least one of the two.
"""

from __future__ import annotations
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("ash.tools.shape_loader")

DEFAULT_SHAPES_DIR = Path(__file__).parent


def load_all_shapes(shapes_dir: str | Path = DEFAULT_SHAPES_DIR) -> List[Dict[str, Any]]:
    """Read every *.json file in the shapes directory. Bad files are
    logged and skipped, not fatal -- one broken shape shouldn't take
    down the whole tool system."""
    shapes_dir = Path(shapes_dir)
    if not shapes_dir.exists():
        logger.warning("Shapes directory not found: %s", shapes_dir)
        return []

    shapes = []
    for path in sorted(shapes_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("tag"):
                logger.warning("Shape file %s missing required 'tag' -- skipping", path.name)
                continue
            data["_source_file"] = str(path)
            shapes.append(data)
        except Exception:
            logger.exception("Failed to parse shape file %s -- skipping", path.name)
    logger.info("Loaded %d tool shape(s) from %s", len(shapes), shapes_dir)
    return shapes


def shapes_max_mtime(shapes_dir: str | Path = DEFAULT_SHAPES_DIR) -> float:
    """Newest mtime across all shape files -- lets classification.py
    know when it needs to rebuild its embedding cache."""
    shapes_dir = Path(shapes_dir)
    if not shapes_dir.exists():
        return 0.0
    mtimes = [p.stat().st_mtime for p in shapes_dir.glob("*.json")]
    return max(mtimes) if mtimes else 0.0


def shapes_to_intent_blocks(shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert shapes into the {"tag", "description", "patterns"} block
    format classification.py's EmbeddingCatalog already expects -- so
    they merge seamlessly with legacy intents.json entries."""
    blocks = []
    for s in shapes:
        blocks.append({
            "tag": s["tag"],
            "description": s.get("description", ""),
            "patterns": s.get("patterns", []),
        })
    return blocks


def register_shape_tools(shapes: List[Dict[str, Any]]):
    """Import each shape's target function and register it into the
    shared tool registry under its own tag as the sole intent.

    A shape file's "tool" block is OPTIONAL: omit it entirely when the
    tag is provided by an MCP server instead of a native function (see
    tools/MCP_client.py). In that case this shape contributes ONLY the
    classifier intent (tag/description/patterns) -- mcp_client.py is
    what actually registers the executable ToolEntry for that tag, and
    load_mcp_servers() must run AFTER register_shape_tools() so its
    registration is the one that survives. This is how an MCP tool
    becomes reachable by the classifier at all: without a matching
    shape file (tool-less or not), classify_and_route() never learns
    to recognize its tag, and it can only ever be called directly.
    """
    from tools.registry import REGISTRY, ToolEntry

    for s in shapes:
        tag = s["tag"]
        tool_ref = s.get("tool")
        if not tool_ref:
            logger.info("Shape '%s' has no 'tool' block -- classifier-only entry, "
                        "expecting an MCP server (or other external registrant) "
                        "to provide the executable ToolEntry for this tag.", tag)
            continue
        try:
            module = importlib.import_module(tool_ref["module"])
            func = getattr(module, tool_ref["function"])
        except Exception:
            logger.exception("Failed to load tool for shape '%s' (%s.%s)",
                              tag, tool_ref.get("module"), tool_ref.get("function"))
            continue

        REGISTRY.register(ToolEntry(
            name=tag,
            intents=[tag],
            func=func,
            description=s.get("description", ""),
            kind="native",
        ), overwrite=True)