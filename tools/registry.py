"""
Generic tool registry for ASH.

Goal: adding a new tool -- native Python OR an MCP server's tool --
should never require touching ash.py's routing logic again.

Two ways to register a tool:

1. Native Python tool, via decorator:

    from tools.registry import tool

    @tool(name="calculator", intents=["calc", "calculate", "math"])
    def calculator_tool(query: str) -> dict:
        ...
        return {"ok": True, "result": ...}

2. MCP server tool: handled automatically by mcp_client.py, which
   discovers a connected server's tools and registers each one here
   under intents drawn from the server's own tool descriptions.

ash.py's _deterministic_execute() no longer has a hardcoded if/elif
chain per tool -- it just does:

    entry = REGISTRY.match(cmd_tag)
    if entry:
        result = entry.run(query)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ash.tools.registry")


@dataclass
class ToolEntry:
    name: str
    # Intent tags this tool answers to (matched against the classifier's
    # cmd_tag, case-insensitively). e.g. ["calc", "calculate", "math"]
    intents: List[str]
    func: Callable[..., Any]
    description: str = ""
    # "native" = plain python function, "mcp" = proxied to an MCP server
    kind: str = "native"
    source_server: Optional[str] = None  # MCP server name, if kind=="mcp"

    def run(self, query: str) -> Dict[str, Any]:
        try:
            out = self.func(query)
            if isinstance(out, dict):
                out.setdefault("ok", True)
                return out
            return {"ok": True, "result": out}
        except Exception as e:
            logger.exception("Tool '%s' failed", self.name)
            return {"ok": False, "error": str(e)}


class ToolRegistry:
    def __init__(self):
        self._by_name: Dict[str, ToolEntry] = {}
        self._intent_index: Dict[str, str] = {}  # intent tag -> tool name

    def register(self, entry: ToolEntry, overwrite: bool = False):
        if entry.name in self._by_name and not overwrite:
            logger.warning("Tool '%s' already registered; skipping", entry.name)
            return
        self._by_name[entry.name] = entry
        for intent in entry.intents:
            key = intent.strip().lower()
            if key in self._intent_index and self._intent_index[key] != entry.name:
                logger.warning(
                    "Intent '%s' already mapped to '%s'; '%s' overrides it",
                    key, self._intent_index[key], entry.name,
                )
            self._intent_index[key] = entry.name
        logger.info("Registered tool '%s' (%s) intents=%s", entry.name, entry.kind, entry.intents)

    def unregister_server(self, server_name: str):
        """Drop all tools that came from a given MCP server (e.g. on disconnect)."""
        dead = [n for n, e in self._by_name.items() if e.source_server == server_name]
        for n in dead:
            entry = self._by_name.pop(n)
            for intent in entry.intents:
                key = intent.strip().lower()
                if self._intent_index.get(key) == n:
                    del self._intent_index[key]
        if dead:
            logger.info("Unregistered %d tool(s) from server '%s'", len(dead), server_name)

    def match(self, cmd_tag: Optional[str]) -> Optional[ToolEntry]:
        if not cmd_tag:
            return None
        return self._by_name.get(self._intent_index.get(cmd_tag.strip().lower()))

    def all_intents(self) -> List[str]:
        return list(self._intent_index.keys())

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": e.name, "kind": e.kind, "intents": e.intents, "description": e.description}
            for e in self._by_name.values()
        ]


# Single process-wide registry instance
REGISTRY = ToolRegistry()


def tool(name: str, intents: List[str], description: str = ""):
    """Decorator for registering a native Python tool in one line."""
    def _wrap(func: Callable[..., Any]):
        REGISTRY.register(ToolEntry(
            name=name, intents=intents, func=func,
            description=description or (func.__doc__ or "").strip(),
            kind="native",
        ))
        return func
    return _wrap