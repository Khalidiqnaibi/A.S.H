"""
MCP integration for ASH.

Lets ASH connect to any MCP server (stdio or SSE-based) and use its
tools exactly like a native deterministic tool -- no code changes to
ash.py's routing logic required.

Config-driven: define servers in mcp_servers.json, e.g.

{
  "servers": [
    {
      "name": "filesystem",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/pi/shared"],
      "intent_overrides": {
        "read_file": ["read_file", "open_file", "show_file"]
      }
    },
    {
      "name": "weather-mcp",
      "transport": "sse",
      "url": "http://localhost:8931/sse"
    }
  ]
}

At startup, call `load_mcp_servers("mcp_servers.json")` once. Each
server's tools get pulled via list_tools() and registered into
tools.registry.REGISTRY automatically, with intents derived from the
tool's own name/description (or overridden per-server in config).

Adding a brand-new MCP tool later means editing mcp_servers.json,
not ash.py.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import REGISTRY, ToolEntry

logger = logging.getLogger("ash.tools.mcp_client")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    logger.warning("mcp package not installed -- run `pip install mcp` to enable MCP tools")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


class MCPServerConnection:
    """
    Owns one MCP server's connection on a dedicated background event loop
    thread, so ASH's main (sync) code can call MCP tools with a plain
    blocking function call -- keeps the rest of ASH's deterministic,
    synchronous style intact instead of forcing async/await everywhere.
    """

    def __init__(self, config: Dict[str, Any]):
        self.name = config["name"]
        self.config = config
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional["ClientSession"] = None
        self._ctx_manager = None
        self._ready = threading.Event()
        self._closed = False

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"mcp-{self.name}")
        self._thread.start()
        if not self._ready.wait(timeout=15):
            raise TimeoutError(f"MCP server '{self.name}' did not initialize in time")

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_serve())

    async def _connect_and_serve(self):
        try:
            transport = self.config.get("transport", "stdio")
            if transport == "stdio":
                params = StdioServerParameters(
                    command=self.config["command"],
                    args=self.config.get("args", []),
                    env=self.config.get("env"),
                )
                self._ctx_manager = stdio_client(params)
            elif transport == "sse":
                self._ctx_manager = sse_client(self.config["url"])
            else:
                raise ValueError(f"Unknown MCP transport: {transport}")

            async with self._ctx_manager as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._register_tools_sync(await session.list_tools())
                    self._ready.set()
                    # keep the session alive until told to stop
                    while not self._closed:
                        await asyncio.sleep(1)
        except Exception as e:
            logger.exception("MCP server '%s' connection failed", self.name)
            self._ready.set()  # unblock start(), caller sees no tools registered

    def _register_tools_sync(self, tools_result):
        overrides = self.config.get("intent_overrides", {})
        for t in tools_result.tools:
            tool_name = f"mcp_{self.name}_{t.name}"
            intents = overrides.get(t.name) or [_slugify(t.name), t.name.lower()]

            def make_runner(mcp_tool_name: str):
                def _runner(query: str):
                    return self.call_tool(mcp_tool_name, query)
                return _runner

            REGISTRY.register(ToolEntry(
                name=tool_name,
                intents=intents,
                func=make_runner(t.name),
                description=t.description or "",
                kind="mcp",
                source_server=self.name,
            ), overwrite=True)

    def call_tool(self, tool_name: str, query: str) -> Dict[str, Any]:
        """Blocking call from ASH's sync code into the async MCP session."""
        if self._session is None or self._loop is None:
            return {"ok": False, "error": f"MCP server '{self.name}' not connected"}

        async def _call():
            # Most MCP tools take a single free-text argument named
            # "query" or "input" -- servers with richer schemas should
            # set intent_overrides + a custom native wrapper tool instead.
            try:
                result = await self._session.call_tool(tool_name, arguments={"query": query})
            except Exception:
                result = await self._session.call_tool(tool_name, arguments={"input": query})
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(texts) if texts else str(result.content)

        future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
        try:
            return {"ok": True, "result": future.result(timeout=30)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def stop(self):
        self._closed = True


_CONNECTIONS: Dict[str, MCPServerConnection] = {}


def load_mcp_servers(config_path: str = "mcp_servers.json"):
    """
    Call once at ASH startup. Connects to every server listed in the
    config file and registers their tools into the shared registry.
    Missing config file or missing `mcp` package = no-op, ASH still
    runs fine with just its native tools.
    """
    if not _MCP_AVAILABLE:
        return
    path = Path(config_path)
    if not path.exists():
        logger.info("No %s found -- skipping MCP tool loading", config_path)
        return

    servers = json.loads(path.read_text()).get("servers", [])
    for server_cfg in servers:
        name = server_cfg["name"]
        try:
            conn = MCPServerConnection(server_cfg)
            conn.start()
            _CONNECTIONS[name] = conn
            logger.info("Connected MCP server '%s'", name)
        except Exception:
            logger.exception("Failed to connect MCP server '%s'", name)


def shutdown_mcp_servers():
    for conn in _CONNECTIONS.values():
        conn.stop()
    _CONNECTIONS.clear()