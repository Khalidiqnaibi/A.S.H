import os, json
from typing import Dict, Callable, Any

COMMAND_PATH = os.getenv(
    "ASH_COMMANDS_PATH",
    os.path.join(os.getcwd(), "ASH2", "commands")
)

class CommandRegistry:
    def __init__(self):
        self.commands: Dict[str, dict] = {}
        self.tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, fn: Callable):
        self.tools[name] = fn

    def load_commands(self):
        self.commands.clear()
        for fname in os.listdir(COMMAND_PATH):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(COMMAND_PATH, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cmd in data.get("commands", []):
                    self.commands[cmd["id"]] = cmd

    def get(self, command_id: str):
        return self.commands.get(command_id)

    def execute(self, command_id: str, query: str):
        cmd = self.get(command_id)
        if not cmd:
            return None
        tool_name = cmd.get("tool")
        tool = self.tools.get(tool_name)
        if not tool:
            return None
        return tool(query)
