from datetime import datetime
from typing import Any

def log_tool_use(
    state: dict,
    tool_name: str,
    tool_input: Any,
    tool_output: Any,
    agent: str = "unknown"
):
    print("time", datetime.now().isoformat(),
        "agent", agent,
        "tool", tool_name,
        "input", tool_input,
        "output", tool_output, flush=True
        )
    state.setdefault("tool_log", []).append({
        "time": datetime.now().isoformat(),
        "agent": agent,
        "tool": tool_name,
        "input": tool_input,
        "output": tool_output,
    })


