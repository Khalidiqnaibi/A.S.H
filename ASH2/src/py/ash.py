"""
ASH — brain-first assistant runtime.

Usage:
    from ASH import ash
    response = ash.run("what time is it?")
"""

import sys,os
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# LLM message helpers (used for rendering prompts)
try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    # Fallback: build simple message container if langchain_core unavailable
    class SystemMessage:
        def __init__(self, content): self.content = content
    class HumanMessage:
        def __init__(self, content): self.content = content

# Import router & tools (make sure these modules exist at these paths)
from ASH2.tools.classification import classify_and_route, classify_intent, sentiment_tool
from ASH2.tools.registry import CommandRegistry 
from ASH2.tools.lesstools import date_time_tool, calculator_tool, retrieve_tool  # factory.build_retriever
from ASH2.tools.emo import init_emo, get_emo, update_emo, reset_emo, EmotionState

# AgentSystem / LLM wrapper (your existing wrapper)
from AgentSystem import mistral  # your mistral wrapper


##############
#~ Immortal ~#
##############

#Ash attempt num 5 
USER = "Immortal" #"khalid afif sami iqnaibi"
  
# kparser=Sen()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_OPENROUTER_MODEL = os.getenv("MISTRAL_OPENROUTER_MODEL")

# Simple runtime state type
AshState = Dict[str, Any]

DEFAULT_LLM_TEMPERATURE = 0.9

# Global shared state (singleton-ish) — persists across runs in same process
ash_state: AshState = {
    "history": [],        # list of {"role": "user"|"ash"|"tool", "text": "...", "time": ISO}
    "emotions": EmotionState().as_dict(),
    "tool_log": [],       # list of tool usage records
    "input": "",
    "res": "",
    "client_time": None,  # optional ISO timestamp set from client context
}

def _now_iso():
    return datetime.now().isoformat()

def _print_log(*args, **kwargs):
    print("[ASH]", *args, file=sys.stderr, flush=True, **kwargs)

class ASH:
    """
    ASH main runtime. Instantiate once and call run(query, context).
    """

    def __init__(self, llm=None, llm_temperature: float = DEFAULT_LLM_TEMPERATURE, name: str = "A.S.H"):
        self.name = name
        self.user = USER
        self.start_time = datetime.now()
        self.status = "online"
        # LLM wrapper (your mistral wrapper). If none passed, create one.
        if llm is None:
            try:
                self.llm = mistral.MistralLLM(
                    temperature=llm_temperature,
                    openrouter_key=OPENROUTER_API_KEY, 
                    openrouter_model=MISTRAL_OPENROUTER_MODEL
                )
            except Exception as e:
                _print_log("Warning: failed to instantiate MistralLLM wrapper:", e)
                self.llm = None
        else:
            self.llm = llm

        # optional local caches / settings
        self.intent_threshold = None  # keep default from classification module
        _print_log(f"{self.name} initialized (LLM present: {bool(self.llm)})")
        self.registry = CommandRegistry()

        # register tools
        self.registry.register_tool("date_time_tool", lambda _: date_time_tool())
        self.registry.register_tool("calculator_tool", calculator_tool)
        self.registry.register_tool("retriever", lambda q: retrieve_tool(q, top=5, llm=self.llm))

        self.registry.load_commands()
        _print_log("ASH online. Commands loaded:", list(self.registry.commands))

    # -----------------------
    # State helpers
    # -----------------------
    def _append_history(self, role: str, text: str):
        ash_state.setdefault("history", [])
        ash_state["history"].append({
            "role": role,
            "text": text,
            "time": _now_iso()
        })

    def _format_history_for_prompt(self, max_turns: int = 6) -> str:
        """
        Returns recent conversation history as a readable transcript
        for the LLM. Only includes user + ash messages.
        """
        history = ash_state.get("history", [])

        # Filter only conversational roles
        convo = [h for h in history if h["role"] in ("user", "ash")]

        # Take last N turns (user+ash pairs)
        convo = convo[-max_turns * 2 :]

        lines = []
        for h in convo:
            role = "User" if h["role"] == "user" else self.name
            lines.append(f"{role}: {h['text']}")

        return "\n".join(lines) if lines else "No prior conversation."

    def _append_tool_log(self, tool_name: str, tool_input: Any, tool_output: Any):
        ash_state.setdefault("tool_log", [])
        entry = {
            "time": _now_iso(),
            "agent": self.name,
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output
        }
        ash_state["tool_log"].append(entry)
        # immediate debug print
        _print_log("[TOOL LOG]", tool_name, "input=", str(tool_input)[:200], "output=", str(tool_output)[:200])

    # -----------------------
    # LLM rendering (narrator)
    # -----------------------
    def _render_with_llm(self, query: str, facts: Dict[str, Any]) -> str:
        """
        Ask the LLM to format a natural assistant response, using facts verbatim.
        LLM must not call tools or change state.
        """
        history_block = self._format_history_for_prompt()
        # Compose a safe system + human prompt
        system_content = (
            f"You are {self.name}, a loyal personal assistant. "
            "Use the facts below in a human readable format where applicable. Do NOT invent facts."
            "answer the query then say a small sentence"
        )
        human_content = (
            "Conversation so far:\n"
            f"{history_block}\n\n"
            f"User query: {query}\n\n"
            "Facts (use if present):\n"
            f"{json.dumps(facts, indent=2)}\n\n"
            "Emotional snapshot (internal state):\n"
            f"{json.dumps(ash_state.get('emotions', {}), indent=2)}\n\n"
            "Respond like your emotional state and in a small paragraph."
        )

        # Create messages if langchain_core is present
        try:
            messages = [SystemMessage(content=system_content), HumanMessage(content=human_content)]
        except Exception:
            messages = [{"role": "system", "content": system_content}, {"role": "user", "content": human_content}]

        if not self.llm:
            # No LLM available: fallback to returning facts or a short template
            _print_log("LLM missing — returning facts or fallback text.")
            if facts.get("tool_output"):
                return facts["tool_output"]
            return f"(No LLM) I received your query: {query}"

        # Try multiple call patterns for different LLM wrappers
        try:
            # preferred: .invoke(messages) returning object with .content
            resp = getattr(self.llm, "invoke", None)
            if callable(resp):
                llm_resp = self.llm.invoke(messages)
                content = getattr(llm_resp, "content", str(llm_resp))
                return content
        except Exception as e:
            _print_log("LLM.invoke failed:", e)

        try:
            # try .chat(messages)
            chat_fn = getattr(self.llm, "chat", None)
            if callable(chat_fn):
                llm_resp = self.llm.chat(messages)
                content = getattr(llm_resp, "content", str(llm_resp))
                return content
        except Exception as e:
            _print_log("LLM.chat failed:", e)

        try:
            # try .generate / .complete
            gen_fn = getattr(self.llm, "generate", None) or getattr(self.llm, "complete", None)
            if callable(gen_fn):
                llm_resp = gen_fn(messages)
                content = getattr(llm_resp, "content", str(llm_resp))
                return content
        except Exception as e:
            _print_log("LLM.generate/complete failed:", e)

        try:
            # last resort: call LLM object if it's callable
            if callable(self.llm):
                llm_resp = self.llm(human_content)
                return str(llm_resp)
        except Exception as e:
            _print_log("LLM callable failed:", e)

        # If all else fails, fallback to tool output or a safe message
        _print_log("All LLM invocation attempts failed; falling back to facts or fallback text.")
        if facts.get("tool_output"):
            return facts["tool_output"]
        return "Sorry — I couldn't generate a response right now."

    # -----------------------
    # Public API: run
    # -----------------------

    def run(self, query: str) -> str:
        """
        Execute full pipeline:
          1) deterministic routing/classification
          2) execute tools in code when required
          3) update state/tool_log/history
          4) render final answer via LLM (narrator)
        """
        ash_state["input"] = query
        self._append_history("user", query)

        route = classify_and_route(query)
        cmd_id = route.get("command")
        score = route.get("score", 0.0)

        tool_output = None
        if cmd_id and score >= 0.3:
            tool_output = self.registry.execute(cmd_id, query)
        else:
            # fallback retriever
            try:
                tool = self.registry.tools.get("retriever")
                docs = tool(query) if tool else None
                if docs:
                    formatted = "\n".join([f"- {d.page_content} (source: {d.metadata.get('source','unknown')})" for d in docs])
                else:
                    formatted = "No relevant knowledge found."
                cmd_id = "conversation"
                tool_output = formatted
                self._append_tool_log("retriever", query, formatted)
                self._append_history("tool", f"retriever -> {formatted}")
            except Exception as e:
                _print_log("Retriever fallback error:", e)
                tool_output = "No information available."

        facts = {
            "command": cmd_id,
            "confidence": score,
            "tool_output": tool_output
        }

        reply = self._render_with_llm(query, facts)
        ash_state["res"] = reply
        self._append_history("ash", reply)

        return reply

    # utility: pretty print current state (developer helper)
    def status_info(self) -> Dict[str, Any]:
        uptime = datetime.now() - self.start_time
        return {
            "name": self.name,
            "uptime_seconds": int(uptime.total_seconds()),
            "status": self.status,
            "last_input": ash_state.get("input"),
            "last_output_preview": str(ash_state.get("res", ""))[:200]
        }

ash = ASH()

# quick local test when run directly
if __name__ == "__main__":
    print("Starting ASH interactive (type 'exit' to quit).", file=sys.stderr)
    while True:
        try:
            q = input("You: ")
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        # provide any context (for local test we pass no client_time)
        resp = ash.run(q)
        print("\nASH:", resp, "\n")
