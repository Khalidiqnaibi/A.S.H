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

# Default LLM settings (adjust env / config if you prefer)
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
    # Router + executor
    # -----------------------
    def _deterministic_execute(self, query: str) -> Dict[str, Any]:
        """
        Use classify_and_route (embedding router) to decide what to run.
        Execute tools in code (no LLM decision).
        Returns a dict: { intent, intent_score, command, command_score, tool_used, tool_output }
        """
        _print_log("Routing query:", query)
        # classify_and_route returns a dict: intent, intent_score, command, command_score
        try:
            route = classify_and_route(query)
        except Exception as e:
            _print_log("Router failed:", e)
            route = {"intent": "conversation", "intent_score": 0.0, "command": None, "command_score": 0.0}

        intent = route.get("intent", "conversation")
        result = {
            "intent": intent,
            "intent_score": route.get("intent_score", 0.0),
            "command": route.get("command"),
            "command_score": route.get("command_score", 0.0),
            "tool_used": None,
            "tool_output": None,
            "raw_route": route
        }

        # Handle commands deterministically
        if intent:
            cmd_tag = intent
            _print_log("Command intent detected:", cmd_tag, "score:", result["command_score"])

            # time / date commands
            if cmd_tag and cmd_tag.lower() in ("time", "date", "datetime" ,"get_time"):
                # prefer client time if provided in context
                tool_out = date_time_tool()
                result["tool_used"] = "date_time_tool"
                result["tool_output"] = tool_out
                self._append_tool_log("date_time_tool","", tool_out)
                # append history
                self._append_history("user", query)
                self._append_history("tool", f"date_time_tool -> {tool_out}")
                return result

            # calculator / math commands
            if cmd_tag and cmd_tag.lower() in ("calc", "calculate", "math", "compute"):
                # crude extraction: pass whole string to calculator tool which will safe-calc or error
                tool_out = calculator_tool(query)
                result["tool_used"] = "calculator_tool"
                result["tool_output"] = tool_out
                self._append_tool_log("calculator_tool", query, tool_out)
                self._append_history("user", query)
                self._append_history("tool", f"calculator_tool -> {tool_out}")
                return result

            # Add more command→tool mappings here as needed
            _print_log("No deterministic tool mapped for command tag:", cmd_tag)
            return result

        # Handle questions -> use retriever (domain knowledge)
        if intent and ("question" in intent.lower() or intent.lower().startswith("qust") or intent.lower().startswith("quest")):
            _print_log("Question intent detected; invoking retriever")
            try:
                # create a retriever via factory (deterministic; avoid registering as LLM-callable tool)
                docs = retrieve_tool(query , llm=self.llm)
                if docs:
                    formatted = "\n".join([f"- {d.page_content} (source: {d.metadata.get('source', 'unknown')})" for d in docs])
                else:
                    formatted = "No relevant knowledge found."
                result["tool_used"] = "domain_knowledge_retriever"
                result["tool_output"] = formatted
                self._append_tool_log("domain_knowledge_retriever", query, formatted)
                self._append_history("user", query)
                self._append_history("tool", f"domain_knowledge_retriever -> {formatted}")
            except Exception as e:
                _print_log("Retriever error:", e)
            return result

        # Conversation or fallback: no tools used — LLM will render
        _print_log("Conversation / fallback; no deterministic tool executed.")
        try:
                # create a retriever via factory (deterministic; avoid registering as LLM-callable tool)
                docs = retrieve_tool(query , llm=self.llm)
                if docs:
                    formatted = "\n".join([f"- {d.page_content} (source: {d.metadata.get('source', 'unknown')})" for d in docs])
                else:
                    formatted = "No relevant knowledge found."
                result["tool_used"] = "retriever"
                result["tool_output"] = formatted
                self._append_tool_log("retriever", query, formatted)
                self._append_history("user", query)
                self._append_history("tool", f"retriever -> {formatted}")
        except Exception as e:
                _print_log("Retriever error:", e)
        return result

    # -----------------------
    # LLM rendering (narrator)
    # -----------------------
    def _render_with_llm(self, query: str, facts: Dict[str, Any]) -> str:
        """
        Ask the LLM to format a natural assistant response, using facts verbatim.
        LLM must not call tools or change state.
        """
        # Compose a safe system + human prompt
        system_content = (
            f"You are {self.name}, a professional personal assistant. "
            "Use the facts below where applicable. Do NOT invent facts."
        )
        human_content = (
            f"User query: {query}\n\n"
            "Facts (use if present):\n"
            f"{json.dumps(facts, indent=2)}\n\n"
            "Emotional snapshot (internal state):\n"
            f"{json.dumps(ash_state.get('emotions', {}), indent=2)}\n\n"
            "Present the data just given if any then say a small sentance, Respond like your emotional state. If facts are provided, use them exactly. Keep the answer one under paragraph."
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

        # 1) routing + deterministic execution
        route_result = self._deterministic_execute(query)

        # 2) prepare facts to give to LLM renderer
        facts = {
            "intent": route_result.get("intent"),
            "intent_score": route_result.get("intent_score"),
            "command": route_result.get("command"),
            "command_score": route_result.get("command_score"),
            "tool_used": route_result.get("tool_used"),
            "tool_output": route_result.get("tool_output"),
        }

        # 3) optionally update emotions (example logic)
        # You can implement richer emotion policies; here is a simple demo:
        try:
            # small heuristic: if wrong tool usage or user asks again, increase frustration
            if facts["tool_used"] is None and facts["intent"] and facts["intent"].lower().startswith("command"):
                # call update_emo to show changing emotion (this is deterministic tool call)
                update_emo(ash_state, "frustration", 1)
                self._append_tool_log("update_emo", {"emo": "frustration", "val": 1}, ash_state["emotions"])
        except Exception as e:
            _print_log("Emotion update failed:", e)

        # 4) render via LLM (narrator)
        final_text = self._render_with_llm(query, facts)

        # 5) persist final result in state and history
        ash_state["res"] = final_text
        self._append_history("ash", final_text)

        # Print short summary to stderr for debugging
        _print_log("Finished run: intent=", facts["intent"], "tool=", facts["tool_used"])
        return final_text

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
