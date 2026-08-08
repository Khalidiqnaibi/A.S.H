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
from tools import (
    file_info_tool,
    read_file_tool,
    write_file_tool,
    list_directory_tool,
    search_files_tool,
    classify_and_route,
    classify_intent, 
    sentiment_tool ,
    _INTENT_CATALOG,
    date_time_tool, 
    calculator_tool, 
    LLM, 
    EmotionEngine, 
    TelemetrySignals, 
    DEFAULT_MOOD
)

# New generic tool system: registry + MCP client + shape-file tools.
# Every native tool now has one JSON file in tools/shapes/ describing
# both its classifier intent (tag/description/patterns) and which
# function implements it. Add a tool by adding a shape file --
# nothing here or in _deterministic_execute() needs to change.
from tools.registry import REGISTRY as TOOL_REGISTRY, ToolEntry
from tools.MCP_client import load_mcp_servers
from tools.shapes.shape_loader import load_all_shapes, register_shape_tools

register_shape_tools(load_all_shapes())

# Connect any MCP servers listed in mcp_servers.json (no-op if the file
# or the `mcp` package isn't present -- ASH runs fine without either).
load_mcp_servers("mcp_servers.json")

# memory
from src.memory.memory_router import MemoryRouter
from src.memory.core.core_manager import CoreMemoryEngine
from src.memory.entity.entity_manager import EntityManager
from src.memory.episodic.episodic_manager import EpisodicMemory


##############
#~ Immortal ~#
##############

#Ash attempt num 5 
USER = "Immortal" #"khalid afif sami iqnaibi"
  
# kparser=Sen()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = "mistral:latest"

# Simple runtime state type
AshState = Dict[str, Any]

# Default LLM settings (adjust env / config if you prefer)
DEFAULT_LLM_TEMPERATURE = 0.9

embedder= _INTENT_CATALOG.get_model()

# Global shared state (singleton-ish) — persists across runs in same process
ash_state: AshState = {
    "history": [],        # list of {"role": "user"|"ash"|"tool", "text": "...", "time": ISO}
    "emotions": dict(DEFAULT_MOOD),   # fast-moving emotion vector; replaced by EmotionEngine.update() each turn
    "mood": dict(DEFAULT_MOOD),       # slow-moving baseline temperament
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

        # Initialize memory engines
        self.core_memory = CoreMemoryEngine(embedder=embedder)
        self.entity_memory = EntityManager()
        self.episodic_memory = EpisodicMemory(embedder=embedder)

        self.memory = MemoryRouter(
            core_mem=self.core_memory,
            entity_mem=self.entity_memory,
            episodic_mem=self.episodic_memory
        )

        self.emotion_engine = EmotionEngine()

        # LLM wrapper. If none passed, create one.
        if llm is None:
            try:
                self.llm = LLM(
                    mode = "ollama",
                    temperature=llm_temperature,
                    ollama_model=OLLAMA_MODEL,
                    ollama_url=OLLAMA_URL,
                    timeout=180,
                )
            except Exception as e:
                _print_log("Warning: failed to instantiate LLM wrapper:", e)
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
    # Router + executor
    # -----------------------
    def _deterministic_execute(self, query: str) -> Dict[str, Any]:
        """
        Use classify_and_route (embedding router) to decide what to run,
        then look the resulting intent up in the shared tool registry
        (tools/registry.py). This covers native Python tools AND any
        MCP server's tools transparently -- adding a new tool never
        requires editing this function again; see tools/registry.py
        and tools/mcp_client.py.
        """
        _print_log("Routing query:", query)
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
            "tool_success": None,  # None = no tool was invoked this turn
            "raw_route": route
        }

        if not intent:
            _print_log("No intent detected; no tools executed.")
            return result

        entry = TOOL_REGISTRY.match(intent)
        if entry is None:
            _print_log("No tool registered for intent:", intent)
            return result

        _print_log("Dispatching intent", intent, "-> tool", entry.name, "(", entry.kind, ")")
        tool_out = entry.run(query)
        tool_ok = bool(tool_out.get("ok"))

        result["tool_used"] = entry.name
        result["tool_output"] = tool_out
        result["tool_success"] = tool_ok

        self._append_tool_log(entry.name, query, tool_out)
        self._append_history("user", query)
        self._append_history("tool", f"{entry.name} -> {tool_out}")
        return result

    # -----------------------
    # LLM rendering (narrator)
    # -----------------------
    def _render_with_llm(self, query: str, facts: Dict[str, Any], mem_context: Dict[str, str], modulation_block: str) -> str:
        """
        Ask the LLM to format a natural assistant response, using facts verbatim.
        LLM must not call tools or change state.

        `mem_context` is fetched once in run() (avoids a duplicate retrieval
        call). `modulation_block` is the deterministic, rule-derived tone
        directive from EmotionEngine -- see tools/emo_v2.py. The LLM is told
        how to sound (warmth/directness/verbosity/energy as concrete
        numbers plus a one-line guidance string), not handed a raw emotion
        dump to interpret however it likes.
        """
        history_block = self._format_history_for_prompt()
        # Compose a safe system + human prompt
        system_content = (
            f"You are {self.name}, a loyal personal assistant. "
            "Use the facts below in a human readable format where applicable. Do NOT invent facts."
            "answer the query then say a small sentence"
        )
        core_block = mem_context.get("core", "")
        episode_block = mem_context.get("episodic", "")
        entity_block = mem_context.get("entity", "")

        human_content = (
            "Conversation so far:\n"
            f"{history_block}\n\n"
            f"User query: {query}\n\n"
        )
        if core_block:
            human_content += (
                "Relevant Core Constraints:\n"
                f"{core_block}\n\n"
            )
        if episode_block:
            human_content += (
                "Relevant Episodic Memory:\n"
                f"{episode_block}\n\n"
            )
        if entity_block:
            human_content += (
                "Relevant Entities & Grounding Information:\n"
                f"{entity_block}\n\n"
            )
        human_content += (
            "Facts (use if present):\n"
            f"{json.dumps(facts, indent=2)}\n\n"
            f"{modulation_block}\n\n"
            "Follow the tone directives above. Respond in a small paragraph."
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
                # CRITICAL FIX: If your custom wrapper returns a tuple, grab the first item (the text)
                if isinstance(llm_resp, tuple):
                    return str(llm_resp[0])
                
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
        ash_state["input"] = query

        # 1) routing + deterministic execution
        route_result = self._deterministic_execute(query)

        # 2) Route the USER query ONCE (with the actor tag applied immediately)
        memory_context = self.memory.route_utterance(
            text=query,
            source="chat",
            importance=0.5,
            actor=self.user
        )

        # 3) Fetch retrieval context once -- used both for the prompt and
        # as a "are we grounded in something real" signal for the emotion
        # engine (memory_hits).
        try:
            mem_context = self.memory.retrieve_context(query)
        except Exception as e:
            _print_log("Failed to retrieve memory context from router:", e)
            mem_context = {"core": "", "episodic": "", "entity": ""}
        memory_hits = sum(1 for v in mem_context.values() if v)

        # 4) Sentiment read on the user's message. sentiment_tool() never
        # raises (falls back to {"sentiment": "unknown"}), but guard anyway
        # since this must never block a response.
        try:
            sentiment = sentiment_tool(query)
        except Exception as e:
            _print_log("sentiment_tool failed:", e)
            sentiment = {"sentiment": "unknown", "confidence": 0.0}

        # 5) Deterministic emotion update -- the only thing that ever
        # mutates emotional state. Built entirely from this turn's real telemetry
        signals = TelemetrySignals(
            query_text=query,
            sentiment_label=sentiment.get("sentiment"),
            sentiment_score=sentiment.get("confidence", 0.0) or 0.0,
            intent=route_result.get("intent"),
            intent_score=route_result.get("intent_score", 0.0) or 0.0,
            tool_used=route_result.get("tool_used"),
            tool_success=route_result.get("tool_success"),
            memory_hits=memory_hits,
        )
        emo_result = self.emotion_engine.update(signals)
        ash_state["emotions"] = emo_result.emotions
        ash_state["mood"] = emo_result.mood
        self._append_tool_log("emotion_update", vars(signals), emo_result.as_dict())

        facts = {
            "intent": route_result.get("intent"),
            "tool_used": route_result.get("tool_used"),
            "tool_output": route_result.get("tool_output"),
            "memory_context": memory_context
        }

        # 6) render via LLM (narrator), guided by the deterministic
        # modulation profile rather than a raw emotion dump.
        final_text = self._render_with_llm(query, facts, mem_context, emo_result.modulation.as_prompt_block())

        ash_state["res"] = final_text
        self._append_history("ash", final_text)

        # 7) Save ASH's response to episodic memory only (NOT the user's
        # query again). source="ash" so it can never be misclassified as
        # a new core rule -- see the guard in route_utterance().
        try:
            self.memory.route_utterance(
                text=final_text,
                source="ash",
                importance=0.5,
                actor=self.name  # Use ASH's name here
            )
        except Exception as e:
            _print_log("Memory routing failed:", e)

        # Print short summary to stderr for debugging
        _print_log("Finished run: intent=", facts["intent"], "tool=", facts["tool_used"],
                    "tone=", emo_result.modulation.tone_label)
        return final_text

    # utility: pretty print current state (developer helper)
    def status_info(self) -> Dict[str, Any]:
        uptime = datetime.now() - self.start_time
        return {
            "name": self.name,
            "uptime_seconds": int(uptime.total_seconds()),
            "status": self.status,
            "last_input": ash_state.get("input"),
            "last_output_preview": str(ash_state.get("res", ""))[:200],
            "emotions": ash_state.get("emotions", {}),
        }

llm = LLM(
    temperature=DEFAULT_LLM_TEMPERATURE,
    openrouter_key=OPENROUTER_API_KEY,
    openrouter_model=OPENROUTER_MODEL,
    timeout=180,
)
ash = ASH(llm=llm)

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