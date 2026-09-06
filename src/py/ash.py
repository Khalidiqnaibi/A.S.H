"""
ASH -- brain-first assistant runtime.

This file used to BE the cognition: a fixed nine-step pipeline that ran
classify -> tool -> memory -> emotion -> LLM, in that order, at full cost,
every single turn.

It is now a thin adapter. All cognition lives in `src/brain/`, which
implements the layered architecture:

    Homeostatic Modulator (limbic) -- broadcasts latent drives downward
    Sensory Extractors             -- text/vision -> percepts, salience, novelty
    Executive Core                 -- weighted-quorum arbitration + pathway gate
    Predictive Model               -- latent forward model, emits surprise
    System 1 Policy                -- actor-critic fast reflexes
    System 2                       -- candidate generation, imagination, narration
    VLA channel                    -- embodied/irreversible action jurisdiction
    Episodic/Semantic Memory       -- working memory + depth-tiered retrieval

The public surface is unchanged: `ash.run(query) -> str`, plus the module-level
`ash` singleton and `ash_state` dict that app.py already imports. Anything
built against the old runtime keeps working.

New surface:
    ash.run(query, image=...)   -- multimodal input
    ash.think(query)            -- full BrainResponse incl. trace
    ash.brain.status()          -- pathway mix, drives, critic weights, board
    ash.brain.last_trace        -- full introspection of the last cycle

Usage:
    from src.py.ash import ash
    print(ash.run("what time is it?"))
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from tools import (
    file_info_tool,
    read_file_tool,
    write_file_tool,
    list_directory_tool,
    search_files_tool,
    classify_and_route,
    classify_intent,
    get_intent_candidates,
    sentiment_tool,
    _INTENT_CATALOG,
    date_time_tool,
    calculator_tool,
    LLM,
    EmotionEngine,
    TelemetrySignals,
    DEFAULT_MOOD,
)

from tools.registry import REGISTRY as TOOL_REGISTRY, ToolEntry
from tools.MCP_client import load_mcp_servers
from tools.shapes.shape_loader import load_all_shapes, register_shape_tools

_SHAPES_LIST = load_all_shapes()
register_shape_tools(_SHAPES_LIST)

# MUST run after register_shape_tools() so the MCP-provided executable
# entry is the one that survives.
load_mcp_servers("mcp_servers.json")

# Shape metadata keyed by intent tag. The brain reads `action_class` and
# `reversible` from here to decide which board members have jurisdiction
# over a given tool -- see src/brain/system1.classify_action().
SHAPES: Dict[str, Dict[str, Any]] = {s["tag"]: s for s in _SHAPES_LIST if s.get("tag")}

# memory
from src.memory.memory_router import MemoryRouter
from src.memory.core.core_manager import CoreMemoryEngine
from src.memory.entity.entity_manager import EntityManager
from src.memory.episodic.episodic_manager import EpisodicMemory

# brain
from src.brain import Brain, Pathway
from src.concepts import ConceptSystem


##############
#~ Immortal ~#
##############

USER = "Immortal"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = "mistral:latest"

DEFAULT_LLM_TEMPERATURE = 0.9

# Vision is off by default -- flip ASH_ENABLE_VISION=1 to load the ViT
# extractor and give the VLA channel real grounding.
ENABLE_VISION = os.getenv("ASH_ENABLE_VISION", "0") == "1"

AshState = Dict[str, Any]

embedder = _INTENT_CATALOG.get_model()

# Global shared state (singleton-ish) -- persists across runs in same process.
# Kept for backward compatibility with app.py and the web UI; the brain owns
# the authoritative version of all of this.
ash_state: AshState = {
    "history": [],
    "emotions": dict(DEFAULT_MOOD),
    "mood": dict(DEFAULT_MOOD),
    "drives": {},
    "tool_log": [],
    "input": "",
    "res": "",
    "pathway": None,
    "client_time": None,
}


def _now_iso():
    return datetime.now().isoformat()


def _print_log(*args, **kwargs):
    print("[ASH]", *args, file=sys.stderr, flush=True, **kwargs)


def _candidate_fn(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """System 2's candidate generator: the classifier's top-k, not just its
    argmax. The fast path only ever saw rank 1; deliberation gets the pool."""
    try:
        return get_intent_candidates(query, top_k=top_k) or []
    except Exception:
        _print_log("get_intent_candidates failed")
        return []


class ASH:
    """Thin runtime wrapper around the brain. Owns memory engines, the LLM
    handle, and the legacy state dict; delegates all cognition."""

    def __init__(self, llm=None, llm_temperature: float = DEFAULT_LLM_TEMPERATURE, name: str = "A.S.H"):
        self.name = name
        self.user = USER
        self.start_time = datetime.now()
        self.status = "online"

        # Memory engines (unchanged).
        self.core_memory = CoreMemoryEngine(embedder=embedder)
        self.entity_memory = EntityManager()
        self.episodic_memory = EpisodicMemory(embedder=embedder)
        self.memory = MemoryRouter(
            core_mem=self.core_memory,
            entity_mem=self.entity_memory,
            episodic_mem=self.episodic_memory,
        )

        self.emotion_engine = EmotionEngine()

        if llm is None:
            try:
                self.llm = LLM(
                    mode="ollama",
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

        # Determine the latent dimensionality from the actual encoder rather
        # than hardcoding 384 -- swapping ASH_EMBED_MODEL shouldn't break the
        # predictive model or the concept prototypes.
        latent_dim = 384
        try:
            if embedder is not None:
                latent_dim = int(embedder.get_sentence_embedding_dimension())
        except Exception:
            pass


        # Associative layer. Seeded from the tool registry, CoreMemory and
        # EntityMemory on first run, restored from state/concepts/ after that.
        # Disable with ASH_CONCEPTS=0 -- the brain runs without it and falls
        # back to raw-embedding prediction and classifier-only routing.
        self.concepts = None
        if os.getenv("ASH_CONCEPTS", "1") != "0":
            try:
                self.concepts = ConceptSystem(
                    embedder=embedder,
                    registry=TOOL_REGISTRY,
                    core_memory=self.core_memory,
                    entity_memory=self.entity_memory,
                    intent_catalog=_INTENT_CATALOG,
                    shapes=SHAPES,
                    dim=latent_dim,
                )
                _print_log(f"Concept graph: {self.concepts.net.status()}")
            except Exception:
                _print_log("Concept layer failed to initialize; continuing without it")

        self.brain = Brain(
            name=self.name,
            user=self.user,
            llm=self.llm,
            embedder=embedder,
            classifier=classify_intent,
            candidate_fn=_candidate_fn,
            registry=TOOL_REGISTRY,
            memory_router=self.memory,
            core_memory=self.core_memory,
            emotion_engine=self.emotion_engine,
            sentiment_fn=sentiment_tool,
            shapes=SHAPES,
            enable_vision=ENABLE_VISION,
            latent_dim=latent_dim,
            concepts=self.concepts,
        )

        _print_log(
            f"{self.name} initialized (LLM: {bool(self.llm)}, vision: {ENABLE_VISION}, "
            f"latent_dim: {latent_dim}, tools: {len(TOOL_REGISTRY.list_tools())})"
        )

    # -----------------------
    # Legacy state bookkeeping
    # -----------------------
    def _append_history(self, role: str, text: str):
        ash_state.setdefault("history", [])
        ash_state["history"].append({"role": role, "text": text, "time": _now_iso()})

    def _append_tool_log(self, tool_name: str, tool_input: Any, tool_output: Any):
        ash_state.setdefault("tool_log", [])
        ash_state["tool_log"].append({
            "time": _now_iso(),
            "agent": self.name,
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output,
        })

    def _sync_state(self, query: str, response):
        ash_state["input"] = query
        ash_state["res"] = response.text
        ash_state["pathway"] = response.pathway.value
        ash_state["drives"] = self.brain.homeostasis.drives.as_dict()
        emo = self.brain.homeostasis.last_emotion_result
        if emo is not None:
            ash_state["emotions"] = emo.emotions
            ash_state["mood"] = emo.mood
        self._append_history("user", query)
        if response.tool_used:
            self._append_history("tool", f"{response.tool_used} -> {response.tool_output}")
            self._append_tool_log(response.tool_used, query, response.tool_output)
        self._append_history("ash", response.text)

    # -----------------------
    # Public API
    # -----------------------
    def think(self, query: str, image: Any = None):
        """Full cognitive cycle. Returns a BrainResponse (text + trace)."""
        if _maintenance_scheduler is not None:
            _maintenance_scheduler.note_activity()

        response = self.brain.think(query, image=image)
        self._sync_state(query, response)

        t = response.trace
        _print_log(
            f"[{t.pathway.value.upper()}] intent={response.intent} tool={response.tool_used} "
            f"surprise={t.prediction.get('surprise')} novelty={t.percepts.get('novelty')} "
            f"total={t.timings_ms.get('total', 0):.0f}ms"
        )
        if t.escalated_from:
            _print_log(f"  escalated from {t.escalated_from}: {'; '.join(t.notes[:2])}")
        return response

    def run(self, query: str, image: Any = None) -> str:
        """Backward-compatible entry point. Returns just the response text."""
        return self.think(query, image=image).text

    # -----------------------
    # Introspection
    # -----------------------
    def status_info(self) -> Dict[str, Any]:
        uptime = datetime.now() - self.start_time
        info = {
            "name": self.name,
            "uptime_seconds": int(uptime.total_seconds()),
            "status": self.status,
            "last_input": ash_state.get("input"),
            "last_output_preview": str(ash_state.get("res", ""))[:200],
            "last_pathway": ash_state.get("pathway"),
            "emotions": ash_state.get("emotions", {}),
        }
        try:
            info["brain"] = self.brain.status()
        except Exception:
            _print_log("brain.status() failed")
        return info

    def explain_last(self) -> Dict[str, Any]:
        """Full trace of the most recent cognitive cycle: which subsystems
        voted how, why the pathway was chosen, where the time went."""
        t = self.brain.last_trace
        return t.as_dict() if t is not None else {}


llm = LLM(
    temperature=DEFAULT_LLM_TEMPERATURE,
    openrouter_key=OPENROUTER_API_KEY,
    openrouter_model=OPENROUTER_MODEL,
    timeout=180,
)
ash = ASH(llm=llm)

# Maintenance / sleep phase. Now also drives brain consolidation: replay
# training of the forward model, homeostatic fatigue recovery, and weight
# persistence -- see src/brain/consolidation.py.
_maintenance_scheduler = None
try:
    from src.memory.maintenance import MaintenanceScheduler
    _maintenance_scheduler = MaintenanceScheduler(ash, idle_seconds=30 * 60)
    _maintenance_scheduler.start()
except Exception:
    logging.getLogger("ash.py").exception("Failed to start maintenance scheduler")


if __name__ == "__main__":
    print("Starting ASH interactive.", file=sys.stderr)
    print("  'exit'    quit", file=sys.stderr)
    print("  'status'  pathway mix, drives, critic weights, board composition", file=sys.stderr)
    print("  'why'     full trace of the last cognitive cycle", file=sys.stderr)
    print("  'concepts' concept graph size, plasticity, last consolidation", file=sys.stderr)
    print("  'sleep'   force a consolidation pass\n", file=sys.stderr)
    while True:
        try:
            q = input("You: ")
        except EOFError:
            break
        if not q:
            continue
        low = q.strip().lower()
        if low in ("exit", "quit"):
            break
        if low == "status":
            print(json.dumps(ash.status_info(), indent=2, default=str), "\n")
            continue
        if low == "why":
            print(json.dumps(ash.explain_last(), indent=2, default=str), "\n")
            continue
        if low == "concepts":
            print(json.dumps(ash.concepts.status() if ash.concepts else {},
                             indent=2, default=str), "\n")
            continue
        if low == "sleep":
            print(json.dumps(ash.brain.sleep(), indent=2, default=str), "\n")
            continue
        resp = ash.think(q)
        print(f"\nASH [{resp.pathway.value}]: {resp.text}\n")
