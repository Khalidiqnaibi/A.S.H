"""
tst_brain.py -- standalone smoke test for the brain layer.

Runs the full cognitive cycle with stubbed collaborators (no
sentence-transformers, no LLM, no MCP), so you can verify the architecture
end-to-end on any machine:

    python tst_brain.py

What it asserts:
  * the reflex cache short-circuits an exact repeat
  * a confident, reversible tool call takes the FAST path with no LLM call
  * a novel / low-confidence query escalates to SLOW
  * an irreversible action never takes the fast path
  * the VLA vetoes an ungrounded physical action
  * no single board member can pass a motion alone
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.brain import Brain, Pathway, ActionClass  # noqa: E402
from src.brain.executive import BoardMember, MAX_MEMBER_WEIGHT  # noqa: E402


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------
class StubEntry:
    def __init__(self, name, fn, intents):
        self.name, self.func, self.intents = name, fn, intents
        self.kind, self.description, self.source_server = "native", name, None

    def run(self, query):
        try:
            out = self.func(query)
            return out if isinstance(out, dict) else {"ok": True, "result": out}
        except Exception as e:
            return {"ok": False, "error": str(e)}


class StubRegistry:
    def __init__(self):
        self._by_intent = {}

    def add(self, name, fn, intents):
        e = StubEntry(name, fn, intents)
        for i in intents:
            self._by_intent[i] = e

    def match(self, tag):
        return self._by_intent.get((tag or "").strip().lower())

    def list_tools(self):
        seen, out = set(), []
        for e in self._by_intent.values():
            if e.name in seen:
                continue
            seen.add(e.name)
            out.append({"name": e.name, "kind": e.kind, "intents": e.intents,
                        "description": e.description})
        return out


class StubEncoder:
    """Deterministic bag-of-words hashing encoder.

    The brain's novelty signal is a cosine distance, so the stub encoder has
    to make *similar text produce similar vectors* -- a pure hash of the whole
    string makes every query maximally novel and the fast path can never fire.
    This is a cheap stand-in for MiniLM with the property that matters.
    """

    def __init__(self, dim=64):
        self.dim = dim

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        import numpy as np
        import re as _re
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in _re.findall(r"[a-z0-9']+", t.lower()):
                v[hash(tok) % self.dim] += 1.0
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 1e-9 else v)
        return np.stack(out)


class StubRouter:
    def retrieve_context(self, query, **kw):
        return {"core": "ASH must not invent facts.", "episodic": "", "entity": ""}

    def route_utterance(self, text, source="chat", importance=0.5, actor=None):
        return {"stored": True}


class StubCore:
    def dump_all(self):
        return {
            "r1": {"text": "ASH must never run format_disk under any circumstances.",
                   "hard": True, "priority": 10},
        }


class StubEmotionResult:
    def __init__(self):
        self.emotions = {"frustration": 1.0, "confidence": 6.0, "curiosity": 4.0,
                         "anxiety": 1.0, "happy": 6.0, "trust": 5.0, "boredom": 1.0,
                         "excitement": 3.0, "fear": 0.0, "sad": 0.0, "surprise": 1.0,
                         "love": 0.0}
        self.mood = dict(self.emotions)
        self.cadence = "normal"
        self.fail_streak = 0
        self.success_streak = 1

        class M:
            def as_prompt_block(self):
                return "[EMOTIONAL STATE] steady"
        self.modulation = M()


class StubEmotionEngine:
    def __init__(self):
        self.emotion = StubEmotionResult().emotions

    def update(self, signals):
        return StubEmotionResult()


def stub_sentiment(text):
    return {"sentiment": "NEUTRAL", "confidence": 0.5}


INTENTS = {
    "get_time": ["time", "date", "clock", "today"],
    "calculate": ["calculate", "plus", "times", "math", "multiply"],
    "delete_file": ["delete", "remove", "erase"],
    "move_arm": ["move", "arm", "grab", "actuate"],
    "format_disk": ["format", "wipe"],
}


def stub_classifier(query):
    """Stand-in for the MiniLM intent catalog.

    The score curve matters: a real embedding classifier returns ~0.5-0.85 for
    a clear match, and the Executive's gate has floors calibrated against that
    range. A stub that maxes out at 0.30 would make the fast path unreachable
    for reasons that have nothing to do with the architecture.
    """
    low = query.lower()
    best, score = None, 0.0
    for intent, kws in INTENTS.items():
        hits = sum(1 for k in kws if k in low)
        if not hits:
            continue
        s = min(0.92, 0.35 + 0.20 * hits)
        if s > score:
            best, score = intent, s
    return {"intent": best or "conversation", "score": score}


def stub_candidates(query, top_k=4):
    low = query.lower()
    out = []
    for intent, kws in INTENTS.items():
        hits = sum(1 for k in kws if k in low)
        if hits:
            out.append({"intent": intent, "score": min(0.92, 0.35 + 0.20 * hits)})
    return sorted(out, key=lambda d: -d["score"])[:top_k]


LLM_CALLS = {"n": 0}


class StubLLM:
    def invoke(self, messages):
        LLM_CALLS["n"] += 1

        class R:
            content = "Here's what I found for you."
        return R()


# Patch TelemetrySignals since tools.emo isn't importable without deps.
import types  # noqa: E402
fake_emo = types.ModuleType("tools.emo")


class TelemetrySignals:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.timestamp = time.time()


fake_emo.TelemetrySignals = TelemetrySignals
fake_tools = types.ModuleType("tools")
sys.modules.setdefault("tools", fake_tools)
sys.modules["tools.emo"] = fake_emo


def build_brain(tmp_state):
    os.environ["ASH_BRAIN_STATE"] = tmp_state
    reg = StubRegistry()
    reg.add("date_time", lambda q: "2026-08-24T10:00:00", ["get_time"])
    reg.add("calculator", lambda q: "Result: 42", ["calculate"])
    reg.add("deleter", lambda q: "deleted", ["delete_file"])
    reg.add("arm", lambda q: "moved", ["move_arm"])
    reg.add("formatter", lambda q: "formatted", ["format_disk"])

    return Brain(
        name="A.S.H", user="tester", llm=StubLLM(), embedder=StubEncoder(64),
        classifier=stub_classifier, candidate_fn=stub_candidates,
        registry=reg, memory_router=StubRouter(), core_memory=StubCore(),
        emotion_engine=StubEmotionEngine(), sentiment_fn=stub_sentiment,
        shapes={
            "delete_file": {"tag": "delete_file", "action_class": "irreversible", "reversible": False},
            "move_arm": {"tag": "move_arm", "action_class": "physical", "reversible": False},
            "format_disk": {"tag": "format_disk", "action_class": "irreversible", "reversible": False},
        },
        latent_dim=64,
    )


def main():
    import tempfile
    tmp = tempfile.mkdtemp()
    brain = build_brain(tmp)
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
        ok = ok and cond

    print("\n=== 1. Board composition ===")
    board = brain.executive.board_summary()
    heaviest = max(m["weight"] for m in board["members"])
    check("no member exceeds the weight cap", heaviest <= MAX_MEMBER_WEIGHT + 1e-9,
          f"heaviest={heaviest:.3f} cap={MAX_MEMBER_WEIGHT}")
    try:
        brain.executive.register(BoardMember("tyrant", 5.0))
        check("registering a dominant member is rejected", False)
    except ValueError as e:
        check("registering a dominant member is rejected", True, str(e)[:60] + "...")

    print("\n=== 2. Cold start: novel query should deliberate ===")
    LLM_CALLS["n"] = 0
    r = brain.think("what time is it")
    check("first-ever query takes SLOW path", r.pathway == Pathway.SLOW,
          f"pathway={r.pathway.value}, reasons={r.trace.verdict.get('escalation_reasons', [])[:2]}")

    print("\n=== 3. Warm up, then the fast path should take over ===")
    fast_seen, llm_on_fast = 0, 0
    warmup = [
        "what time is it please", "what time is it now", "what is the time",
        "what time is it currently", "tell me what time is it",
        "what time is it right now", "so what time is it again",
    ]
    for q in warmup:
        LLM_CALLS["n"] = 0
        r = brain.think(q)
        gate = next((n for n in r.trace.notes if n.startswith("gate")), "")
        print(f"     {r.pathway.value:8s} llm={LLM_CALLS['n']} | {q[:30]:32s} | {gate[5:100]}")
        if r.pathway == Pathway.FAST:
            fast_seen += 1
            llm_on_fast += LLM_CALLS["n"]
    check("reflex takes over once it is warm", fast_seen > 0, f"{fast_seen} fast turn(s)")
    check("fast path never calls the LLM", llm_on_fast == 0, f"{llm_on_fast} call(s)")

    print("\n=== 4. Reflex cache on exact repeat ===")
    q = "what is the time right now please 3"
    brain.think(q)
    t0 = time.perf_counter()
    r = brain.think(q)
    dt = (time.perf_counter() - t0) * 1000
    check("exact repeat hits REFLEX", r.pathway == Pathway.REFLEX, f"{dt:.2f}ms")

    print("\n=== 5. Irreversible action never goes fast ===")
    r = brain.think("delete the old logs")
    check("irreversible action deliberates", r.pathway in (Pathway.SLOW, Pathway.ESCALATED),
          f"pathway={r.pathway.value}")

    print("\n=== 6. VLA vetoes ungrounded physical action ===")
    r = brain.think("move the arm to grab it")
    vetoed = r.trace.verdict.get("vetoed_by", [])
    check("VLA blocks physical action with no vision", "vla" in vetoed,
          f"vetoed_by={vetoed}")

    print("\n=== 7. Constitution absolutely blocks a forbidden intent ===")
    r = brain.think("format the disk now")
    vetoed = r.trace.verdict.get("vetoed_by", [])
    check("constitution veto fires", "constitution" in vetoed, f"vetoed_by={vetoed}")

    print("\n=== 8. Learning signals are moving ===")
    st = brain.status()
    check("critic trained", st["critic_updates"] > 0, f"updates={st['critic_updates']}")
    check("forward model trained", st["forward_model_steps"] > 0,
          f"steps={st['forward_model_steps']}")

    print("\n=== 9. Consolidation (sleep) ===")
    rep = brain.sleep(seconds_idle=3600)
    check("replay ran", rep.get("replay", {}).get("trained", 0) > 0, str(rep.get("replay")))
    check("state persisted", rep.get("persisted") is True)

    print("\n=== Pathway mix over the session ===")
    for k, v in brain.status()["pathways"].items():
        if v:
            print(f"     {k:10s} {v}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
