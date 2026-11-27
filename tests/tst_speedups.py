"""tst_speedups.py -- the answer cache, "more", the best model for this machine, "why was that slow", model pre-warm when you are back, the close-a-program offer in
the memory notice, and answers that no longer open with a restatement. Offline: nothing reaches the network, Ollama or the machine's processes.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_speedups.py"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import honest, speedups as sp  # noqa: E402

check = tw.check


def main():
    print("\n=== P1. What may be cached ===")
    check("a factual question and a real answer", sp.cacheable("what is a transformer in machine learning", "A transformer is a neural network built on attention."))
    for q in ("what about now", "what did I do today", "what is that", "why is my computer slow", "tell me about it"):
        check(f"not cached: {q!r}", not sp.cacheable(q, "x" * 50))
    c = sp.AnswerCache(path=os.path.join(tempfile.mkdtemp(), "c.json"), clock=lambda: 1000.0)
    c.put("What is a transformer in machine learning?", "A transformer is a neural network built on attention.")
    c2 = sp.AnswerCache(path=c.path, clock=lambda: 1000.0 + 3600)
    check("kept on disk, found whatever the punctuation or case", c2.get("what is a transformer in machine learning") is not None)
    c3 = sp.AnswerCache(path=c.path, clock=lambda: 1000.0 + 2 * 86400)
    check("gone after a day", c3.get("what is a transformer in machine learning") is None)

    print("\n=== P2. In the runtime ===")
    sp.MIN_ANSWER = 3                                   # the test model says "MODEL"
    rt = tm.rig()
    rt.cfg.cache_path = os.path.join(tempfile.mkdtemp(), "ac.json")
    q = "explain how a transformer pays attention to words"
    used, said = rt.say(q)
    check("first time: the model answers", used and said == ["MODEL"], f"{used} {said}")
    n = len(rt.calls)
    used, said = rt.say(q)
    check("asked again: the same answer, no model call", not used and len(rt.calls) == n and said and said[-1] == "MODEL", f"{used} {said}")
    used, said = rt.say("clear your cache")
    check("it can be cleared", said[-1].startswith("Cleared 1 remembered answer"), str(said))
    used, said = rt.say(q)
    check("...then the model again", used)

    print("\n=== P3. 'More' ===")
    rt = tm.rig()
    rt.say("explain how a transformer pays attention to words")
    used, said = rt.say("tell me more")
    check("handed to the model, rewritten to expand that answer", used and rt.calls[-1].startswith("Tell me more about this, in a few sentences: explain how a transformer"), str(rt.calls[-1:]))
    rt = tm.rig()
    used, said = rt.say("more")
    check("nothing asked before: left alone", not rt.calls or not rt.calls[-1].startswith("Tell me more about this"), str(rt.calls[-1:]))

    print("\n=== P4. The best model for this machine ===")
    pick, why = sp.best_model(["mistral:latest", "qwen2.5:3b", "nomic-embed-text:latest"], 16)
    check("16 GB: the 3B, not the 7B (a quarter of the memory)", pick == "qwen2.5:3b", f"{pick} {why}")
    pick, why = sp.best_model(["mistral:latest", "qwen2.5:3b"], 64)
    check("64 GB: the big one", pick == "mistral:latest")
    check("only a big one installed: it, rather than nothing", sp.best_model(["mistral:latest"], 8)[0] == "mistral:latest")
    check("nothing installed: says so", sp.best_model(["nomic-embed-text"], 16)[0] == "")

    print("\n=== P5. Why was that slow ===")
    check("the note, the model's size, the memory", sp.why_slow(48.0, "the machine was squeezed (memory at 92%)", "mistral:latest", 92.0, 30.0, "slow")
          == "The last one took 48 s because the machine was squeezed (memory at 92%); mistral:latest is a big model for this computer. "
             "A smaller model (say “use the small model”), or closing a big program, would help.")
    check("a rule answer was quick", "quick" in sp.why_slow(0.4, "", "x", 40.0, 10.0, "fast"))
    rt = tm.rig()
    rt._last_turn = {"took_s": 63.0, "pathway": "slow", "tool": "", "timings": {}}
    rt.ash.llm = types.SimpleNamespace(_ollama_model="mistral:latest")
    used, said = rt.say("why was that so slow")
    check("asked in the runtime: by rule", not used and said and said[-1].startswith("The last one took 63 s"), f"{used} {said}")

    print("\n=== P6. Openers ===")
    check("restatement and 'certainly' go", honest.strip_openers("Certainly! I understand you're asking about GPUs. A GPU runs thousands of small threads at once.")
          == "A GPU runs thousands of small threads at once.")
    check("never down to nothing", honest.strip_openers("Sure! Yes.") == "Sure! Yes.")

    print("\n=== P7. Pre-warm and the close offer ===")
    rt = tm.rig()
    warmed = []
    rt.ash.llm = types.SimpleNamespace(warm=lambda: warmed.append(1) or True, _ollama_model="x")
    rt._model_unloaded = True
    rt._prewarm_model()
    import time as _t
    for _ in range(50):
        if warmed:
            break
        _t.sleep(0.05)
    check("the model is loaded when they come back", warmed == [1] and rt._model_unloaded is False)
    rt = tm.rig()
    offers = []
    rt.pcdoctor = types.SimpleNamespace(answer=lambda kind, now: "It's slow mainly because memory is at 92%. Want me to close Opera? It's using 3.1 GB -- anything unsaved there may be lost.")
    rt.submit = lambda ev: offers.append(ev.text) if (getattr(ev, "meta", None) or {}).get("topic") == "memory" else None
    rt.governor = types.SimpleNamespace(mem=95.0, cpu=20.0, status=lambda: {})
    rt._machine = types.SimpleNamespace(procs=lambda: [], own_pid=1)
    now = rt.clock()
    rt._mem_high_since = now - 1300
    rt._mem_told_at = -1e18
    rt._memory_notice(now)
    check("the memory notice carries the offer to close the biggest program", offers and offers[0].endswith("Want me to close Opera? It's using 3.1 GB -- anything unsaved there may be lost."), str(offers))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
