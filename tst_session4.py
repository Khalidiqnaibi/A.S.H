"""
tst_session4.py -- round three of the JARVIS / BMO ideas, one section per item.

  S1  The model is preloaded at start (one token, kept alive, no thinking cost)
  S2  A slow model answer gets a short spoken cue first; a quick one does not

Run:  PYTHONIOENCODING=utf-8 python tst_session4.py
"""
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402,F401
import tst_wondering as tw  # noqa: E402

check = tw.check


def load_llm():
    spec = importlib.util.spec_from_file_location("ash_llm_s4", os.path.join(HERE, "tools", "LLM.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    return L


def s1():
    print("\n=== S1. Model preload ===")
    L = load_llm()
    sent = []

    class R:
        def __init__(self, code=200, text="{}"):
            self.status_code, self.text = code, text

    orig = L.requests.post
    L.requests.post = lambda url, json=None, timeout=None, **k: (sent.append(dict(json)), R())[1]
    try:
        llm = L.LLM(mode="ollama", ollama_model="m", ollama_url="http://x/api/chat")
        check("warm() reports success", llm.warm() is True)
        p = sent[-1]
        check("one token only", p["options"]["num_predict"] == 1)
        check("kept loaded for hours", p["keep_alive"] == "6h")
        check("same model as real calls", p["model"] == "m")
        L.requests.post = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        check("an unreachable server is False, not a crash", llm.warm() is False)
        off = L.LLM(mode="openrouter", openrouter_key="k")
        check("a non-ollama backend does nothing", off.warm() is False)
    finally:
        L.requests.post = orig
    src = open(os.path.join(HERE, "ashd.py"), encoding="utf-8").read()
    check("the daemon starts the preload in the background", "ash-preload" in src and "ASH_PRELOAD_MODEL" in src)


def s2():
    print("\n=== S2. Thinking cue ===")
    import time
    for wait, work, expect in ((0.05, 0.4, True), (1.0, 0.01, False), (0.0, 0.3, False)):
        rt = te.rig(think_cue_s=wait)
        orig = rt.ash.think

        def slow(q, image=None, context="", orig=orig, work=work):
            time.sleep(work)
            return orig(q, image=image, context=context)
        rt.ash.think = slow
        rt.say("tell me something interesting about owls please")
        time.sleep(0.1)
        cues = [t for t in rt.out if t in rt._CUES]
        check(f"wait={wait} work={work}: cue {'spoken' if expect else 'not spoken'}", bool(cues) == expect, str(rt.out))
        if expect:
            check("the real answer still follows the cue", rt.out[-1] == "Sure." and rt.out.index(cues[0]) < len(rt.out) - 1)
    th = te.rig(think_cue_s=0.05)
    check("an unprompted remark gets no cue", th._thinking_cue(False) is None)


if __name__ == "__main__":
    s1()
    s2()
    print("\nALL OK" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
