"""
tst_followups.py -- the fixes and features that came out of reviewing the browsing and profile work.

Each section is one change and was added with it; the file grows one commit at a time. Fixes first (the bugs a probe
confirmed), then the features. Nothing here touches the real machine: no real media key, no app launch, no network.

Run:  PYTHONIOENCODING=utf-8 python tst_followups.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
SECTIONS = []


def section(fn):
    SECTIONS.append(fn)
    return fn


@section
def fix_small_model_outage():
    print("\n=== FIX: one Ollama hiccup does not switch the small model off for the session ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ash_llm_fu", os.path.join(HERE, "tools", "LLM.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    calls = []
    state = {"down": True, "missing": False}

    class R:
        status_code = 200
        text = '{"message": {"content": "ok"}}'

    class R404:
        status_code = 404
        text = 'model "tiny:1b" not found'

    def post(url, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == "tiny:1b":
            if state["missing"]:
                return R404()
            if state["down"]:
                raise ConnectionError("Ollama is restarting")
        return R()
    L.requests.post = post
    llm = L.LLM(mode="ollama", ollama_model="big:7b")
    llm.set_light_model("tiny:1b")
    msg = [L.HumanMessage(content="hi")]
    with llm.route("light"):
        said = llm._call(msg)
    check("a refused connection: the main model answers", said == "ok" and calls == ["tiny:1b", "big:7b"], f"{said!r} {calls}")
    check("...and the small model stays on", llm._light_model == "tiny:1b", repr(llm._light_model))
    state["down"] = False
    calls.clear()
    with llm.route("light"):
        llm._call(msg)
    check("...so it is used again as soon as Ollama is back", calls == ["tiny:1b"], str(calls))
    state["missing"] = True
    calls.clear()
    with llm.route("light"):
        llm._call(msg)
    check("a model that is not installed is still switched off (404)", llm._light_model == "", repr(llm._light_model))
    calls.clear()
    with llm.route("light") as on:
        llm._call(msg)
    check("...and not tried again", calls == ["big:7b"] and not on, str(calls))


if __name__ == "__main__":
    for fn in SECTIONS:
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
