"""
tst_reasonleak.py -- a model's reasoning is never spoken (qwen3:4b: seven times in one day).

Live: "Okay, let's break this down. User says 'When?'...", "We are in the middle of a conversation where the user has been on
Stremio...", "Okay, let's unpack this carefully. User says 'Again.' after multiple screenshots...". The detector knew "the user" and
"let me" only. Now it knows these, and a leaked answer is asked for once more, directly, before anything is said.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_reasonleak.py
"""

import importlib.util
import json as _json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    os.environ["ASH_LLM_STATE"] = os.path.join(tempfile.mkdtemp(), "llm_notices.json")      # the 'said once, ever' record: not this machine's
    spec = importlib.util.spec_from_file_location("ash_llm_reasonleak", os.path.join(ROOT, "tools", "LLM.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    return L


def r1_detector():
    print("\n=== R1. What is reasoning ===")
    L = load()
    for t in ['Okay, let\'s break this down. User says "When?" after a bunch of recent activity logs.',
              "We are in the middle of a conversation where the user has been on Stremio for about 9 minutes without typing.",
              'Okay, let\'s unpack this carefully. User says "Again." after multiple screenshots of Stremio activity.',
              "Hmm, the user is asking how we are feeling.", "Wait, the actual user query here is Mute microphone for 10 minutes.",
              "Let's see what they want. They said create a routine."]:
        check(f"reasoning: {t[:52]!r}", L.looks_like_reasoning(t))
    for t in ["Sure, the user guide is in docs.", "Let's go to the park tomorrow!", "Okay, let's do it.", "User accounts are in settings.",
              "Seven. There are seven continents.", "We are done for today."]:
        check(f"an answer: {t[:52]!r}", not L.looks_like_reasoning(t))


class R:
    status_code = 200

    def __init__(self, content):
        self.text = _json.dumps({"message": {"content": content}})


def r2_switch():
    print("\n=== R2. A model that only thinks out loud is replaced, once, by one that does not ===")
    L = load()
    check("which models think", L.is_reasoning_model("qwen3:4b") and L.is_reasoning_model("deepseek-r1:1.5b") and not L.is_reasoning_model("qwen2.5:3b")
          and not L.is_reasoning_model("qwen3:4b-instruct-2507-q4_K_M") and not L.is_reasoning_model("mistral:latest"))
    sent, gets, unloaded = [], [], []
    leak = 'Okay, let' + "'" + 's break this down. User says "When?" after a bunch of logs.'

    class Resp:
        def __init__(self, content, status=200, tags=None):
            self.status_code = status
            self.text = _json.dumps({"message": {"content": content}})
            self._tags = tags

        def json(self):
            return self._tags

    tags = {"models": [{"name": "qwen3:4b", "size": 2.6e9}, {"name": "nomic-embed-text", "size": 2.7e8}, {"name": "deepseek-r1:1.5b", "size": 1.1e9},
                       {"name": "qwen2.5:3b", "size": 1.9e9}, {"name": "big:70b", "size": 40e9}]}

    def post(url, json=None, timeout=None, **k):
        if url.endswith("/api/generate"):
            unloaded.append(json["model"])
            return Resp("")
        sent.append((json["model"], json["options"].get("num_thread")))
        return Resp(leak if json["model"] == "qwen3:4b" else "At three.")
    og, op = L.requests.get, L.requests.post
    L.requests.get = lambda url, timeout=None, **k: (gets.append(url), Resp("", tags=tags))[1]
    L.requests.post = post
    try:
        llm = L.LLM(mode="ollama", ollama_model="qwen3:4b", ollama_url="http://x/api/chat")
        out = llm._call_ollama([{"role": "user", "content": "when is it?"}])
        check("the answer came from the model that does not think", out == "At three.", repr(out))
        check("qwen3 was asked once (not twice), then qwen2.5:3b once", sent[0][0] == "qwen3:4b" and sent[1][0] == "qwen2.5:3b" and len(sent) == 2, str(sent))
        check("the thinking model was unloaded, so both are not held in memory", unloaded == ["qwen3:4b"], str(unloaded))
        check("an embedding model and a 40 GB one were not chosen", all(m[0] in ("qwen3:4b", "qwen2.5:3b") for m in sent))
        check("it says so, once", len(llm._notices) == 1 and "qwen2.5:3b" in llm._notices[0] and llm.pop_notice() and llm.pop_notice() == "")
        sent.clear()
        out = llm._call_ollama([{"role": "user", "content": "and now?"}])
        check("from then on that model answers, in one request", sent == [("qwen2.5:3b", sent[0][1])] and out == "At three.", str(sent))
        check("the CPU threads are capped to a quarter of the logical cores", isinstance(sent[0][1], int) and 2 <= sent[0][1] <= max(2, (os.cpu_count() or 4) // 4))
        sent.clear(); unloaded.clear()
        llm = L.LLM(mode="ollama", ollama_model="qwen3:4b", ollama_url="http://x/api/chat")
        llm.warm()
        check("at warm-up, before any question, a thinking model is already replaced and unloaded", llm._ollama_model == "qwen2.5:3b"
              and unloaded == ["qwen3:4b"] and sent and sent[-1][0] == "qwen2.5:3b", str((llm._ollama_model, unloaded, sent)))
        sent.clear()
        os.environ["ASH_ALLOW_REASONING"] = "1"
        try:
            llm2 = L.LLM(mode="ollama", ollama_model="qwen3:4b", ollama_url="http://x/api/chat")
            llm2.warm()
            check("ASH_ALLOW_REASONING=1 keeps the model that was set", llm2._ollama_model == "qwen3:4b")
        finally:
            os.environ.pop("ASH_ALLOW_REASONING", None)
    finally:
        L.requests.get, L.requests.post = og, op
    # nothing else installed: nothing is said, and it does not try again and again
    tags["models"] = [{"name": "qwen3:4b", "size": 2.6e9}]
    sent.clear()
    L.requests.get = lambda url, timeout=None, **k: Resp("", tags=tags)
    L.requests.post = post
    try:
        llm = L.LLM(mode="ollama", ollama_model="qwen3:4b", ollama_url="http://x/api/chat")
        out = llm._call_ollama([{"role": "user", "content": "hi"}])
        check("no other model: nothing is said, one request", out == "" and len(sent) == 1 and "no other model" in (llm._notices[0] if llm._notices else ""), str((out, sent, llm._notices)))
        sent.clear()
        llm._call_ollama([{"role": "user", "content": "hi again"}])
        check("and it does not look for one every turn", len(llm._notices) == 1 and len(sent) == 1)
    finally:
        L.requests.get, L.requests.post = og, op


def r3_notice():
    print("\n=== R3. It tells the person, once ===")
    import tst_effects as te
    from types import SimpleNamespace as NS
    rt = te.rig()
    msgs = ["qwen3:4b only thinks out loud here, so I'm answering with qwen2.5:3b now."]
    rt.ash.llm = NS(pop_notice=lambda: msgs.pop(0) if msgs else "")
    n = len(rt.out)
    rt._model_notice()
    check("said once", len(rt.out) == n + 1 and "qwen2.5:3b" in rt.out[-1], str(rt.out[n:]))
    rt._model_notice()
    check("and not again", len(rt.out) == n + 1)
    rt.ash.llm = None
    rt._model_notice()
    check("no model layer: nothing, no crash", len(rt.out) == n + 1)


if __name__ == "__main__":
    r1_detector()
    r2_switch()
    r3_notice()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
