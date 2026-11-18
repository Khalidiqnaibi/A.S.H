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


def r2_retry():
    print("\n=== R2. A leaked answer is asked for again ===")
    L = load()
    sent = []
    replies = ['Okay, let\'s break this down. User says "When?" after a bunch of logs.', "At three."]
    orig = L.requests.post
    L.requests.post = lambda url, json=None, timeout=None, **k: (sent.append(json), R(replies[min(len(sent) - 1, 1)]))[1]
    try:
        llm = L.LLM(mode="ollama", ollama_model="m", ollama_url="http://x/api/chat")
        out = llm._call_ollama([{"role": "user", "content": "when is it?"}])
        check("the second, direct answer is what is said", out == "At three.", repr(out))
        check("it was asked once more, shorter, with the direct instruction", len(sent) == 2 and sent[1]["options"]["num_predict"] == 100
              and "Answer directly" in sent[1]["messages"][-1]["content"], str(sent[-1:])[:200])
        sent.clear()
        replies[:] = ["Okay, let's unpack this. User says hi.", "Hmm, the user is greeting me."]
        out = llm._call_ollama([{"role": "user", "content": "hi"}])
        check("still reasoning the second time: nothing is said", out == "" and len(sent) == 2, repr(out))
        sent.clear()
        replies[:] = ["It's three o'clock."]
        out = llm._call_ollama([{"role": "user", "content": "time?"}])
        check("a normal answer is one request and passes", out == "It's three o'clock." and len(sent) == 1)
    finally:
        L.requests.post = orig


if __name__ == "__main__":
    r1_detector()
    r2_retry()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
