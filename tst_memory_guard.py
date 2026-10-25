"""
tst_memory_guard.py -- a failed model call must never be remembered.

tools/LLM.py does not raise when the provider is down: it returns the error as
plain text. Before the guard that text -- and ASH's own "I couldn't put together
a response" -- was written to memory like anything ASH had said, where entity
extraction turned it into entities (one was named after the error) and
"last seen" context, and the reflex cache could keep it as the answer to a query.

  * is_failure_text: the provider errors and the canned no-reply line, and
    nothing else (a tool saying "Error: no such file" is a real answer)
  * MemoryEngine: failures are not routed, not buffered; the user's half of the
    turn still is
  * the whole brain with a model that is down: the reply is the canned line,
    nothing of it reaches the router, and it is not cached as a reflex

Run:  python tst_memory_guard.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_brain  # noqa: E402
from src.brain.memory_engine import NO_REPLY_TEXT, MemoryEngine, is_failure_text  # noqa: E402
from src.brain.signals import Pathway  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


class Recorder:
    def __init__(self):
        self.seen = []

    def route_utterance(self, text, source="chat", importance=0.5, actor=None):
        self.seen.append((source, text))
        return {"stored": True}

    def retrieve_context(self, query, **kw):
        return {"core": "", "episodic": "", "entity": ""}


class DownLLM:
    """What tools/LLM.py returns when Ollama is not running."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        class R:
            content = ("[LLM] Error calling Ollama API: HTTPConnectionPool(host='127.0.0.1', "
                       "port=4444): Max retries exceeded")
        return R()


def main():
    print("\n=== M1. What counts as a failure ===")
    fails = [
        NO_REPLY_TEXT,
        "  " + NO_REPLY_TEXT + "\n",
        "[LLM] Error calling Ollama API: timed out",
        "[LLM ERROR] Ollama request failed",
        "Error calling Ollama API: x",
        "Error from LLM provider: 500",
        "API Error: overloaded",
    ]
    for t in fails:
        check(f"failure: {t.strip()[:50]!r}", is_failure_text(t))
    fine = [
        "The time is 10:00.", "Error: no such file", "Result: 42", "I couldn't find that file.",
        "I couldn't put together a response for that, sorry -- but here is the list", "",
        None, 5, "my [LLM] notes", "Here's what I found",
    ]
    for t in fine:
        check(f"not a failure: {str(t)[:50]!r}", not is_failure_text(t))

    print("\n=== M2. The memory engine ===")
    r = Recorder()
    m = MemoryEngine(r)
    out = m.commit_utterance("[LLM] Error calling Ollama API: x", source="ash")
    check("a provider error is not routed to memory", r.seen == [] and "skipped" in out)
    m.commit_utterance(NO_REPLY_TEXT, source="ash")
    check("nor is the canned no-reply line", r.seen == [])
    m.commit_utterance("my name is Sam", source="chat")
    check("an ordinary utterance still is", r.seen == [("chat", "my name is Sam")])

    r = Recorder()
    m = MemoryEngine(r)
    m.commit_turn("tell me a story", "[LLM] Error calling Ollama API: x", "user", "ash",
                  pathway=Pathway.SLOW)
    check("a turn whose reply failed stores the user's half only",
          r.seen == [("chat", "tell me a story")], str(r.seen))
    block = m.working_block()
    check("...and the failed reply is not in working memory either",
          "tell me a story" in block and "Ollama" not in block, block)

    r = Recorder()
    m = MemoryEngine(r)
    m.commit_turn("hi", "Hello! How can I help?", "user", "ash", pathway=Pathway.SLOW)
    check("a real turn stores both halves", [s for s, _ in r.seen] == ["chat", "ash"], str(r.seen))
    r = Recorder()
    m = MemoryEngine(r)
    m.commit_turn("open it", "Error: no such file", "user", "ash", pathway=Pathway.SLOW)
    check("a tool's own 'Error: ...' answer is a real reply and is kept",
          [s for s, _ in r.seen] == ["chat", "ash"], str(r.seen))

    print("\n=== M3. The whole brain with the model down ===")
    brain = tst_brain.build_brain(tempfile.mkdtemp(prefix="ash_guard_"))
    rec = Recorder()
    brain.memory.router = rec
    llm = DownLLM()
    brain.system2.llm = llm
    brain.llm = llm
    for q in ("tell me about the history of the roman empire", "what do you think about jazz"):
        resp = brain.think(q)
        check(f"{q!r}: the reply is the canned line, not the error",
              resp.text == NO_REPLY_TEXT or "Ollama" not in resp.text, resp.text[:80])
    stored = " | ".join(t for _, t in rec.seen)
    check("no provider error text reached the router", "Ollama" not in stored and "[LLM]" not in stored, stored[:160])
    check("the canned line did not either", NO_REPLY_TEXT not in stored, stored[:160])
    check("the user's words were remembered", "roman empire" in stored and "jazz" in stored)
    check("the model really was called (so this tested the failure path)", llm.calls >= 1, str(llm.calls))
    cached = [v[1] for v in brain.system1._reflex.values()]
    check("the failed reply is not cached as a reflex", not any(is_failure_text(t) for t in cached),
          str(cached[:2]))

    print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
