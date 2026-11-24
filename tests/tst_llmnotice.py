"""
tst_llmnotice.py -- the model-swap notice is said once, ever.

"qwen3:4b only thinks out loud here, so I'm answering with qwen2.5:3b now" was spoken at every start, because .env still names the reasoning model.
It is news once.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_llmnotice.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check


def main():
    from tools.LLM import told_before
    p = os.path.join(tempfile.mkdtemp(), "state", "llm_notices.json")
    print("\n=== N1. Once ===")
    check("the first time: not told before (and now recorded)", told_before("qwen3:4b->qwen2.5:3b", p) is False)
    check("every later start: told before", told_before("qwen3:4b->qwen2.5:3b", p) is True and told_before("qwen3:4b->qwen2.5:3b", p) is True)
    check("a different swap is news again", told_before("qwen3:4b->llama3.2:3b", p) is False)
    check("and the file keeps them", sorted(__import__("json").load(open(p, encoding="utf-8"))) == ["qwen3:4b->llama3.2:3b", "qwen3:4b->qwen2.5:3b"])
    open(p, "w").write("{ torn")
    check("a torn file means 'say it', not a crash", told_before("x->y", p) is False)
    check("an unwritable place means 'say it' too", told_before("x->y", os.path.join(os.devnull, "no", "llm.json")) is False)
    print("\n=== N2. Where it is used ===")
    import inspect
    from tools import LLM
    src = inspect.getsource(LLM)
    check("both notices go through it", src.count("told_before(") >= 2 and "told_before(f\"{model}->{alt}\")" in src and "told_before(f\"{model}->none\")" in src)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
