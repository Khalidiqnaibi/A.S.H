"""
tst_llmsafety.py -- what a failing model call must not do.

On a non-200 answer from a hosted model the LLM wrapper printed the request it had made -- headers (the API key) and the whole
conversation -- to stdout, which the daemon logs.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_llmsafety.py
"""

import importlib.util
import io
import os
import sys
import tempfile
import contextlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_llm():
    spec = importlib.util.spec_from_file_location("llm_under_test", os.path.join(ROOT, "tools", "LLM.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Resp:
    status_code = 401
    text = '{"error": "invalid key"}'

    def json(self):
        return {}


def t1_no_secrets():
    print("\n=== No key in the log ===")
    m = load_llm()
    secret = "sk-or-SECRET-KEY-12345"
    llm = m.LLM.__new__(m.LLM)
    for k, v in dict(_openrouter_key=secret, _openrouter_model="x/y", _max_tokens=50, _temperature=0.2, _timeout=5,
                     _hf_token=secret, _hf_model="x/y", _hf_url="http://example.invalid").items():
        object.__setattr__(llm, k, v)
    m._post = lambda url, **kw: Resp()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = llm._call_openrouter([{"role": "user", "content": "my private sentence about Omar"}])
    printed = buf.getvalue()
    check("a 401 from OpenRouter prints the status", "401" in printed, printed[:120])
    check("...but not the API key", secret not in printed, printed[:300])
    check("...nor what the person said", "private sentence" not in printed and "Omar" not in printed, printed[:300])
    check("the error the caller gets does not carry the key either", secret not in str(out))
    src = open(os.path.join(ROOT, "tools", "LLM.py"), encoding="utf-8").read()
    check("no print in the file still formats the headers", "headers={headers}" not in src)


if __name__ == "__main__":
    t1_no_secrets()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
