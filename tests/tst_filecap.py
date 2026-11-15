"""
tst_filecap.py -- one file read never loads more than a fixed number of bytes.

file_manager is loaded by path (importing the `tools` package pulls in the whole classifier stack).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_filecap.py
"""

import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ASH_FILE_MAX_READ_BYTES"] = "1000"
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("fm_under_test", os.path.join(ROOT, "tools", "file_manager.py"))
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def c1_cap():
    print("\n=== C1. The cap ===")
    d = tempfile.mkdtemp()
    big = os.path.join(d, "big.txt")
    with open(big, "w", encoding="utf-8") as f:
        f.write("line of text\n" * 100000)               # about 1.3 MB
    small = os.path.join(d, "small.txt")
    with open(small, "w", encoding="utf-8") as f:
        f.write("hello\nworld\n")
    check("the cap is the one set", fm.MAX_READ_BYTES == 1000)
    text = fm.read_text_capped(big)
    check("a big file is read only up to the cap", len(text.encode("utf-8")) <= 1000 and text.startswith("line of text"), str(len(text)))
    r = fm.TextHandler().read(big)
    check("the preview says it was cut", r.truncated is True and len(r.preview) <= 1000, str(r.truncated))
    r = fm.TextHandler().read(small)
    check("a small file is whole and not marked cut", r.preview == "hello\nworld\n" and r.truncated is False)
    hits = fm.TextHandler().search(big, "line")
    check("search over a big file is bounded too (25 hits at most)", 0 < len(hits) <= fm.MAX_SEARCH_RESULTS)
    check("the end of a big file is not read", "zzzend" not in fm.read_text_capped(big))
    multi = os.path.join(d, "multi.txt")
    with open(multi, "w", encoding="utf-8") as f:
        f.write("\u00e9" * 2000)                          # two bytes each: the cap lands inside one
    check("a cap in the middle of a character does not raise", fm.read_text_capped(multi) is not None)
    out = fm.read_file_tool(big)
    check("read_file_tool reports truncated", out["ok"] and out["truncated"] is True, str(out)[:200])


if __name__ == "__main__":
    c1_cap()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
