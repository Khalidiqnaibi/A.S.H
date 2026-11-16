r"""
tst_spacedpath.py -- a Windows path with spaces in it is one path.

Live: "Read the file C:\Users\pc\Desktop\kb\papers\Mortality of nurom\mortal ai paper.pdf" was read as
"C:\Users\pc\Desktop\kb\papers\Mortality" -- "No file exists at that path" -- and the model then said the user was confusing the
file system with something else.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_spacedpath.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from tools.file_manager import _extract_path  # noqa: E402

check = tw.check


def p1_spaces():
    print("\n=== P1. Paths with spaces ===")
    d = tempfile.mkdtemp()
    deep = os.path.join(d, "kb", "papers", "Mortality of nurom")
    os.makedirs(deep)
    f = os.path.join(deep, "mortal ai paper.pdf")
    open(f, "w").write("x")
    check("a real path with spaces, whole", _extract_path(f"Read the file {f}") == f, str(_extract_path(f"Read the file {f}")))
    check("with a question mark after it", _extract_path(f"what is in {f}?") == f, str(_extract_path(f"what is in {f}?")))
    ghost = os.path.join(d, "kb", "papers", "Not There", "some paper.pdf")
    check("a path that does not exist is still taken up to the file name", _extract_path(f"Read the file {ghost}") == ghost,
          str(_extract_path(f"Read the file {ghost}")))
    check("a path with no spaces is as before", _extract_path(r"read C:\Users\pc\notes.txt please") == r"C:\Users\pc\notes.txt",
          str(_extract_path(r"read C:\Users\pc\notes.txt please")))
    check("the live line", _extract_path(r"Read the file C:\Users\pc\Desktop\kb\papers\Mortality of nurom\mortal ai paper.pdf")
          == r"C:\Users\pc\Desktop\kb\papers\Mortality of nurom\mortal ai paper.pdf",
          str(_extract_path(r"Read the file C:\Users\pc\Desktop\kb\papers\Mortality of nurom\mortal ai paper.pdf")))
    check("a quoted path is as before", _extract_path('open "C:\\a b\\c.txt"') == "C:\\a b\\c.txt")


if __name__ == "__main__":
    p1_spaces()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
