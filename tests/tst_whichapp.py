"""tst_whichapp.py -- two apps that sound like what was heard: ASH asks which instead of guessing or giving up. Fake executors only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_whichapp.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_desktop as td  # noqa: E402

check = tw.check


def rig():
    d = td.drig()
    d.apps["notepad"] = ("notepad", "notepad.exe")
    d.apps["notepal"] = ("notepal", "notepal.exe")
    return d


def main():
    print("\n=== W1. Two sound alike ===")
    d = rig()
    check("one known app that sounds the same is still just opened", d.hear("open outlok") == "Opening outlok." or d.ex_.opened[-1:] == ["outlook"], str(d.ex_.opened))
    d = rig()
    r = d.hear("open notepas")
    check("two candidates: asked, nothing opened", r == "Did you mean Notepad or Notepal?" and d.ex_.opened == [], f"{r} {d.ex_.opened}")
    check("naming one opens it", d.hear("notepad") == "Opening notepad." and d.ex_.opened == ["notepad"], str(d.ex_.opened))
    d = rig()
    d.hear("open notepas")
    d.hear("the second one")
    check("'the second one' is the second", d.ex_.opened == ["notepal"], str(d.ex_.opened))
    d = rig()
    d.hear("open notepas")
    check("neither: dropped", d.hear("neither") == "Okay." and d.ex_.opened == [])
    check("and not asked twice: the next line is just a line", d.hear("notepad") is None and d.ex_.opened == [])
    d = rig()
    d.hear("open notepas")
    d.t[0] += 500
    check("a late answer is not an answer", d.hear("notepad") is None and d.ex_.opened == [])
    d = rig()
    d.ex_.procs.add("notepal.exe")
    check("switch asks the same way", d.hear("switch to notepas") == "Did you mean Notepad or Notepal?")
    r = d.hear("notepal")
    check("and does a switch", r == "Switching to notepal." and d.ex_.focused == ["notepal.exe"], f"{r} {d.ex_.focused}")
    d = rig()
    check("nothing like it: left to the rest of the rules", d.hear("open flibbertigibbet") is None)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
