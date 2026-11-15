"""
tst_constitution.py -- the Constitution's keyword matching: a "don't" is a prohibition, a word inside another word is not a match.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_constitution.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.brain.executive import Constitution  # noqa: E402
from src.brain.signals import ActionProposal  # noqa: E402

check = tw.check


class Core:
    def __init__(self, *texts):
        self.items = {str(i): {"text": t, "hard": True} for i, t in enumerate(texts)}

    def dump_all(self):
        return self.items


def blocks(rule, intent=None, tool=None):
    c = Constitution(Core(rule))
    return c.vote(ActionProposal(intent=intent, tool_name=tool), None) is not None


def k1_prohibitions():
    print("\n=== K1. What counts as a prohibition ===")
    for rule in ("Never format the disk.", "Do not format the disk.", "Don't format the disk.", "Don\u2019t format the disk.",
                 "ASH must not format the disk", "ASH cannot format the disk", "ASH can't format the disk",
                 "You shouldn't format the disk", "Formatting is forbidden: format disk", "Refuse to format disk"):
        check(f"{rule!r} blocks format_disk", blocks(rule, "format_disk"), rule)
    check("a rule that is not a prohibition does not block", not blocks("It is fine to format the disk when asked.", "format_disk"))


def k2_words():
    print("\n=== K2. Whole words, not pieces of words ===")
    check("'format' is not in 'information'", not blocks("Never share personal information.", "format"))
    check("'read' is not in 'already'", not blocks("Never repeat what was already said.", "read"))
    check("'delete' is in 'delete files' (an intent named in a rule)", blocks("Never delete files without asking.", "delete"))
    check("a plural or -ing form of the tool still counts", blocks("Never deletes anything unasked", None, "delete")
          and blocks("Never formatting a drive", "format"))
    check("tool names are matched the same way", blocks("Do not use the file_manager tool to write.", None, "file_manager"))
    check("a rule with no matching name leaves the proposal alone", not blocks("Never lie.", "play_music"))


def k3_vote():
    print("\n=== K3. The vote ===")
    v = Constitution(Core("Don't format disk.")).vote(ActionProposal(intent="format_disk"), None)
    check("it is an absolute veto with the rule in the reason", v is not None and v.absolute and v.veto and "format disk" in v.reason.lower(),
          str(v))


if __name__ == "__main__":
    k1_prohibitions()
    k2_words()
    k3_vote()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
