"""
tst_emptyreply.py -- an empty model reply is empty.

A reasoning model that spends its whole budget thinking returns "" -- the designed "nothing to say". It was turned into
str(reply), "content='' additional_kwargs={} response_metadata={}", and spoken.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_emptyreply.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check


def t2_empty_reply():
    print("\n=== An empty reply is empty ===")
    from src.brain.system2 import System2Deliberator as S

    class Msg:
        def __init__(self, content):
            self.content = content

        def __repr__(self):
            return f"AIMessage(content={self.content!r}, additional_kwargs={{}}, response_metadata={{}})"
    check("an empty reply is the empty string, not the object's repr", S._text_of(Msg("")) == "")
    check("a reply with text is the text", S._text_of(Msg("Seven.")) == "Seven.")
    check("a reply with no content attribute is its string form", S._text_of("plain") == "plain")
    check("a list of parts is joined as text, not dropped", isinstance(S._text_of(Msg(["a", "b"])), str))


if __name__ == "__main__":
    t2_empty_reply()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
