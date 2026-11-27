"""tst_fix28.py -- small things from 28 Nov: the face has no "nod" (its small yes is "acknowledge"); a toast whose first line is "You may have new messages" has no sender (it was
read as one). Offline.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_fix28.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
from src.daemon import toasts  # noqa: E402

check = tw.check


def main():
    print("\n=== F1. Toast senders ===")
    for t in ("You may have new messages", "You have 3 new messages", "New message", "2 new notifications"):
        check(f"{t!r} has no sender", toasts.first_line(f"<toast><text>{t}</text></toast>") == "")
    check("a name is still a name", toasts.first_line("<toast><text>Omar</text><text>hi</text></toast>") == "Omar")
    print("\n=== F2. The face's small yes ===")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "daemon", "ambient.py"), encoding="utf-8").read()
    check("no call plays an animation the face does not have", '_play("nod"' not in src)
    from src.face import bridge
    check("'acknowledge' is one the bridge plays", '"acknowledge"' in open(bridge.__file__, encoding="utf-8").read())
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
