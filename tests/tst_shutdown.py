"""
tst_shutdown.py -- closing the window stops ASH cleanly, and the process does not leave while it is still saving.

The console-close handler runs on a thread Windows starts; the main thread, in its loop, called the same shutdown when the
runtime stopped -- found it "already running", returned, and the process exited under the thread that was still writing what
the brain had learned. Stopper makes the second caller wait for the first (about what Windows allows a closing console), and the
handler skips the goodbye sound so the saves come first.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_shutdown.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.service import Stopper, console_event_handler, install_signal_handlers  # noqa: E402

check = tw.check


def s1_stopper():
    print("\n=== One shutdown, the others wait ===")
    calls = []
    t0 = time.time()

    def work(quick):
        calls.append(quick)
        time.sleep(0.4)
    st = Stopper(work, wait_s=3.0)
    th = threading.Thread(target=lambda: st(quick=True))
    th.start()
    time.sleep(0.1)
    st()                                                 # the main thread arrives while the other is still saving
    waited = time.time() - t0
    check("the work ran once", calls == [True], str(calls))
    check("the second caller did not return until the first had finished", waited >= 0.38 and st.finished.is_set(), f"{waited:.2f}s")
    th.join()
    t1 = time.time()
    st()
    check("after it is done, asking again returns at once", time.time() - t1 < 0.2)

    slow = Stopper(lambda q: time.sleep(5), wait_s=0.3)
    threading.Thread(target=slow, daemon=True).start()
    time.sleep(0.05)
    t2 = time.time()
    slow()
    check("a shutdown that hangs is waited for only so long", 0.25 < time.time() - t2 < 1.5, f"{time.time() - t2:.2f}s")

    boom = Stopper(lambda q: 1 / 0, wait_s=1.0)
    try:
        boom()
    except ZeroDivisionError:
        pass
    check("a shutdown that raises still lets the others go", boom.finished.is_set())


def s2_console():
    print("\n=== The console event ===")
    seen = []
    h = console_event_handler(lambda: seen.append("closed"))
    check("a closed window (2) stops ASH and says it was handled", h(2) == 1 and seen == ["closed"])
    check("a log-off (5) and a shutdown (6) too", h(5) == 1 and h(6) == 1 and len(seen) == 3)
    check("Ctrl-C (0) is left to Python's own handling", h(0) == 0 and len(seen) == 3)
    check("Ctrl-Break (1) too", h(1) == 0)
    bad = console_event_handler(lambda: 1 / 0)
    check("a shutdown that raises still returns 'handled' (and does not take the handler down)", bad(2) == 1)

    quick = []
    full = []
    import src.daemon.service as svc
    install_signal_handlers(lambda: full.append(1), on_console_close=lambda: quick.append(1))
    if sys.platform == "win32":
        check("on Windows the handler is registered with the quick path, not the full one", svc._CONSOLE_HANDLER is not None)
        svc._CONSOLE_HANDLER(2)
        check("...and a closed window runs the quick path only", quick == [1] and full == [])
    else:
        check("(not Windows: nothing to register)", True)


def s3_wiring():
    print("\n=== ashd.py ===")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ashd.py"), encoding="utf-8").read()
    check("the daemon uses the Stopper for the signal, the console close and its own loop", "Stopper(_shutdown)" in src
          and "on_console_close=lambda: shutdown(quick=True)" in src)
    check("a quick stop skips the goodbye sound", "if not quick:" in src and "play_outro=True" in src)


if __name__ == "__main__":
    s1_stopper()
    s2_console()
    s3_wiring()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
