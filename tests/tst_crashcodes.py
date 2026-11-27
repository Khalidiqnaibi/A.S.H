"""tst_crashcodes.py -- 26 Nov: "The computer shut down unexpectedly at 11:42 today (stop code 0x10E). That's 7 times in the last 30 days, which is a
pattern: worth testing the memory" -- 0x10E is the graphics driver, and the advice was the generic one. Also: Ollama started when it is installed
and not answering (it was off at 20:35). No process is ever started here: the spawn is a fake.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_crashcodes.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
from src.daemon.crashwatch import CrashWatch  # noqa: E402
from tools.LLM import ensure_ollama  # noqa: E402

check = tw.check
T0 = 1_800_000_000.0
HOUR, DAY = 3600.0, 86400.0


def watch(log):
    c = CrashWatch(path=os.path.join(tempfile.mkdtemp(), "c.json"), clock=lambda: T0, reader=lambda: list(log), background=False)
    c.refresh(force=True)
    return c


def main():
    print("\n=== K1. The code in words, and advice that fits it ===")
    log = [{"t": T0 - 3 * HOUR, "code": 0x10E}, {"t": T0 - 2 * DAY, "code": 0x10E}, {"t": T0 - 4 * DAY, "code": 0x116}, {"t": T0 - 6 * DAY, "code": 0},
           {"t": T0 - 9 * DAY, "code": 0x10E}, {"t": T0 - 12 * DAY, "code": 0}, {"t": T0 - 20 * DAY, "code": 0x113}]
    line = watch(log).hear("any crashes lately", T0)
    check("0x10E is named, not just numbered", "graphics driver's video memory manager" in line and "0x10E" not in line.split("(")[0], line)
    check("the month is tallied by cause", "7 times in the last 30 days; 5 of them were the graphics driver" in line, line)
    check("and the advice is about the graphics driver, not the memory test", "graphics driver, since that is what keeps failing" in line
          and "mdsched" not in line, line)
    log2 = [{"t": T0 - 3 * HOUR, "code": 0x7A}, {"t": T0 - 2 * DAY, "code": 0x154}, {"t": T0 - 5 * DAY, "code": 0}]
    line = watch(log2).hear("any crashes lately", T0)
    check("disk codes point at the disk", "2 of them were the disk" in line and "chkdsk" in line, line)
    log3 = [{"t": T0 - 3 * HOUR, "code": 10}, {"t": T0 - 5 * DAY, "code": 0}, {"t": T0 - 12 * DAY, "code": 0}]
    line = watch(log3).hear("any crashes lately", T0)
    check("no cause shared by two: the general advice, as before", "mdsched.exe" in line and "of them were" not in line, line)
    log4 = [{"t": T0 - 3 * HOUR, "code": 0x3F1}, {"t": T0 - 2 * DAY, "code": 0}]
    line = watch(log4).hear("any crashes lately", T0)
    check("an unknown code is still given as a number", "stop code 0x3F1" in line, line)

    print("\n=== K2. Starting Ollama ===")
    spawned = []
    up = [False]

    def spawn(cmd):
        spawned.append(cmd)
        up[0] = True
    r = ensure_ollama("http://127.0.0.1:11434/api/chat", which=lambda n: "C:/Ollama/ollama.exe", probe=lambda: up[0], spawn=spawn, sleep=lambda s: None)
    check("installed and not answering: started, and it answers", r == "started" and spawned == [["C:/Ollama/ollama.exe", "serve"]], f"{r} {spawned}")
    spawned.clear()
    r = ensure_ollama("http://127.0.0.1:11434/api/chat", which=lambda n: "C:/Ollama/ollama.exe", probe=lambda: True, spawn=spawn, sleep=lambda s: None)
    check("already running: left alone", r == "running" and spawned == [], r)
    os.environ["LOCALAPPDATA"] = tempfile.mkdtemp()
    r = ensure_ollama("http://127.0.0.1:11434", which=lambda n: None, probe=lambda: False, spawn=spawn, sleep=lambda s: None)
    check("not installed: nothing started", r == "absent" and spawned == [], r)
    r = ensure_ollama("http://192.168.1.20:11434", which=lambda n: "x", probe=lambda: False, spawn=spawn, sleep=lambda s: None)
    check("a server on another machine is never started from here", r == "remote" and spawned == [], r)
    r = ensure_ollama("http://127.0.0.1:11434", which=lambda n: "x", probe=lambda: False, spawn=lambda c: None, sleep=lambda s: None, wait_s=2)
    check("started but never answers: said as failed", r == "failed", r)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
