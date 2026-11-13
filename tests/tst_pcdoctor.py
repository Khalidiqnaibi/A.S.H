"""
tst_pcdoctor.py -- "why is my computer slow?" from the real numbers, by rule.

  P1  the ways people ask, and what is not asking
  P2  what is wrong, biggest first: memory, processor, swapping, a full drive, days without a restart, battery
  P3  "what's using my memory / cpu": grouped by program, in plain units
  P4  all fine: says so, offers nothing
  P5  "want me to close X?" -- only on yes, through the executor; never ASH, its model server or Windows itself
  P6  in the runtime: no model call; only for a line meant for ASH; the real reader works on this machine (read-only)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_pcdoctor.py
"""

import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.pcdoctor import Group, PCDoctor, Snapshot, nice, read  # noqa: E402

check = tw.check
GB = 1024 ** 3
T = 1_800_000_000.0


def snap(mem=55.0, cpu=10.0, swap=5.0, disk_free=200 * GB, disk_used=50.0, uptime_days=1, battery=None, groups=None):
    gs = groups if groups is not None else [
        Group("chrome", "chrome.exe", 40, 3.1 * GB, 80.0), Group("python", "python.exe", 3, 2.5 * GB, 30.0),
        Group("llama-server", "llama-server.exe", 1, 4.0 * GB, 10.0), Group("spotify", "Spotify.exe", 7, 0.9 * GB, 5.0),
        Group("svchost", "svchost.exe", 90, 1.4 * GB, 4.0)]
    return Snapshot(at=T, cpu=cpu, cores=8, mem_total=16 * GB, mem_used=16 * GB * mem / 100, mem_percent=mem, swap_percent=swap,
                    disks=[("C:\\", disk_free, disk_used)], groups=sorted(gs, key=lambda g: -g.rss), uptime_s=uptime_days * 86400,
                    battery=battery)


def doc(s):
    return PCDoctor(reader=lambda: s, clock=lambda: T)


def p1_asking():
    print("\n=== P1. The ways people ask ===")
    for said, kind in [("why is my computer so slow", "slow"), ("Ash, why is my pc being laggy?", "slow"),
                       ("why is everything so slow", "slow"), ("what's slowing my laptop down", "slow"),
                       ("is my computer okay", "check"), ("check my pc", "check"), ("what's eating my ram", "memory"),
                       ("what is using all of my memory", "memory"), ("which programs are using the memory", "memory"),
                       ("what's hogging my cpu", "cpu"), ("am I running out of disk space", "disk"),
                       ("how much space do I have left", "disk")]:
        check(f"{said!r} -> {kind}", PCDoctor.parse(said) == kind, str(PCDoctor.parse(said)))
    for said in ("why is the sky blue", "my computer is new", "how is the system doing", "the memory of that day",
                 "what's using my car", "why is my brother slow"):
        check(f"not asking: {said!r}", PCDoctor.parse(said) is None, str(PCDoctor.parse(said)))


def p2_findings():
    print("\n=== P2. What is wrong, biggest first ===")
    out = doc(snap(mem=93)).answer("slow", T)
    check("memory: how full, and who", out.startswith("It's slow mainly because memory is 93% full") and "Chrome 3.1 GB" in out, out)
    out = doc(snap(cpu=97)).answer("slow", T)
    check("the processor, per program in whole-machine percent", "processor is at 97%" in out and "Chrome 10%" in out, out)
    out = doc(snap(mem=90, swap=80)).answer("slow", T)
    check("swapping is named too", "swapping" in out, out)
    out = doc(snap(mem=40, disk_free=4 * GB, disk_used=97)).answer("slow", T)
    check("a full drive", "drive C: has only 4.0 GB free" in out, out)
    out = doc(snap(mem=40, uptime_days=12)).answer("slow", T)
    check("days without a restart", "12 days" in out, out)
    out = doc(snap(mem=40, battery=(12.0, False))).answer("slow", T)
    check("on a low battery", "battery is at 12%" in out, out)


def p3_using():
    print("\n=== P3. What is using it ===")
    out = doc(snap(mem=60)).answer("memory", T)
    check("memory, grouped, in GB, biggest first", out.startswith("Memory is 60% used") and out.index("Llama-server") < out.index("Chrome")
          and "across 40 processes" in out, out)
    check("...and at 60% nothing is offered", "Want me to close" not in out, out)
    out = doc(snap(cpu=30)).answer("cpu", T)
    check("processor, per program", out.startswith("The processor is at 30%") and "Chrome" in out and "close" not in out, out)
    out = doc(snap(disk_free=300 * GB, disk_used=40)).answer("disk", T)
    check("disk: free space, and plenty", "C: has 300.0 GB free" in out and "plenty" in out, out)
    out = doc(snap(disk_free=15 * GB, disk_used=91)).answer("disk", T)
    check("disk: getting low", "getting low" in out, out)
    check("names people know", nice("msedge.exe") == "Edge" and nice("winword") == "Word" and nice("foo") == "Foo")


def p4_fine():
    print("\n=== P4. All fine ===")
    d = doc(snap())
    out = d.answer("check", T)
    check("says so, with the numbers", out.startswith("It looks fine right now: processor 10%, memory 55%") and d.pending is None, out)


def p5_closing():
    print("\n=== P5. Closing, only on yes ===")
    closed = []
    d = doc(snap(mem=93))
    out = d.answer("slow", T)
    check("offers the biggest program it may close (not the model server, not ASH)", "Want me to close Chrome?" in out
          and "Llama" not in out.split("Want me")[1], out)
    check("'no' leaves it", d.reply("no", closed.append, T) == "Okay, I'll leave it." and closed == [])
    d.answer("slow", T)
    check("anything else is not an answer", d.reply("what time is it", closed.append, T) is None and d.pending is not None)
    check("'yes' closes it, through the executor", d.reply("yes", lambda p: closed.append(p) or True, T) == "Closed Chrome."
          and closed == ["chrome.exe"])
    d.answer("slow", T)
    check("an old offer is not answered by a late yes", d.reply("yes", closed.append, T + 600) is None and len(closed) == 1)
    d.answer("slow", T)
    check("a close that fails says so", "couldn't close Chrome" in d.reply("yes", lambda p: False, T))
    only_protected = snap(mem=95, groups=[Group("python", "python.exe", 3, 6 * GB, 50), Group("llama-server", "llama-server.exe", 1, 5 * GB, 9),
                                          Group("svchost", "svchost.exe", 90, 2 * GB, 4), Group("explorer", "explorer.exe", 1, 0.3 * GB, 1)])
    d = doc(only_protected)
    out = d.answer("slow", T)
    check("only ASH, its model and Windows using it: nothing is offered", "Want me to close" not in out and d.pending is None, out)


def p6_runtime():
    print("\n=== P6. In the runtime ===")
    import tst_effects as te
    from tst_desktop import FakeEx
    rt = te.rig(pcdoctor=True, pcdoctor_reader=lambda: snap(mem=93), desktop=True)
    fake = FakeEx()
    rt.desktop.ex = fake
    n = len(rt.ash.calls)
    rt.say("Ash, why is my computer so slow?")
    check("answered by rule, no model", len(rt.ash.calls) == n and "memory is 93% full" in rt.out[-1], rt.out[-1])
    rt.say("yes")
    check("'yes' closes it through the (fake) executor", fake.closed == ["chrome.exe"] and rt.out[-1] == "Closed Chrome.", str(fake.closed))
    rt.say("why is my computer so slow", addressed=False)
    check("overheard: not answered by the doctor", "memory is" not in rt.out[-1] or len(rt.out) == 0, rt.out[-1])
    no_desk = te.rig(pcdoctor=True, pcdoctor_reader=lambda: snap(mem=93))
    no_desk.say("Ash, why is my computer so slow?")
    check("with desktop control off it does not offer what it cannot do", "Want me to close" not in no_desk.out[-1]
          and no_desk.pcdoctor.pending is None, no_desk.out[-1])
    s = read()
    check("the real reader works here (read-only)", s.mem_total > 0 and s.groups and s.cores >= 1, f"{s.mem_percent}% {len(s.groups)} groups")


def p7_off_the_loop():
    print("\n=== P7. Reading the machine never blocks the loop ===")
    from src.daemon.pcdoctor import CachedReader
    T0 = 1_800_000_000.0
    clock = [T0]
    jobs = []
    calls = []

    def fn():
        calls.append(clock[0])
        s = snap(mem=70)
        s.at = clock[0]
        return s
    cr = CachedReader(fn, clock=lambda: clock[0], fresh_s=20, max_age_s=600, spawn=jobs.append)
    check("nothing read yet: it says so rather than waiting", cr() is None and len(jobs) == 1 and not calls)
    check("asked again while the first is being taken: no second one started", cr() is None and len(jobs) == 1)
    jobs.pop(0)()
    got = cr()
    check("after the background read, the answer is instant", got is not None and len(calls) == 1 and not jobs)
    clock[0] += 5
    check("a recent one is used as it is", cr() is got and not jobs)
    clock[0] += 30
    again = cr()
    check("an old one is answered from at once, and a fresh one is started in the background",
          again is got and len(jobs) == 1 and len(calls) == 1)
    jobs.pop(0)()
    check("...and used next time", cr() is not got and len(calls) == 2)
    clock[0] += 3600
    check("one far too old is not believed", cr() is None and len(jobs) == 1)
    d = PCDoctor(reader=CachedReader(fn, clock=lambda: clock[0], spawn=lambda j: None), clock=lambda: clock[0])
    check("first look not ready: said plainly, nothing breaks", "first look" in d.answer("slow", clock[0]))
    t0 = time.time()
    real = PCDoctor()
    real.cached.warm()
    check("making the real one returns at once (the reading is on its own thread)", time.time() - t0 < 0.5, f"{time.time() - t0:.2f}s")
    t1 = time.time()
    r = real.answer("slow")
    check("...and so does the first question, answered or told to wait", time.time() - t1 < 0.5 and bool(r), f"{time.time() - t1:.2f}s")


if __name__ == "__main__":
    for fn in (p1_asking, p2_findings, p3_using, p4_fine, p5_closing, p6_runtime, p7_off_the_loop):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
