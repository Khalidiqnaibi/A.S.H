"""
tst_watch.py -- watching something for the user, and saying when it happens.

  W1  Reading a request: download, a program closing, battery, CPU, a file;
      builds said plainly to be out of reach; and what is not a request
  W2  Setting one: checked against the machine first ("I don't see chrome
      running"), no duplicates, a cap, cancelling
  W3  Firing: each kind, once; not before; downloads only once settled and
      with no partial file left; CPU only after holding calm; expiry
  W4  The disk trend: warns when it will be full within two days, once a day
  W5  Persistence
  W6  The runtime: the request is acknowledged without the model, the alert
      is said even unasked, and kept while muted
  W7  The real probe does not crash on this machine

Run:  PYTHONIOENCODING=utf-8 python tests/tst_watch.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.watch import Probe, WatchConfig, Watcher, parse_request, wants_cancel  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check = tw.check
T0, HOUR, DAY = tw.T0, tw.HOUR, tw.DAY
GB = 1024 ** 3


class FakeProbe(Probe):
    def __init__(self):
        self.procs = {"chrome.exe", "code.exe", "npm.cmd"}
        self.bat = (40.0, True)
        self.load = 85.0
        self.files = {"~/Downloads": {"old.zip": (10, T0 - 999)}}
        self.mtimes = {"report.docx": T0 - 50}
        self.du = (400 * GB, 500 * GB)

    def processes(self):
        return None if self.procs is None else set(self.procs)

    def battery(self):
        return self.bat

    def cpu(self):
        return self.load

    def disk(self):
        return self.du

    def listdir(self, path):
        d = self.files.get(path)
        return None if d is None else dict(d)

    def mtime(self, path):
        return self.mtimes.get(path)


def wrig(**cfg):
    t = [T0]
    p = FakeProbe()
    cfg.setdefault("disk_trend", False)
    w = Watcher(WatchConfig.from_dict(cfg), probe=p, clock=lambda: t[0])
    w.t, w.p = t, p
    return w


def adv(w, dt):
    w.t[0] += dt
    return w.tick(w.t[0])


def w1_reading():
    print("\n=== W1. Reading a request ===")
    cases = {
        "tell me when the download finishes": "download",
        "let me know when my downloads are done": "download",
        "ping me when chrome closes": "process",
        "tell me when npm finishes": "process",
        "let me know once the battery is charged": "battery",
        "tell me when the battery is at 80%": "battery",
        "tell me when the cpu calms down": "cpu",
        "let me know when the computer cools down": "cpu",
        "tell me when report.docx changes": "file",
        "tell me when the build finishes": "unsupported",
        "let me know if the tests fail": "unsupported",
    }
    for text, kind in cases.items():
        r = parse_request(text)
        check(f"{text!r} -> {kind}", r is not None and r["kind"] == kind, str(r))
    check("battery 80%", parse_request("tell me when the battery is at 80%")["pct"] == 80)
    check("battery charged is 100", parse_request("tell me when the battery is charged")["pct"] == 100)
    for text in ("tell me a joke", "when does the download finish", "tell me when it finishes",
                 "let me know when the meeting finishes", "tell me when the battery", "", None, "x" * 300,
                 "tell me when chrome closes " + "x" * 300, "tell me when the battery is at 150%"):
        check(f"not a request: {str(text)[:30]!r}", parse_request(text) is None, str(parse_request(text)))
    check("cancel: all", wants_cancel("stop watching") == "")
    check("cancel: one", wants_cancel("never mind the download") == "download")
    check("not a cancel", wants_cancel("I stopped watching TV") is None)


def w2_setting():
    print("\n=== W2. Setting one ===")
    w = wrig()
    check("a running program", w.hear("tell me when chrome closes") == "Okay, I'll tell you when chrome closes.")
    check("one that isn't running", w.hear("tell me when firefox closes") == "I don't see firefox running right now.")
    check("the same twice", w.hear("tell me when chrome closes") == "I'm already watching for that.")
    w.p.bat = (95.0, True)
    check("battery already there", w.hear("tell me when the battery is at 90%") == "The battery is already at 95%.")
    w.p.bat = None
    check("no battery", w.hear("tell me when the battery is charged") == "I can't read a battery on this machine.")
    check("a missing file", w.hear("tell me when notes.txt changes") == "I can't find notes.txt.")
    check("a build is said plainly to be out of reach", "can't see builds" in w.hear("tell me when the build finishes"))
    w.p.procs = None
    check("no process list", "can't see running programs" in w.hear("tell me when code closes"))
    check("not a request: None", w.hear("what's the time") is None)
    check("(one set)", w.status()["active"] == 1)
    check("cancel one", w.hear("never mind chrome") == "Okay, I'll stop watching chrome." and w.status()["active"] == 0)
    check("cancel with nothing set", w.hear("stop watching") == "I wasn't watching anything like that.")
    w = wrig(max_watches=1)
    w.hear("tell me when chrome closes")
    check("capped", "too many" in w.hear("tell me when the download finishes"))
    w = wrig(enabled=False)
    check("off: not taken", w.hear("tell me when chrome closes") is None)


def w3_firing():
    print("\n=== W3. Firing ===")
    w = wrig()
    w.hear("tell me when chrome closes")
    check("not before", adv(w, 10) == [])
    w.p.procs.discard("chrome.exe")
    check("when it closes", adv(w, 10) == ["chrome has closed."])
    check("once", adv(w, 10) == [] and w.status()["active"] == 0)

    w = wrig()
    w.hear("tell me when the download finishes")
    w.p.files["~/Downloads"]["big.iso.crdownload"] = (5, w.t[0])
    check("in progress: nothing", adv(w, 10) == [])
    del w.p.files["~/Downloads"]["big.iso.crdownload"]
    w.p.files["~/Downloads"]["big.iso"] = (500, w.t[0] + 10)
    check("just appeared: waits for it to settle", adv(w, 10) == [])
    check("then says it, by name", adv(w, 15) == ["Your download finished: big.iso."])
    w = wrig()
    w.hear("tell me when the download finishes")
    w.p.files["~/Downloads"]["a.pdf"] = (5, w.t[0])
    w.p.files["~/Downloads"]["b.part"] = (5, w.t[0])
    check("another still downloading: waits", adv(w, 20) == [])
    del w.p.files["~/Downloads"]["b.part"]
    w.p.files["~/Downloads"]["b"] = (5, w.t[0])
    w.p.files["~/Downloads"]["old.zip"] = (10, w.t[0] - 999)
    check("then names both, and not the file that was already there",
          adv(w, 20) == ["Your downloads finished: a.pdf, b."], str(w.status()))

    w = wrig()
    w.hear("tell me when the battery is at 80%")
    check("battery below: nothing", adv(w, 10) == [])
    w.p.bat = (81.0, True)
    check("battery reached", adv(w, 10) == ["The battery is at 81%."])
    w = wrig()
    w.hear("tell me when the battery is charged")
    w.p.bat = (100.0, True)
    check("battery charged", adv(w, 10) == ["The battery is charged."])

    w = wrig()
    w.hear("tell me when the cpu calms down")
    w.p.load = 10.0
    check("calm for a moment: not yet", adv(w, 10) == [])
    w.p.load = 70.0
    adv(w, 10)
    w.p.load = 12.0
    adv(w, 10)
    check("a spike restarts the wait", adv(w, 10) == [])
    check("calm long enough", adv(w, 25) == ["The CPU has calmed down (12%)."])

    w = wrig()
    w.hear("tell me when report.docx changes")
    check("unchanged: nothing", adv(w, 10) == [])
    w.p.mtimes["report.docx"] = w.t[0]
    check("changed", adv(w, 10) == ["report.docx has changed."])

    w = wrig()
    w.hear("tell me when chrome closes")
    adv(w, 2)
    w.p.procs.discard("chrome.exe")
    check("checked only every few seconds", adv(w, 1) == [] and adv(w, 5) == ["chrome has closed."])
    w = wrig()
    w.hear("tell me when chrome closes")
    check("a watch that never fires is let go after a day", adv(w, DAY + 1) == [] and w.status()["expired"] == 1)


def w4_trend():
    print("\n=== W4. The disk trend ===")
    w = wrig(disk_trend=True)
    used = 400 * GB
    out = []
    for i in range(8):
        w.p.du = (used, 500 * GB)
        out += adv(w, HOUR)
        used += GB // 2
    check("growing slowly (200 hours to full): nothing", out == [], str(out))
    w = wrig(disk_trend=True)
    used = 480 * GB
    out = []
    for i in range(6):
        w.p.du = (used, 500 * GB)
        out += adv(w, HOUR)
        used += 2 * GB
    check("full within two days: warned once", len(out) == 1 and "full in about" in out[0], str(out))
    for i in range(5):
        w.p.du = (used, 500 * GB)
        out += adv(w, HOUR)
        used += 1 * GB
    check("not again the same day", len(out) == 1)
    w = wrig(disk_trend=True)
    out = []
    for i in range(6):
        w.p.du = (450 * GB - i * GB, 500 * GB)
        out += adv(w, HOUR)
    check("shrinking: nothing", out == [], str(out))
    for i in range(10):
        adv(w, 60)
    check("sampled once an hour, not every tick", len(w.disk_samples) == 6, str(len(w.disk_samples)))


def w5_persist():
    print("\n=== W5. Persistence ===")
    p = os.path.join(tempfile.mkdtemp(), "w.json")
    t = [T0]
    fp = FakeProbe()
    w = Watcher(WatchConfig(disk_trend=False), path=p, probe=fp, clock=lambda: t[0])
    w.hear("tell me when chrome closes")
    w2 = Watcher(WatchConfig(disk_trend=False), path=p, probe=fp, clock=lambda: t[0])
    check("a restart keeps the watch", len(w2.active()) == 1)
    fp.procs.discard("chrome.exe")
    t[0] += 10
    check("and it still fires", w2.tick() == ["chrome has closed."])
    w3 = Watcher(WatchConfig(disk_trend=False), path=p, probe=fp, clock=lambda: t[0])
    check("a fired one is gone after a restart", w3.active() == [])


def rt_rig(muted=False):
    clock = [T0]
    out = []
    ash = tw.Ash()
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, watch=True,
                             watch_cfg={"disk_trend": False}),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    probe = FakeProbe()
    rt.watcher.probe = probe
    rt.watcher.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.clock, h.probe, h.idle = rt, ash, out, clock, probe, idle

    def say(text, source="console", addressed=True):
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()

    def step(dt):
        clock[0] += dt
        idle.value = 300.0
        rt._tick()

    h.say, h.step = say, step
    h.said = lambda: [o[0] for o in out]
    return h


def w6_runtime():
    print("\n=== W6. The runtime ===")
    h = rt_rig()
    h.say("tell me when chrome closes")
    check("acknowledged without the model", h.ash.calls == [] and h.said() == ["Okay, I'll tell you when chrome closes."])
    h.step(10)
    check("nothing yet", len(h.said()) == 1)
    h.probe.procs.discard("chrome.exe")
    h.step(10)
    check("said when it happens, unasked", h.said()[-1] == "chrome has closed.", str(h.said()))
    check("as a watch", h.out[-1][1]["pathway"] == "watch")

    h = rt_rig()
    h.say("tell me when chrome closes")
    h.rt.gate.mute(True)
    h.probe.procs.discard("chrome.exe")
    h.step(10)
    check("muted: kept, not lost", h.said()[-1] != "chrome has closed." and h.rt._watch_pending == ["chrome has closed."])
    h.rt.gate.mute(False)
    h.step(1)
    check("said once unmuted", h.said()[-1] == "chrome has closed." and h.rt._watch_pending == [])

    h = rt_rig()
    h.say("tell me when chrome closes", source="screen", addressed=False)
    check("text on the screen is not a request", h.rt.watcher.active() == [])
    h = rt_rig()
    h.say("tell me when the build finishes")
    check("a build: the honest answer, no model", "can't see builds" in h.said()[-1] and h.ash.calls == [])
    h.say("what's the time")
    check("anything else goes to the model as usual", h.ash.calls == ["what's the time"])
    check("status carries watch", h.rt.status()["watch"]["set"] == 0)


def w7_probe():
    print("\n=== W7. The real probe ===")
    p = Probe()
    for name in ("processes", "battery", "cpu", "disk"):
        try:
            getattr(p, name)()
            ok = True
        except Exception as e:
            ok, name = False, f"{name}: {e}"
        check(f"{name} does not raise", ok)
    check("listdir of a missing folder is None", p.listdir("~/no-such-folder-xyz") is None)
    check("mtime of a missing file is None", p.mtime("~/no-such-file.xyz") is None)


if __name__ == "__main__":
    for fn in (w1_reading, w2_setting, w3_firing, w4_trend, w5_persist, w6_runtime, w7_probe):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
