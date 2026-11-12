"""
tst_misread.py -- the morning ASH misread "what do you recommend".

What happened: at 08:00 ASH said the disk was filling up fast (from two
hours of samples -- a burst of downloads); at 08:32 "what do you recommend"
went to the model, which answered something generic; at 08:48 "not what I
had in mind nope .." was not read as a correction, went to the model again,
and the machine crashed while the GPU was loading it.

  X1  "Not what I had in mind" and its kin are corrections
  X2  ...even a while after the answer, when it is still the last one;
      a bare "no" is not, after two minutes
  X3  The disk trend: a burst of downloads is not "filling up fast"; a
      steady climb still is
  X4  Disk advice: what is free, whether it is falling, the quick wins,
      the biggest things in Downloads
  X5  The runtime: "what do you recommend" after the disk warning gets the
      advice without the model; without the warning it goes to the model;
      "how do I free up space" any time
  X6  The model off the GPU: llm.gpu = false sends num_gpu 0 to Ollama,
      switchable live from the console

Run:  PYTHONIOENCODING=utf-8 python tests/tst_misread.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
import tst_corrections as tc  # noqa: E402
import tst_watch as tws  # noqa: E402
from src.brain import corrections  # noqa: E402
from src.brain.corrections import read_correction  # noqa: E402

check = tw.check
T0, HOUR = tw.T0, tw.HOUR
GB = 1024 ** 3


def x1_phrases():
    print("\n=== X1. 'Not what I had in mind' ===")
    for text, want in (("not what i had in mind nope ..", ""), ("not what I meant", ""), ("that is not what i wanted", ""),
                       ("that wasn't what I asked for", ""), ("not quite what i was looking for", ""),
                       ("it's not what i had in mind", ""), ("nope, not what i had in mind. i meant disk space", "disk space"),
                       ("not what I asked, I meant the disk", "the disk")):
        check(f"{text!r} -> {want!r}", read_correction(text) == want, repr(read_correction(text)))
    for text in ("what i had in mind was pizza", "not now", "no thanks", "nope", "that's what I meant", "not bad"):
        check(f"not a correction: {text!r}", read_correction(text) is None, repr(read_correction(text)))


def x2_late():
    print("\n=== X2. A while later ===")
    old = corrections.default()
    try:
        corrections.configure(None)
        h = tc.rt_rig()
        h.say("remind me about the weather")
        n = len(h.ash.calls)
        h.say("not what i had in mind nope ..", dt=16 * 60)
        check("16 minutes on, still the last answer: asked what was wanted, no model",
              h.said()[-1] == "Sorry -- what did you want?" and len(h.ash.calls) == n, str(h.said()[-3:]))
        corrections.configure(None)
        h = tc.rt_rig()
        h.say("remind me about the weather")
        n = len(h.ash.calls)
        h.say("no, I meant search my files", dt=16 * 60)
        check("a plain 'no, I meant' that late is a new request", h.ash.calls[-1] == "no, I meant search my files"
              and len(h.ash.calls) == n + 1, str(h.ash.calls[-1:]))
        corrections.configure(None)
        h = tc.rt_rig()
        h.say("remind me about the weather")
        n = len(h.ash.calls)
        h.say("not what i had in mind", dt=31 * 60)
        check("after half an hour: not a correction", len(h.ash.calls) == n + 1)
    finally:
        corrections._DEFAULT = old


class Probe(tws.FakeProbe):
    def __init__(self):
        super().__init__()
        self.big = [("ubuntu.iso", 5 * GB), ("Setup.exe", 300 * 1024 ** 2)]

    def biggest(self, path, n=3, budget_s=2.0):
        return list(self.big)


def x3_trend():
    print("\n=== X3. The disk trend ===")
    w = tws.wrig(disk_trend=True)
    used = 211.5 * GB
    out = []
    for step in (0, 3.2, 0.8, 0.0, 0.0, 0.1, 0.0):          # a burst of downloads, then flat
        used += step * GB
        w.p.du = (used, 250 * GB)
        out += tws.adv(w, HOUR)
    check("a burst of downloads is not 'filling up fast'", out == [], str(out))
    w = tws.wrig(disk_trend=True)
    used = 220 * GB
    out = []
    for _ in range(3):
        used += 2 * GB
        w.p.du = (used, 250 * GB)
        out += tws.adv(w, HOUR)
    check("two hours of growth is not enough to say", out == [], str(out))
    for _ in range(4):
        used += 2 * GB
        w.p.du = (used, 250 * GB)
        out += tws.adv(w, HOUR)
    check("a steady climb still is", len(out) == 1 and "full in about" in out[0], str(out))
    w = tws.wrig(disk_trend=True)
    used = 220 * GB
    out = []
    for step in (2, 2, 2, 2, -3):                            # growing, but it just dropped
        used += step * GB
        w.p.du = (used, 250 * GB)
        out += tws.adv(w, HOUR)
    check("not while it is going down", out == [], str(out))


def x4_advice():
    print("\n=== X4. Disk advice ===")
    w = tws.wrig(disk_trend=True)
    w.p = Probe()
    w.probe = w.p
    w.p.du = (210 * GB, 250 * GB)
    s = w.disk_advice(w.t[0])
    check("free space", s.startswith("You have 40 GB free of 250 GB."), s)
    check("the quick wins", "Recycle Bin" in s and "Temporary files" in s and "Storage Sense" in s)
    check("the biggest things in Downloads", "ubuntu.iso (5.0 GB), Setup.exe (300 MB)" in s, s)
    for used in (210, 212, 214):
        w.p.du = (used * GB, 250 * GB)
        tws.adv(w, HOUR)
    s = w.disk_advice(w.t[0])
    check("falling: by how much", "about 2.0 GB an hour" in s, s)
    w2 = tws.wrig(disk_trend=True)
    w2.p = Probe()
    w2.probe = w2.p
    for _ in range(3):
        tws.adv(w2, HOUR)
    check("steady: said so", "isn't dropping right now" in w2.disk_advice(w2.t[0]))
    w2.p.du = None
    check("the disk can't be read: nothing", w2.disk_advice(w2.t[0]) is None)
    real = tws.Probe()
    d = tempfile.mkdtemp()
    open(os.path.join(d, "a.bin"), "wb").write(b"x" * 5000)
    os.makedirs(os.path.join(d, "sub"))
    open(os.path.join(d, "sub", "b.bin"), "wb").write(b"x" * 9000)
    open(os.path.join(d, "c.txt"), "wb").write(b"x" * 10)
    got = real.biggest(d, n=2)
    check("the real probe: biggest first, folders counted with what is in them", got == [("sub\\", 9000), ("a.bin", 5000)],
          str(got))
    check("a missing folder: nothing", real.biggest(d + "_nope") == [])


def x5_runtime():
    print("\n=== X5. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, watch=True,
                             watch_path=os.path.join(tempfile.mkdtemp(), "w.json"),
                             watch_cfg={"disk_trend": False}),
        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    p = Probe()
    p.du = (210 * GB, 250 * GB)
    rt.watcher.probe = p

    def say(text, dt=3):
        clock[0] += dt
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6))
        rt._tick()

    say("what do you recommend")
    check("no disk warning: the model answers", ash.calls == ["what do you recommend"])
    rt._watch_pending.append("Your disk is filling up fast: at this rate it will be full in about 20 hours.")
    rt._watch_step(clock[0])
    check("(the warning is said)", out[-1].startswith("Your disk is filling up fast"))
    say("what do you recommend", dt=32 * 60)
    check("32 minutes later, 'what do you recommend' is about the disk, no model",
          out[-1].startswith("You have 40 GB free") and ash.calls == ["what do you recommend"], str(out[-1:]))
    say("what do you recommend", dt=4 * HOUR)
    check("hours later it is a question for the model again", ash.calls[-1] == "what do you recommend"
          and len(ash.calls) == 2)
    say("how do I free up some disk space?")
    check("asked outright, any time", out[-1].startswith("You have 40 GB free") and len(ash.calls) == 2)
    say("storage recommendations")
    check("'storage recommendations'", out[-1].startswith("You have 40 GB free") and len(ash.calls) == 2)


def x6_gpu():
    print("\n=== X6. The model off the GPU ===")
    import importlib
    L = importlib.import_module("tools.LLM")
    sent = []

    class R:
        status_code = 200
        text = '{"message": {"content": "hi"}, "done": true}'

    real = L.requests.post
    L.requests.post = lambda url, json=None, timeout=None, **k: (sent.append(json), R())[1]
    old = os.environ.pop("ASH_OLLAMA_NUM_GPU", None)
    try:
        m = L.LLM(mode="ollama")
        m._call_ollama([{"role": "user", "content": "x"}])
        check("normally: Ollama decides", "num_gpu" not in sent[-1]["options"], str(sent[-1]["options"]))
        os.environ["ASH_OLLAMA_NUM_GPU"] = "0"
        m._call_ollama([{"role": "user", "content": "x"}])
        check("llm.gpu off: num_gpu 0", sent[-1]["options"].get("num_gpu") == 0, str(sent[-1]["options"]))
        os.environ["ASH_OLLAMA_NUM_GPU"] = "junk"
        m._call_ollama([{"role": "user", "content": "x"}])
        check("junk is ignored", "num_gpu" not in sent[-1]["options"])
        os.environ.pop("ASH_OLLAMA_NUM_GPU", None)
        from src.ui import settings as st
        p = os.path.join(tempfile.mkdtemp(), "daemon.json")
        r = st.change(None, p, "llm.gpu", "off")
        check("switched off from the console, live", r["ok"] and r["live"] and os.environ.get("ASH_OLLAMA_NUM_GPU") == "0")
        check("and listed as off", {x["key"]: x for x in st.listing(None, p)}["llm.gpu"]["value"] is False)
        r = st.change(None, p, "llm.gpu", "on")
        check("and on again", r["ok"] and "ASH_OLLAMA_NUM_GPU" not in os.environ)
        import json
        check("saved", json.load(open(p))["llm"]["gpu"] is True)
    finally:
        L.requests.post = real
        os.environ.pop("ASH_OLLAMA_NUM_GPU", None)
        if old is not None:
            os.environ["ASH_OLLAMA_NUM_GPU"] = old


if __name__ == "__main__":
    for fn in (x1_phrases, x2_late, x3_trend, x4_advice, x5_runtime, x6_gpu):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
