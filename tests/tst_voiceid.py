"""
tst_voiceid.py -- knowing the user's voice from a guest's, roughly.

Uses synthetic voices (harmonics of a pitch through a vocal-tract shape,
saying random vowels), so the numbers prove the pipeline separates voices
that differ, not that it is accurate on real people.

  V1  The print: a unit vector; none for too little sound, silence, junk
  V2  Learning and forgetting: the requests, the samples, a mixed set refused,
      the print saved and deleted
  V3  Telling: the user's voice, a different one, a similar one (benefit of
      the doubt), not enrolled
  V4  The runtime: a guest is answered but cannot act or teach; the user can;
      typed input is always the user

Run:  PYTHONIOENCODING=utf-8 python tests/tst_voiceid.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.voiceid import VoiceID, VoiceIDConfig  # noqa: E402
from src.senses.voiceprint import fingerprint  # noqa: E402

check = tw.check
SR = 16000
VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
          "e": (530, 1840, 2480), "o": (570, 840, 2410)}


def voice(f0, tract, seed, secs=2.0):
    rng = np.random.default_rng(seed)
    n = int(SR * secs)
    out = np.zeros(n)
    seg = n // 5
    for k in range(5):
        fm = [f * tract for f in VOWELS[rng.choice(list(VOWELS))]]
        t = np.arange(seg) / SR
        f = f0 * (1 + 0.03 * np.sin(2 * np.pi * 3 * t + rng.random()))
        ph = 2 * np.pi * np.cumsum(f) / SR
        s = np.zeros(seg)
        for h in range(1, 40):
            if h * f0 > SR / 2 - 200:
                break
            gain = sum(np.exp(-((h * f0 - F) / (80 + 0.05 * F)) ** 2) for F in fm) + 0.02
            s += gain * np.sin(h * ph) / h ** 0.5
        out[k * seg:(k + 1) * seg] = s * np.hanning(seg)
    out += 0.003 * rng.standard_normal(n)
    return (out / np.abs(out).max() * 12000).astype(np.int16)


USER = lambda i: fingerprint(voice(110, 1.0, i))            # noqa: E731
GUEST = lambda i: fingerprint(voice(210, 1.18, 100 + i))    # noqa: E731
SIMILAR = lambda i: fingerprint(voice(125, 1.06, 200 + i))  # noqa: E731


def v1_print():
    print("\n=== V1. The print ===")
    v = USER(0)
    check("a unit vector", v is not None and abs(np.linalg.norm(v) - 1.0) < 1e-6 and len(v) == 38)
    check("too little sound: none", fingerprint(voice(110, 1.0, 0, secs=0.5)) is None)
    check("silence: none", fingerprint(np.zeros(SR * 2, dtype=np.int16)) is None)
    check("float audio works too", fingerprint(voice(110, 1.0, 0).astype(np.float32) / 32768) is not None)
    check("junk: none", fingerprint("not audio") is None and fingerprint(None) is None)


def enrolled(path=None):
    v = VoiceID(VoiceIDConfig(), path=path)
    v.hear_request("learn my voice")
    for i in range(6):
        r = v.learn(USER(i))
    v.done = r
    return v


def v2_learning():
    print("\n=== V2. Learning and forgetting ===")
    v = VoiceID()
    check("not a request", v.hear_request("what's my voice like") is None)
    r = v.hear_request("ash, learn my voice")
    check("learning starts", v.learning and "next 6 things" in r)
    for i in range(5):
        check(f"(sample {i + 1} taken quietly)", v.learn(USER(i)) is None)
    r = v.learn(USER(5))
    check("after six, it knows the voice", v.enrolled and not v.learning and r.startswith("Got it"))
    check("the threshold comes from how alike the samples were", 0.85 < v.threshold < 1.0, str(v.threshold))
    check("a print outside learning is ignored", v.learn(USER(9)) is None)

    v = VoiceID()
    v.hear_request("learn my voice")
    for i in range(3):
        v.learn(USER(i))
    for i in range(3):
        r = v.learn(GUEST(i))
    check("two voices mixed: refused", not v.enrolled and "didn't sound like one voice" in r)

    p = os.path.join(tempfile.mkdtemp(), "voice.json")
    v = enrolled(p)
    check("saved", os.path.exists(p) and VoiceID(VoiceIDConfig(), path=p).enrolled)
    check("forgotten", v.hear_request("forget my voice") == "Done, I've forgotten your voice." and not v.enrolled
          and not os.path.exists(p))
    check("forgetting nothing", v.hear_request("forget my voice") == "I didn't have your voice saved.")
    check("off: no requests", VoiceID(VoiceIDConfig(enabled=False)).hear_request("learn my voice") is None)


def v3_telling():
    print("\n=== V3. Telling ===")
    v = enrolled()
    users = [v.who(USER(i)) for i in range(10, 16)]
    check("the user's voice: user", users.count("user") >= 5 and "guest" not in users, str(users))
    guests = [v.who(GUEST(i)) for i in range(10, 16)]
    check("a different voice: guest", guests.count("guest") == 6, str(guests))
    sims = [v.who(SIMILAR(i)) for i in range(10, 16)]
    check("a similar voice is never called a guest (the benefit of the doubt)", "guest" not in sims, str(sims))
    check("nothing to compare: unknown", v.who(None) == "unknown")
    check("not enrolled: unknown", VoiceID().who(USER(1)) == "unknown")
    check("stats counted", v.status()["guest"] >= 6)


def v4_runtime():
    print("\n=== V4. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    import tst_desktop as td
    clock = [tw.T0]
    out = []
    ash = tw.Ash()
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, voice=True,
                             voice_path=os.path.join(d, "v.json"), desktop=True, wonder=True,
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    rt.desktop.ex = td.FakeEx()

    def mic(text, vec):
        clock[0] += 3
        rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text=text, addressed=True,
                              salience=0.9, urgency=0.6, meta={"voice": vec}))
        rt._tick()

    def said():
        return [o[0] for o in out]

    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="learn my voice", addressed=True,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("learning starts, without the model", "teach me your voice" in said()[-1] and ash.calls == [])
    for i in range(6):
        mic(f"just talking number {i}", USER(i))
    check("and finishes after six utterances (each still answered as usual)",
          sum(s.startswith("Got it, I know your voice") for s in said()) == 1 and len(ash.calls) == 6, str(said()))
    mic("open outlook", GUEST(20))
    check("a guest asking to act is told no, nothing done", said()[-1] == "Sorry, I only do that for Khalid."
          and rt.desktop.ex.opened == [], str(said()))
    mic("my brother Omar is great", GUEST(21))
    check("a guest does not teach ASH about the user", rt.wonder.kb.relation_of("Omar") == "")
    n = len(ash.calls)
    mic("what's the time", GUEST(22))
    check("but a guest is answered", len(ash.calls) == n + 1)
    mic("open outlook", USER(30))
    check("the user can act", rt.desktop.ex.opened == ["outlook"])
    mic("open outlook", SIMILAR(31))
    check("a similar voice gets the benefit of the doubt", rt.desktop.ex.opened == ["outlook", "outlook"])
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="close word", addressed=True,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("typed input is always the user", said()[-1].startswith("Close word?"))
    check("status carries voice", rt.status()["voice"]["enrolled"] is True)


if __name__ == "__main__":
    for fn in (v1_print, v2_learning, v3_telling, v4_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
