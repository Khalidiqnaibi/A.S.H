"""
tst_modellast.py -- the language model is the last resort.

The real lines of the sessions of 17-23 Nov that reached the model although a rule, a tool, memory or ASH's own record could answer them
(62 of 112 addressed lines went to it; about one in ten needed to). Each is fed to a runtime with the features the shipped config turns on,
and a model that counts its calls and answers "MODEL". What should answer each is named; if the model did, this fails -- so a change that
quietly sends one back to it fails here, not in the user's evening.

A line that really is open (a diagram of a chip, "how can you help me build...") is in OPEN: those are the model's, and it must still get them.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_modellast.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

check = tw.check
T0 = tw.T0

# (what was said, what should answer it)
ANSWERED_WITHOUT_THE_MODEL = [
    # ASH's memory and the people it knows
    ("who is khalid in your memory", "arbiter / notes"),
    ("What's your name?", "who it is, by rule"),
    ("what do you reme", "notes (a truncated 'remember')"),
    ("What can you do then?", "capabilities"),
    ("What can you do with Google Docs?", "capabilities: it has nothing for it, said so"),
    # control
    ("cam off", "voice control"),
    ("camera off", "voice control"),
    ("Cry for me, Ash.", "faces"),
    ("Did you do it?", "its own record of the last thing it did"),
    ("Why did you take that long to read the clock?", "the brain's own timing"),
    ("Give me your emotional telemetry.", "feelings"),
    # about ASH's own state and what it did
    # code and screen
    ("What do you think of the code I'm working on right now?", "it cannot see: said so, at once"),
    ("What do you think of the function I'm highlighting right now?", "it cannot see: said so, at once"),
    ("what do you think of my code", "it cannot see: said so, at once"),
    # complaints
    ("you said the same joke 6 times already", "jokes: the book"),
    ("You are so dumb today.", "a complaint: owned, with what went wrong"),
    ("You are hallucinating, boy.", "a complaint: owned, with what went wrong"),
    # short replies with nothing pending
    ("kk", "an acknowledgement"),
    ("Later", "an acknowledgement"),
    ("will do thanks", "courtesy"),
    # statements: learned and acknowledged
    ("I'm just building my custom hardware chip ash.", "a statement: kept, acknowledged"),
    ("Ash, I've bought a 3D printer and I worked on it for a little bit.", "a statement: kept, acknowledged"),
]

# Lines that still reach the model, to be taken one at a time: each is a known gap. When one is fixed this says so, and it moves out of
# this list into the table above.
TODO = {
    "what do you reme",
    "You are so dumb today.",
    "You are hallucinating, boy.",
    "I'm just building my custom hardware chip ash.",
    "Ash, I've bought a 3D printer and I worked on it for a little bit.",
}

# Open questions: the model's, and it must still get them.
OPEN = [
    "Draw me a diagram of a single-purpose hardware for LLMs, always on",
    "How can you help me build a hardware single purpose?",
]


def rig():
    from src.face.bridge import FaceBridge
    clock = [T0]
    out, calls = [], []
    d = tempfile.mkdtemp()

    class Ash(tw.Ash):
        def think(self, query, image=None, context=""):
            calls.append(query)
            self.calls.append(query)
            r = tw.Resp("MODEL")
            r.pathway = type("P", (), {"value": "slow"})()
            return r
    ash = Ash()

    class Player:
        library = {n: object() for n in ("happy", "sad", "wink", "idle", "boot")}
        current = None

        def play(self, name, force=False):
            self.current = type("A", (), {"name": name})()
            return True

        def set_idle(self, n):
            pass
    # What the shipped config turns on, without the features that reach outside (media keys, the network, the phone, the mail).
    cfg = AmbientConfig(
        tick_seconds=0.02, sleep_after_idle_s=10 ** 9, wonder=True, wonder_path=os.path.join(d, "w.json"),
        knowledge_path=os.path.join(d, "k.json"), notes=True, notes_path=os.path.join(d, "n.json"), taste=True, feelings=True,
        feelings_path=os.path.join(d, "f.json"), fun=True, fun_path=os.path.join(d, "fun.json"), jokes_path=os.path.join(d, "j.json"),
        reminders=True, reminders_path=os.path.join(d, "r.json"), focus=True, arbiter=True, arbiter_path=os.path.join(d, "a.json"),
        courtesy=True, activity=True, activity_path=os.path.join(d, "act.json"), play=True, dwell=True, asks=True, moments=True)
    rt = AmbientRuntime(ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(), config=cfg, response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    rt.face = FaceBridge(Player())

    def say(text, source="console"):
        clock[0] += 5
        n = len(calls)
        before = len(out)
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=True, salience=0.9, urgency=0.6))
        rt._tick()
        return len(calls) > n, out[before:]
    rt.say, rt.calls = say, calls
    return rt


def main():
    print("\n=== M1. The model is the last resort ===")
    went = []
    for text, tier in ANSWERED_WITHOUT_THE_MODEL:
        rt = rig()
        used, said = rt.say(text)
        if text in TODO:
            check(f"(known gap) {text[:60]!r} -> {tier}", used, "this one is answered now: move it out of TODO")
        else:
            check(f"{text[:60]!r} -> {tier}", not used and bool(said), f"model={used} said={said[-1:]}")
        if used:
            went.append(text)
    print(f"\n  {len(went)} of {len(ANSWERED_WITHOUT_THE_MODEL)} still reach the model")

    print("\n=== M2. What is really open still goes to it ===")
    for text in OPEN:
        rt = rig()
        used, said = rt.say(text)
        check(f"{text[:60]!r} is the model's", used, str(said[-1:]))

    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
