"""
tst_replay.py -- real utterances from the real sessions, through the real brain.

The sessions of 2026-11-06 and 2026-11-07 showed every reply taking the model
(20-70 s), including "what's 15 times 23". The cause was in the brain's own
feedback loops (see tst_fastpath). This test keeps the sentences that were
actually said, and states where each should go, so a change that quietly pushes
them back to the model fails here instead of in the user's evening:

  fast   a clear request for a tool: runs the tool, no language model
  slow   conversation, or something that needs thought: the model answers
  rule   a control command: handled by a rule before the brain is asked (the
         daemon, not the brain -- checked through the runtime)

Everything is real except the language model, which is a stub that counts its
calls. The brain starts from a fresh state (the warm-start critic).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_replay.py        (about a minute; loads the real encoder)
"""

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
STATE = tempfile.mkdtemp()
os.environ["ASH_BRAIN_STATE"] = os.path.join(STATE, "brain")
os.environ["ASH_LLM_MODE"] = "local"

import logging  # noqa: E402

logging.disable(logging.CRITICAL)

# The memory engines write these tracked files in the working directory.
_SNAP = {}
for _f in ("entities.json", "episodic_memory.jsonl"):
    _p = os.path.join(ROOT, _f)
    _SNAP[_f] = open(_p, "rb").read() if os.path.exists(_p) else None

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


CALLS = {"n": 0}


class StubLLM:
    def invoke(self, *a, **k):
        CALLS["n"] += 1
        return type("R", (), {"content": "stub"})()

    def __call__(self, *a, **k):
        CALLS["n"] += 1
        return "stub"

    def generate(self, *a, **k):
        CALLS["n"] += 1
        return "stub"


# What was really said (2026-11-06 / 2026-11-07), and where it should go.
FAST = [
    "what time is it",
    "what's 15 times 23",
    "convert 10 km to miles",
    "list the files in my documents",
    "how much memory am I using",
    "Ash, what's 7 times 8 please",
    "can you tell me the time please",
]
SLOW = [
    "Ash, sweet.",
    "tell me about yourself",
    "I'm not sure what you mean by that",
    "that's not what I had in mind",
    "why do you think the sky is blue and what would change if it weren't",
    "I'm feeling a bit tired today",
]
RULES = {
    # what was said -> what the daemon did (checked in the runtime, with no model call)
    "turn on the mic": "Microphone on.",
    "turn off the mic": "Microphone off.",
    "go to privacy mode": "Privacy mode on",
    "privacy off": "Privacy mode off",
    "smile for me": "Like this.",
    "what faces can you make?": "I can look",
    "go to sleep": "Going quiet",
    "wake up": "I'm back.",
}


def brain_part():
    from src.py.ash import ASH
    ash = ASH(llm=StubLLM())
    print("\n=== R1. Clear tool requests go fast ===")
    got = {}
    for q in FAST:
        CALLS["n"] = 0
        try:
            r = ash.brain.think(q)
        except Exception as e:                                  # a tool that needs a file may not run here
            check(f"{q!r}", False, f"raised {type(e).__name__}: {e}")
            continue
        got[q] = (r.pathway.value, CALLS["n"])
        check(f"{q!r} is fast, with no model call", r.pathway.value in ("fast", "reflex") and CALLS["n"] == 0,
              f"{r.pathway.value}, llm calls={CALLS['n']}")
    print("\n=== R2. Conversation goes to the model ===")
    for q in SLOW:
        r = ash.brain.think(q)
        check(f"{q!r} is slow", r.pathway.value in ("slow", "escalated"), r.pathway.value)
    print("\n=== R3. The same, second time round (the reflex has learned them) ===")
    for q in ("what time is it", "what's 16 times 23"):
        CALLS["n"] = 0
        r = ash.brain.think(q + " please")
        check(f"{q!r} (again) stays off the model", CALLS["n"] == 0 and r.pathway.value in ("fast", "reflex"),
              f"{r.pathway.value}, llm calls={CALLS['n']}")
    return ash


def rule_part():
    print("\n=== R4. Commands are rules, not model calls ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    import tst_wondering as tw
    out = []

    class Sensor:
        def __init__(self, name, on=False):
            from src.senses.base import State
            self.S = State
            self.name, self.state = name, (State.ACTIVE if on else State.DISABLED)

        def enable(self):
            self.state = self.S.ACTIVE

        def disable(self):
            self.state = self.S.DISABLED

        def due(self, now):
            return False

    class Player:
        library = {n: object() for n in ("happy", "sad", "wink", "idle", "boot")}
        current = None

        def play(self, name, force=False):
            self.current = type("A", (), {"name": name})()
            return True

        def set_idle(self, n):
            pass

    per = PeripheryRegistry()
    for n in ("microphone", "camera", "screenshot"):
        per.sensors[n] = Sensor(n, on=(n == "microphone"))

    class Ash(tw.Ash):
        def think(self, query, image=None, context=""):
            CALLS["n"] += 1
            self.calls.append(query)
            r = tw.Resp("stub")
            r.pathway = type("P", (), {"value": "slow"})()
            return r
    ash = Ash()
    clock = [tw.T0]
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9),
                        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    from src.face.bridge import FaceBridge
    rt.face = FaceBridge(Player())
    for text, want in RULES.items():
        clock[0] += 3
        CALLS["n"] = 0
        src = "console" if text in ("privacy off",) else "microphone"
        rt.submit(SensorEvent(source=src, modality=Modality.TEXT, text=text, addressed=True, salience=0.9, urgency=0.6))
        rt._tick()
        check(f"{text!r} -> {want!r}, no model call", out and out[-1].startswith(want) and CALLS["n"] == 0,
              f"{out[-1:]} calls={CALLS['n']}")


if __name__ == "__main__":
    t0 = time.time()
    try:
        brain_part()
        rule_part()
    finally:
        for f, data in _SNAP.items():                           # leave the tracked memory files as they were
            p = os.path.join(ROOT, f)
            try:
                if data is None:
                    if os.path.exists(p):
                        os.remove(p)
                else:
                    open(p, "wb").write(data)
            except OSError:
                pass
    print(f"\n({time.time() - t0:.0f}s)\n" + ("ALL PASS" if OK else "FAILURES"))
    sys.exit(0 if OK else 1)
