"""
tst_guard.py -- guard mode: the camera watches while you're away; your phone hears about it. The camera, the frames
and the phone are fakes.

  G1  the ways people turn it on and off
  G2  movement: two changed looks in a row (not one flicker, not camera noise), at most once every five minutes
  G3  someone unknown in view (camera presence); your own face coming back ends it
  G4  the summary when you're back; with no way to reach the phone it says so up front
  G5  in the runtime: turns the camera on and back off after; not while privacy mode is on; "I'm back" with no guard
      on is left alone; alerts go to the phone (fake KDE Connect)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_guard.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.guard import Guard  # noqa: E402

check = tw.check
T0 = 1_800_000_000.0


def still(seed=0, noise=4):
    rng = np.random.RandomState(seed)
    base = np.full((240, 320, 3), 120, np.uint8)
    return np.clip(base.astype(int) + rng.randint(-noise, noise + 1, base.shape), 0, 255).astype(np.uint8)


def moved(seed=0, x=100):
    f = still(seed)
    f[60:200, x:x + 90] = 230
    return f


def guard(sent=None, can=True):
    sent = sent if sent is not None else []
    t = [T0]
    g = Guard(notify=lambda text: sent.append(text) or True, can_alert=lambda: can, clock=lambda: t[0])
    g.t = t
    return g, sent


def step(g, frame, secs=3, **kw):
    g.t[0] += secs
    return g.look(frame, g.t[0], **kw)


def g1_asking():
    print("\n=== G1. Asking ===")
    for said, kind in [("guard mode on", "on"), ("Ash, watch the house", "on"), ("keep an eye on the place while I'm out", "on"),
                       ("I'm leaving, guard the room", "on"), ("guard mode off", "off"), ("stop guarding", "off"),
                       ("I'm back", "off"), ("ok I'm home now", "off")]:
        check(f"{said!r} -> {kind}", Guard.parse(said) == kind, str(Guard.parse(said)))
    for said in ("watch a movie", "keep an eye on the oven", "I'm back from the shop with milk and eggs and more", "guard my secrets"):
        check(f"not guard: {said!r}", Guard.parse(said) is None, str(Guard.parse(said)))


def g2_movement():
    print("\n=== G2. Movement ===")
    g, sent = guard()
    g.start(T0)
    step(g, still(1))
    for i in range(5):
        step(g, still(i + 2))
    check("camera noise is not movement", sent == [])
    step(g, moved(10, 40))
    check("one changed look is not enough", sent == [])
    step(g, still(11))
    step(g, moved(12, 40))
    check("...a flicker either", sent == [])
    out1 = step(g, moved(13, 40))
    step(g, moved(14, 200))
    out2 = step(g, moved(15, 40))
    check("two changed looks in a row: an alert", len(sent) == 1 and sent[0].startswith("ASH guard: Movement in front of the camera ("), str(sent))
    for i in range(6):
        step(g, moved(20 + i, 40 if i % 2 else 200))
    check("not again within five minutes", len(sent) == 1, str(sent))
    g.t[0] += 300
    step(g, moved(30, 40))
    step(g, moved(31, 200))
    check("...then again", len(sent) == 2, str(sent))
    check("looks are spaced out", g.look(moved(40, 40), g.t[0] + 0.5) is None)
    check("no frame (camera not delivering): nothing", step(g, None) is None)


def g3_presence():
    print("\n=== G3. Someone unknown; you coming back ===")
    g, sent = guard()
    g.start(T0)
    step(g, still(1), stranger=True)
    check("someone unknown: an alert", sent and "Someone I don't know is in view" in sent[0], str(sent))
    check("you coming back (after the first minute) ends it", step(g, still(2), secs=120, user_here=True) == "welcome")
    g2, _ = guard()
    g2.start(T0)
    check("you still in view in the first minute (leaving) does not end it", step(g2, still(1), user_here=True) is None)


def g4_summary():
    print("\n=== G4. Summaries ===")
    g, sent = guard()
    g.start(T0)
    g.t[0] += 600
    check("nothing happened", g.stop(g.t[0]) == "Guard mode off. Nothing happened in the 10 minutes you were away.")
    g.start(g.t[0])
    step(g, still(1), stranger=True)
    out = g.stop(g.t[0] + 60, welcome=True)
    check("what happened, with times", out.startswith("Welcome back -- guard mode off. While you were away: Someone I don't know is in view at "), out)
    nobody, _ = guard(can=False)
    check("no way to reach the phone: said up front", "no way to reach your phone" in nobody.start(T0))
    check("already on / not on", nobody.start(T0) == "Guard mode is already on." and guard()[0].stop(T0) == "Guard mode isn't on.")


class Cam:
    def __init__(self, on=False):
        from src.senses.base import State
        self.S = State
        self.state = State.ACTIVE if on else State.DISABLED
        self.name, self.last_frame, self.last_frame_ts = "camera", None, 0.0

    def enable(self):
        self.state = self.S.ACTIVE

    def disable(self):
        self.state = self.S.DISABLED

    def due(self, now):
        return False

    def close(self):
        pass

    def status(self):
        return {}


def g5_runtime():
    print("\n=== G5. In the runtime ===")
    import time as _t
    import tst_effects as te
    from tst_phone import FakeCLI
    cli = FakeCLI()
    rt = te.rig(phone=True, phone_io={"cli": "kdeconnect-cli", "runner": cli}, guard=True)
    cam = Cam(on=False)
    rt.periphery.sensors["camera"] = cam
    rt.say("Ash, guard mode on")
    check("on: the camera is turned on for it", rt.guard.active and cam.state.value == "active" and "send anything I see to your phone" in rt.out[-1], rt.out[-1])
    for i in range(4):
        cam.last_frame, cam.last_frame_ts = (moved(i, 40 if i % 2 else 200) if i else still(0)), _t.time()
        rt.t[0] += 3
        rt._tick()
    pings = [c for c in cli.calls if "--ping-msg" in c]
    check("movement went to the phone (fake KDE Connect), words only", len(pings) == 1 and pings[0][-1].startswith("ASH guard: Movement"), str(pings))
    rt.say("Ash, I'm back")
    check("'I'm back': off, with what happened, and the camera back off", not rt.guard.active and "While you were away: Movement" in rt.out[-1]
          and cam.state.value != "active", rt.out[-1])
    n = len(rt.out)
    rt.say("Ash, I'm back")
    check("'I'm back' with no guard on is left alone (not answered by the guard)", not any("Guard mode" in o for o in rt.out[n:]), str(rt.out[n:]))
    rt.privacy.pause("test")
    rt.say("Ash, guard mode on")
    check("not while privacy mode is on", not rt.guard.active and "privacy mode" in rt.out[-1], rt.out[-1])
    rt.privacy.resume()
    on_cam = Cam(on=True)
    rt2 = te.rig(guard=True)
    rt2.periphery.sensors["camera"] = on_cam
    rt2.say("Ash, watch the house")
    check("no phone: says so", "no way to reach your phone" in rt2.out[-1], rt2.out[-1])
    rt2.say("Ash, guard mode off")
    check("a camera that was already on stays on", on_cam.state.value == "active")
    rt3 = te.rig(guard=True)
    del rt3.periphery.sensors["camera"]           # (and so no real camera class can ever be switched on here)
    rt3.say("Ash, guard mode on")
    check("no camera at all: says so", "no camera" in rt3.out[-1].lower() or "There's no camera" in rt3.out[-1], rt3.out[-1])


if __name__ == "__main__":
    for fn in (g1_asking, g2_movement, g3_presence, g4_summary, g5_runtime):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
