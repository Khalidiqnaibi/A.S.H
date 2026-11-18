"""
tst_telemetry.py -- the machine's own readings are perceived but never become concepts.

"memory at 93%", battery, "screen changed", window switches and motion made 23 concepts in the graph and took a quarter of
all activation. They are the body's state (homeostasis already has them); the graph is for what the person means.

  T1  What counts as telemetry (and what never does: a person's words, whatever carries them)
  T2  A telemetry percept does not grow a concept, drag a prototype, or strengthen wires; a person's sentence is noted as a candidate
  T3  The brain's ambient path passes it on

  T4  The sensors' own words and names (active_window, now_playing, "power connected", "user went away") and overheard room speech
      grow nothing; what is said TO ASH still can

Run:  PYTHONIOENCODING=utf-8 python tests/tst_telemetry.py
"""

import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_CONCEPT_STATE", tempfile.mkdtemp())
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.concepts import ConceptSystem  # noqa: E402

check = tw.check
DIM = 32


def rnd(seed):
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def is_telemetry():
    from src.concepts.telemetry import is_telemetry as f
    return f


def t1_what():
    print("\n=== T1. What is telemetry ===")
    f = is_telemetry()
    for text, source in [("memory at 93%", "system"), ("CPU at 91%", "system"), ("battery at 15% and unplugged", "system"),
                         ("screen changed while in Discord", "screen"), ("switched to Chrome (was on A.S.H for 480s)", "screen"),
                         ("motion in camera view (delta 0.16)", "camera")]:
        check(f"{text!r} from {source} is telemetry", f(text, source))
    for text, source in [("You've been on Stremio for 15 minutes without typing.", "flow"), ("Want today's trivia question?", "flow")]:
        check(f"ASH's own remark {text!r} from {source} grows nothing", f(text, source))
    for text, source in [("the lecturer said a vector is a keyword", "microphone"), ("what time is it", "console"),
                         ("my memory is at 93% right now", "microphone"), ("remind me to check memory usage", "console"),
                         ("I use 90% of my memory for tabs", "microphone")]:
        check(f"{text!r} from {source} is not", not f(text, source))


def t2_graph():
    print("\n=== T2. In the graph ===")
    cs = ConceptSystem(None, None, dim=DIM, load=False, seed=False)
    cs.net.vigilance = 0.9
    n0 = len(cs.net.concepts)
    for i in range(10):
        cs.perceive(rnd(100 + i), text=f"memory at {80 + i}%", telemetry=True)
    check("ten different telemetry percepts grow no concept", len(cs.net.concepts) == n0, f"{n0} -> {len(cs.net.concepts)}")
    w0 = cs.plasticity.stats.get("potentiated", 0) if hasattr(cs.plasticity, "stats") else 0
    cs.perceive(rnd(200), text="battery at 10%", telemetry=True)
    w1 = cs.plasticity.stats.get("potentiated", 0) if hasattr(cs.plasticity, "stats") else 0
    check("...and strengthen no wires", w1 == w0)
    for i in range(10):
        cs.perceive(rnd(300 + i), text=f"a sentence about topic {i}")
    check("whereas a person's sentences are noted as candidates (a concept is made at sleep, if they come back)",
          len(cs.net.candidates.items) >= 5 and len(cs.net.concepts) == n0, str(len(cs.net.candidates.items)))
    cs.net.add_concept(rnd(100))
    act = cs.perceive(rnd(100), text="memory at 80%", telemetry=True)
    check("a telemetry percept is still perceived (an activation to explain)", act is not None and bool(act.spread))


def t4_sensor_names_and_room():
    print("\n=== T4. The sensors' own names; the room ===")
    from src.concepts.telemetry import observed_is_telemetry
    f = is_telemetry()
    for text, source in [("now playing Without Me by Halsey on Spotify", "now_playing"), ("stopped playing Favorite by Isabel LaRosa", "now_playing"),
                         ("power connected", "system"), ("user went away from the machine", "idle"), ("user came back after 0s away", "idle"),
                         ("switched to GitHub Desktop", "active_window"),
                         ("switched to Topological Energy-Landscape Morphing Algorithm - Google Gemini - Opera at some length of window title text", "active_window"),
                         ("screen changed while in A.S.H -- a mix of text and images", "screenshot"), ("now playing X on YouTube", "ambient")]:
        check(f"{text[:44]!r} from {source} is telemetry", f(text, source))
    for text, source in [("what time is it", "console"), ("what can you do then", "microphone"), ("Later", "console"),
                         ("remind me that now playing music is annoying", "console")]:
        check(f"{text!r} from {source} is a person's, so it is not", not f(text, source))
    for text in ("Should be the remote.", "I believe you.", "She's going to fall.", "We'll clean a little.", "Bitch.", "All right, guys, it's done. Okay, let's go."):
        check(f"{text!r} heard from the room (observed) grows nothing", observed_is_telemetry(text, "microphone"))
    check("a typed line is never room speech", not observed_is_telemetry("what time is it", "console"))
    cs = ConceptSystem(None, None, dim=DIM, load=False, seed=False)
    cs.net.vigilance = 0.9
    n0 = len(cs.net.concepts)
    for i, t in enumerate(("now playing A on Spotify", "now playing B on Spotify", "power connected", "user went away from the machine")):
        cs.perceive(rnd(500 + i), text=t, telemetry=is_telemetry()(t, "now_playing"))
    check("in the graph those four grow no concept and no candidate", len(cs.net.concepts) == n0 and len(cs.net.candidates.items) == 0,
          f"{len(cs.net.concepts) - n0} {len(cs.net.candidates.items)}")


def t3_brain():
    print("\n=== T3. The ambient path ===")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "brain", "brain.py"), encoding="utf-8").read()
    check("Brain.observe judges what was only observed with room speech included", "self._conceive(bundle, telemetry=observed_is_telemetry(text, source))" in src)
    check("Brain.observe hands telemetry to the concept layer as telemetry",
          re.search(r"self\._conceive\(bundle, telemetry=(?:observed_is_telemetry|is_telemetry)\(text, source\)\)", src) is not None
          and "from src.concepts.telemetry import is_telemetry" in src)


if __name__ == "__main__":
    t1_what()
    t2_graph()
    t3_brain()
    t4_sensor_names_and_room()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
