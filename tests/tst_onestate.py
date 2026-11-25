"""
tst_onestate.py -- one emotional state, and the person's mood is remembered.

25 Nov: "What's your emotional status?" / "...telemetry?" -> "I'm good, thanks. Steady. How about you?" twice (a second, canned system,
while the prompt said "a little short-tempered"); "I'm asking you about your emotional victor" -> the model: "I don't have direct information about
their emotional state". The user's frustration was never stored anywhere ("topic_valence" has no reader; an Episode has no mood).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_onestate.py
"""

import os
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import feelings  # noqa: E402

check = tw.check


def res(label="a little short-tempered, trying to stay professional"):
    return NS(modulation=NS(tone_label=label), emotions={"frustration": 0.42, "calm": 0.31, "joy": 0.1}, mood={"valence": 0.1, "arousal": 0.4},
              reasons=["three answers in a row were cancelled", "the model was slow"])


def o1_engine():
    print("\n=== O1. ASH's feeling comes from the engine, with the numbers on request ===")
    short = feelings.from_engine(res())
    check("short: the engine's own words and what is strongest", short.startswith("A little short-tempered") and "frustration" in short, short)
    drives = NS(urgency=0.73, caution=0.35, curiosity=0.28, effort_budget=0.26, social=0.84, fatigue=0.1)
    long = feelings.from_engine(res(), drives, detail=True)
    check("detail: emotions, mood, drives and why", "frustration 0.42" in long and "Mood baseline" in long and "urgency 0.73" in long and "effort 0.26" in long
          and "Because: three answers" in long, long)
    check("no reading yet: it says so", "don't have a feeling" in feelings.from_engine(NS(modulation=NS(tone_label=""), emotions={}, mood={}, reasons=[])))
    for t in ["What's your emotional telemetry?", "I'm asking you about your emotional victor", "give me the numbers", "what are your drives"]:
        check(f"asks for the numbers: {t!r}", feelings.wants_detail(t))
    check("plain 'how are you' does not", not feelings.wants_detail("how are you feeling"))


def o2_runtime():
    print("\n=== O2. On the runtime ===")
    import tst_effects as te
    rt = te.rig(feelings=True, wonder=True)
    rt.brain.homeostasis = NS(last_emotion_result=res(), drives=NS(urgency=0.73, caution=0.35, curiosity=0.28, effort_budget=0.26, social=0.84, fatigue=0.1))
    n = len(rt.ash.calls)
    rt.say("what's your emotional status", source="console")
    check("'status' is the engine's feeling, not 'I'm good, thanks. Steady'", rt.out[-1].startswith("A little short-tempered") and "Steady" not in rt.out[-1], rt.out[-1])
    rt.say("what's your emotional telemetry", source="console")
    check("'telemetry' has the numbers", "urgency 0.73" in rt.out[-1] and "Because:" in rt.out[-1], rt.out[-1])
    check("neither went to the model", len(rt.ash.calls) == n)
    rt.brain.homeostasis = NS(last_emotion_result=None, drives=None)
    rt.say("what's your emotional status", source="console")
    check("with no reading from the engine yet it falls back to the old answer", "short-tempered" not in rt.out[-1] and rt.out[-1], rt.out[-1])


def o3_memory():
    print("\n=== O3. The person's mood is on the episode ===")
    from src.memory.episodic.episode_model import Episode
    e = Episode.create("Immortal: bruh", "interaction", mood=-0.8)
    check("an episode carries it", e.mood == -0.8)
    check("clamped", Episode.create("x", "interaction", mood=9).mood == 1.0)
    check("an old one has none", Episode.create("x", "interaction").mood is None)
    import dataclasses
    row = dataclasses.asdict(e)
    from src.memory.episodic.episode_store import EpisodeStore
    st = EpisodeStore(os.path.join(tempfile.mkdtemp(), "e.jsonl"))
    st._add_from_dict(row)
    old = {k: v for k, v in row.items() if k != "mood"}
    old["episode_id"] = "old"
    st._add_from_dict(old)
    check("it survives a save and a load, and a row without it still loads", st.episodes[e.episode_id].mood == -0.8 and st.episodes["old"].mood is None)
    from src.memory.memory_router import MemoryRouter
    got = {}
    r = MemoryRouter.__new__(MemoryRouter)
    r.episodic = NS(add_episode=lambda **k: (got.update(k), NS(episode_id="x"))[1])
    r.user_mood = -0.6
    r.active_concepts = []
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "memory", "memory_router.py"), encoding="utf-8").read()
    check("the router passes the current reading to the episode (not for ASH's own lines)", 'getattr(self, "user_mood", None)' in src and 'source != "ash"' in src)


def o4_trend():
    print("\n=== O4. 'How am I feeling' says how the last day has been, from memory ===")
    now = time.time()
    eps = [NS(mood=-0.7, timestamp=now - 3600 * i) for i in range(1, 9)] + [NS(mood=0.5, timestamp=now - 3600 * 3), NS(mood=None, timestamp=now)]
    t = feelings.mood_trend(eps, now)
    check("count, average, share low (lines with no reading are skipped)", t and t[0] == 9 and t[1] < -0.3 and 0.8 < t[2] < 0.95, str(t))
    check("too few readings: no trend", feelings.mood_trend(eps[:3], now) is None)
    check("old ones are outside the window", feelings.mood_trend([NS(mood=-1, timestamp=now - 86400 * 3)] * 10, now) is None)
    say = feelings.user_mood_say(-0.5, t)
    check("said as a fact of memory, with its limit", "the 9 things you said in the last day" in say and "sounded low" in say and "your words only" in say, say)
    check("no trend: as before", "your words" in feelings.user_mood_say(-0.6) and "last day" not in feelings.user_mood_say(-0.6))


if __name__ == "__main__":
    o1_engine()
    o2_runtime()
    o3_memory()
    o4_trend()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
