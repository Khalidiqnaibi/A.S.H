"""
tst_followups.py -- the fixes and features that came out of reviewing the browsing and profile work.

Each section is one change and was added with it; the file grows one commit at a time. Fixes first (the bugs a probe
confirmed), then the features. Nothing here touches the real machine: no real media key, no app launch, no network.

Run:  PYTHONIOENCODING=utf-8 python tst_followups.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
SECTIONS = []


def section(fn):
    SECTIONS.append(fn)
    return fn


@section
def fix_small_model_outage():
    print("\n=== FIX: one Ollama hiccup does not switch the small model off for the session ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ash_llm_fu", os.path.join(HERE, "tools", "LLM.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    calls = []
    state = {"down": True, "missing": False}

    class R:
        status_code = 200
        text = '{"message": {"content": "ok"}}'

    class R404:
        status_code = 404
        text = 'model "tiny:1b" not found'

    def post(url, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == "tiny:1b":
            if state["missing"]:
                return R404()
            if state["down"]:
                raise ConnectionError("Ollama is restarting")
        return R()
    L.requests.post = post
    llm = L.LLM(mode="ollama", ollama_model="big:7b")
    llm.set_light_model("tiny:1b")
    msg = [L.HumanMessage(content="hi")]
    with llm.route("light"):
        said = llm._call(msg)
    check("a refused connection: the main model answers", said == "ok" and calls == ["tiny:1b", "big:7b"], f"{said!r} {calls}")
    check("...and the small model stays on", llm._light_model == "tiny:1b", repr(llm._light_model))
    state["down"] = False
    calls.clear()
    with llm.route("light"):
        llm._call(msg)
    check("...so it is used again as soon as Ollama is back", calls == ["tiny:1b"], str(calls))
    state["missing"] = True
    calls.clear()
    with llm.route("light"):
        llm._call(msg)
    check("a model that is not installed is still switched off (404)", llm._light_model == "", repr(llm._light_model))
    calls.clear()
    with llm.route("light") as on:
        llm._call(msg)
    check("...and not tried again", calls == ["big:7b"] and not on, str(calls))


@section
def fix_exploration_in_new_activities():
    print("\n=== FIX: a new activity context is not explored harder than the situation it came from ===")
    from src.daemon.social import SocialLearner
    now = 1_800_000_000.0
    lr = SocialLearner(clock=lambda: now)
    for _ in range(30):
        lr.update("moment", "idle/evening", 1.0, 1.0, now)
    coarse = lr.exploration("moment", "idle/evening", 0.0, now)
    fine = lr.exploration("moment", "idle/evening/coding", 0.0, now)
    check("with 30 reactions in 'idle/evening', 'idle/evening/coding' is no keener to speak (it was 0.10 vs 0.02)",
          fine <= coarse + 1e-9, f"coarse {coarse:.3f} fine {fine:.3f}")
    fresh = SocialLearner(clock=lambda: now)
    check("a kind it has never seen at all is still explored, as before",
          abs(fresh.exploration("joke", "idle/evening/coding", 0.0, now) - fresh.explore) < 1e-9)
    for _ in range(20):
        lr.update("joke", "away/night/gaming", 1.0, 1.0, now)          # lots known about jokes while gaming
    check("what is known about the kind during that activity counts too",
          lr.exploration("joke", "idle/evening/gaming", 0.0, now) < 0.03, str(lr.exploration("joke", "idle/evening/gaming", 0.0, now)))
    check("without an activity nothing changed", abs(lr.exploration("moment", "idle/evening", 0.0, now) - coarse) < 1e-12)
    check("urgency still scales it down", lr.exploration("joke", "idle/evening/coding", 1.0, now) == 0.0)


@section
def fix_this_is_not_the_background_song():
    print("\n=== FIX: 'I hate this' about a bug is not a verdict on the song playing behind it ===")
    from tst_taste import kb_with
    from src.daemon.taste import Taste, TasteConfig, BIND_RECENT_S
    tk = Taste(kb_with(), user="Khalid", cfg=TasteConfig(), clock=lambda: 1000.0)

    def say(line, started_ago):
        tk.kb.tastes.clear()
        playing = {"item": "Midnight City by M83", "kind": "music", "at": 1000.0 - started_ago}
        return tk.observe(line, "console", 1000.0, playing=playing)
    check("a track that has only just started: a bare 'I hate this' is about it", say("I hate this.", 10) == ["The user hates Midnight City by M83."])
    check("...five minutes in, a bare 'I hate this.' is not (could be anything)", say("I hate this.", 300) == [])
    check("...nor 'this is so annoying'", say("this is so annoying", 300) == [])
    check("'this song' always means the song, however long it has been on", say("I hate this song", 300) == ["The user hates Midnight City by M83."])
    check("...and 'this track'", say("I love this track so much", 300) == ["The user loves Midnight City by M83."])
    check("a taste with no 'this' in it is unaffected", say("I love Interstellar", 300) == ["The user loves Interstellar."])
    check("the window is short and named", 0 < BIND_RECENT_S <= 60, str(BIND_RECENT_S))
    out = say("I hate this, it's so slow", 10)
    check("'it's' is never kept as something liked or hated", not any("it's" in s.lower() for s in out), str(out))
    check("...whatever the verb", all("it's" not in s.lower() for s in say("I can't stand this, it's so slow", 300)))

    # the runtime records when the track started, which is what the rule above reads
    import tst_effects as te
    from src.senses.base import Modality, SensorEvent
    rt = te.rig()
    rt.t[0] = 5000.0
    rt.submit(SensorEvent(source="now_playing", modality=Modality.TEXT, text="now playing", salience=0.1, ts=5000.0,
                          persistable=False, meta=dict(event="started", player="Spotify", kind="music", title="Flowers",
                                                       artist="Miley Cyrus", item="Flowers by Miley Cyrus")))
    rt._tick()
    check("the runtime keeps when the track started", rt._playing and rt._playing.get("at") == 5000.0, str(rt._playing))


if __name__ == "__main__":
    for fn in SECTIONS:
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
