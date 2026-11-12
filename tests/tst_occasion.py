"""
tst_occasion.py -- "Khalid listens to this while he's coding".

Two facts and a link: a song (senses/nowplaying.py), a stretch of work
(daemon/occasion.py), and the link between them, which is only worth keeping once
it has happened more than once. The link goes into the same profile as what the
person says ("I listen to lo-fi while I'm coding"), under the same words.

  O1  Programs: which ones say what you are doing, and which say nothing
  O2  The tracker: what covered a stretch, idle time, blocked windows
  O3  The link: one sighting is a coincidence, three is a habit; what is said stands
  O4  Asked and used: "what do I listen to while coding?", suggestions, entity memory
  O5  On the runtime: window changes and plays become a link; the manual teaches it

Run:  PYTHONIOENCODING=utf-8 python tests/tst_occasion.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
NOW = 1_800_000_000.0


def t1_programs():
    print("\n=== O1. Programs ===")
    from src.daemon.occasion import activity_of
    for proc, want in [("Code.exe", "coding"), ("pycharm64.exe", "coding"), ("cursor.exe", "coding"),
                       ("WINWORD.EXE", "writing"), ("EXCEL.EXE", "working"), ("Teams.exe", "working"),
                       ("SumatraPDF.exe", "reading"), ("Figma.exe", "designing"), ("steam.exe", "gaming"),
                       ("chrome.exe", ""), ("msedge.exe", ""), ("WindowsTerminal.exe", ""), ("cmd.exe", ""),
                       ("Spotify.exe", ""), ("Discord.exe", ""), ("", ""), ("explorer.exe", "")]:
        check(f"{proc or '(none)'} -> {want or 'nothing'}", activity_of(proc) == want, activity_of(proc))
    from src.daemon.taste import _ACTIVITIES
    from src.daemon.occasion import ACTIVITY_OF
    check("every activity it can name is one the profile accepts", set(ACTIVITY_OF.values()) <= _ACTIVITIES,
          str(set(ACTIVITY_OF.values()) - _ACTIVITIES))


def t2_tracker():
    print("\n=== O2. The tracker ===")
    from src.daemon.occasion import Occasion
    o = Occasion(clock=lambda: NOW)
    o.window("Code.exe", NOW)
    check("in an editor: coding", o.now() == "coding")
    check("a stretch wholly inside it is coding", o.during(NOW + 10, NOW + 200, NOW + 300) == "coding")
    o.window("chrome.exe", NOW + 600)
    check("a browser says nothing, so the editor stretch is closed", o.now() == "")
    check("...a stretch half in it, half not: still coding (at least half)",
          o.during(NOW + 500, NOW + 700, NOW + 800) == "coding")
    check("...mostly after it: nothing", o.during(NOW + 590, NOW + 900, NOW + 900) == "")
    check("...with no stretch at all: nothing", o.during(NOW + 5, NOW + 5, NOW + 6) == "")

    o2 = Occasion(clock=lambda: NOW)
    o2.window("Code.exe", NOW)
    o2.tick(NOW + 1000, idle_s=700)                       # went quiet 300s in
    check("away from the keyboard ends it where they went quiet", o2.now() == "")
    check("...so a play during the idle time is not 'while coding'", o2.during(NOW + 400, NOW + 900, NOW + 1000) == "")
    check("...but the first part was", o2.during(NOW + 10, NOW + 290, NOW + 1000) == "coding")

    o3 = Occasion(clock=lambda: NOW)
    o3.window("Code.exe", NOW, blocked=True)
    check("a blocked window is nothing, not coding", o3.now() == "")

    o4 = Occasion(clock=lambda: NOW, keep_s=100)
    o4.window("Code.exe", NOW)
    o4.window("chrome.exe", NOW + 10)
    o4.tick(NOW + 500, idle_s=0)
    check("old segments are forgotten", o4.status()["segments"] == 0)

    o5 = Occasion(clock=lambda: NOW)
    o5.window("Code.exe", NOW)
    o5.window("EXCEL.EXE", NOW + 100)
    check("the one that covered most of it wins", o5.during(NOW, NOW + 150, NOW + 150) == "coding"
          and o5.during(NOW + 90, NOW + 400, NOW + 400) == "working")


def music(item_start, secs=200, title="Lofi Beats", artist="Lofi Girl"):
    return dict(finished=f"{title} by {artist}", finished_title=title, finished_artist=artist,
                finished_kind="music", finished_s=secs, finished_how="changed", finished_skip=False,
                finished_start=item_start, finished_end=item_start + secs)


def t3_link():
    print("\n=== O3. The link ===")
    import tst_taste as tt_
    from src.daemon.taste import occasion_of
    tt = tt_.taste()
    key = "lofi beats by lofi girl"
    item = lambda: tt.kb.tastes["khalid"]["items"][key]

    tt.watch(music(NOW), NOW, occasion="coding")
    check("once while coding is a coincidence, not a habit", occasion_of(item()) == "", str(item()))
    check("...but it was seen", item()["seen_when"] == {"coding": 1})
    tt.watch(music(NOW), NOW, occasion="coding")
    check("twice is still not", occasion_of(item()) == "")
    said = tt.watch(music(NOW), NOW, occasion="coding")
    check("three times, every time: a habit", occasion_of(item()) == "coding", str(item()))
    check("...and it says so once, as a sentence the undo step can carry",
          said == ["The user likes Lofi Beats by Lofi Girl, while coding."], str(said))
    again = tt.watch(music(NOW), NOW, occasion="coding")
    check("...and not again for the fourth", again == [], str(again))

    tt.watch(music(NOW), NOW, occasion="")
    for _ in range(6):
        tt.watch(music(NOW), NOW, occasion="")
    check("played mostly with nothing in particular going on: no longer 'while coding'",
          occasion_of(item()) == "", str(item()))

    tt.watch(music(NOW), NOW, occasion="napping")
    check("an occasion it does not keep is not kept", "napping" not in item().get("seen_when", {}))

    # A skip is not a sighting.
    tt2 = tt_.taste()
    skip = dict(music(NOW, 10), finished_skip=True)
    tt2.watch(skip, NOW, occasion="coding")
    tt2.watch(skip, NOW, occasion="coding")
    check("skipped while coding says nothing about 'while coding'",
          "seen_when" not in tt2.kb.tastes["khalid"]["items"]["lofi beats by lofi girl"])

    # What was said stands over what was seen.
    tt3 = tt_.taste()
    tt3.observe("I listen to lo-fi when I'm studying", "console", NOW)
    got = tt3.kb.tastes["khalid"]["items"]["lo fi"]
    check("said while studying: that is the occasion at once", occasion_of(got) == "studying", str(got))
    for _ in range(5):
        tt3.watch(music(NOW, title="Lo-fi", artist=""), NOW, occasion="coding")
    check("...and seen while coding does not overrule it",
          occasion_of(tt3.kb.tastes["khalid"]["items"]["lo fi"]) == "studying")


def t4_asked_and_used():
    print("\n=== O4. Asked and used ===")
    import tst_taste as tt_
    tt = tt_.taste()
    for _ in range(3):
        tt.watch(music(NOW), NOW, occasion="coding")
    for _ in range(3):
        tt.watch(music(NOW, title="Gym Rap", artist="Eminem"), NOW, occasion="exercising")
    tt.observe("I love Interstellar", "console", NOW)

    a = tt.answer("what do I listen to while coding?")
    check("asked what you listen to while coding, it answers from what it saw",
          a == "While coding, you listen to Lofi Beats by Lofi Girl.", str(a))
    for q in ("what do I listen to when I'm coding", "What do I usually listen to while I code?",
              "what do I listen to when I am coding?"):
        check("...however it is put: " + q, tt.answer(q) is not None and "Lofi Beats" in tt.answer(q), str(tt.answer(q)))
    check("another activity gets its own answer", "Gym Rap" in (tt.answer("what do I listen to while exercising?") or ""))
    b = tt.answer("what do I listen to while cooking?")
    check("with nothing known it says so and how to teach it", b and "haven't noticed" in b and "lo-fi" in b, str(b))
    check("a mood is accepted too", tt.answer("what do I listen to when I'm sad?") is not None)
    check("something it does not keep is left to the model", tt.answer("what do I listen to when it rains?") is None)
    check("a question about choosing is still left to the model", tt.answer("what should I listen to while coding?") is None)
    check("what you like is still answered as before", (tt.answer("what do I like?") or "").startswith("You love Interstellar"))

    ctx = tt.context("what should I listen to while I'm coding?", NOW)
    check("choosing music while coding: what goes with coding comes first and is marked",
          "Lofi Beats by Lofi Girl (while coding)" in ctx and ctx.index("Lofi Beats") < ctx.index("Gym Rap"), ctx)

    from src.daemon.taste import mirror
    kb = tt.kb
    calls = []
    kb.entities = object()
    kb._mirror = lambda kind, name, fields: calls.append((kind, name, fields))
    mirror(kb, ["khalid"], "Khalid", 0.3)
    check("entity memory carries it: 'Lofi Beats by Lofi Girl (while coding)'",
          calls and "Lofi Beats by Lofi Girl (while coding)" in calls[-1][2]["likes"], str(calls))
    check("...and the profile shows it", "Lofi Beats by Lofi Girl" in tt.report())


def t5_runtime():
    print("\n=== O5. On the runtime ===")
    from src.daemon.taste import Taste, TasteConfig, occasion_of
    from src.senses.base import Modality, SensorEvent
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, taste=True, knowledge_path=os.path.join(d, "knowledge.json"),
                wonder_path=os.path.join(d, "wonders.json"))
    rt.brain.user = "Khalid"
    rt.taste = Taste(rt.wonder.kb, user="Khalid", cfg=TasteConfig(), clock=lambda: rt.t[0])
    items = lambda: rt.wonder.kb.tastes.get("khalid", {}).get("items", {})
    check("the runtime has the tracker", hasattr(rt, "occasion"))

    def window(proc):
        rt.t[0] += 1
        rt.submit(SensorEvent(source="active_window", modality=Modality.TEXT, text=f"switched to {proc}", salience=0.2,
                              meta={"process": proc, "blocked": False}))
        rt._tick()

    def play(secs=200):
        end = rt.t[0]
        rt.t[0] += 2
        rt.submit(SensorEvent(source="now_playing", modality=Modality.TEXT, text="now playing", salience=0.1,
                              persistable=False, meta={"event": "changed", **music(end - secs, secs)}))
        rt._tick()

    window("Code.exe")
    rt.t[0] += 300                                        # a long time in the editor, the song playing throughout
    for _ in range(3):
        play()
        rt.t[0] += 300
    e = items().get("lofi beats by lofi girl")
    check("three plays while the editor was in front: it is 'while coding'",
          e is not None and occasion_of(e) == "coding", str(e))
    check("...and the undo step says so", any("while coding" in s[1] for s in rt.undo._steps), str(rt.undo._steps))
    rt.say("what do I listen to while coding?")
    check("asked on the runtime, it answers by rule", "Lofi Beats by Lofi Girl" in rt.out[-1], rt.out[-1])

    window("chrome.exe")
    rt.t[0] += 600
    play(200)
    window("Code.exe")
    for _ in range(3):
        play(200)
    check("with nothing recognised in front, a play adds no occasion",
          items()["lofi beats by lofi girl"].get("seen_when", {}).get("coding", 0) >= 3)


def t6_manual():
    print("\n=== O6. The manual ===")
    from src.daemon import manual
    ids = {e[0]: e[2] for e in manual.ENTRIES}
    check("there is an entry for what goes with what, with the phrase that works",
          "occasion" in ids and "what do I listen to while coding" in ids["occasion"], str(list(ids)))
    check("...and it says what it cannot tell", "browser or terminal says nothing" in ids["occasion"])
    check("every entry still fits", all(len(t) < 760 for t in ids.values()), str(max(len(t) for t in ids.values())))
    check("the phrase it teaches is understood", __import__("tst_taste").taste().wants(
        "what do I listen to while coding?") is not None)


if __name__ == "__main__":
    t1_programs()
    t2_tracker()
    t3_link()
    t4_asked_and_used()
    t5_runtime()
    t6_manual()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
