"""
tst_nowplaying.py -- what is playing, and what that says about what you like.

A player puts what it is playing in its window title. senses/nowplaying.py reads
those titles (only of known players), follows each play from start to finish and
reports what just ended and for how long; taste.Taste.watch() turns "heard to the
end", "replayed" and "skipped after ten seconds" into a profile, in small steps
that never override what the person said.

  N1  Titles: what is a song or a video, and what is not
  N2  The sensor: starts, changes, stops, skips, pauses; privacy; two players at once
  N3  The profile: listens build up, skips count against, the person's word stands
  N4  On the runtime: a play reaches the profile, can be undone, writes nothing to the journal
  N5  Wired in: the periphery has it, the config names it, the manual teaches it

Run:  PYTHONIOENCODING=utf-8 python tests/tst_nowplaying.py
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


def t1_titles():
    print("\n=== N1. Titles ===")
    from src.senses.nowplaying import parse

    def got(title, proc):
        p = parse(title, proc)
        return None if p is None else (p.player, p.kind, p.title, p.artist)

    for title, proc, want in [
        ("The Weeknd - Blinding Lights", "Spotify.exe", ("Spotify", "music", "Blinding Lights", "The Weeknd")),
        ("Daft Punk - Around the World - Remastered", "Spotify.exe",
         ("Spotify", "music", "Around the World - Remastered", "Daft Punk")),
        ("Spotify Premium", "Spotify.exe", None),
        ("Spotify", "Spotify.exe", None),
        ("Advertisement", "Spotify.exe", None),
        ("Lofi Girl - lofi hip hop radio - YouTube - Google Chrome", "chrome.exe",
         ("YouTube", "video", "Lofi Girl - lofi hip hop radio", "")),
        ("(3) Interstellar Docking Scene - YouTube - Google Chrome", "chrome.exe",
         ("YouTube", "video", "Interstellar Docking Scene", "")),
        ("Starboy - YouTube Music - Google Chrome", "chrome.exe", ("YouTube Music", "music", "Starboy", "")),
        ("YouTube - Google Chrome", "chrome.exe", None),
        ("Inbox (12) - someone@example.com - Gmail - Google Chrome", "chrome.exe", None),
        ("Dark | Netflix - Google Chrome", "chrome.exe", ("Netflix", "show", "Dark", "")),
        ("Watch Dark | Netflix - Mozilla Firefox", "firefox.exe", ("Netflix", "show", "Dark", "")),
        ("Some Video - YouTube - Personal - Microsoft​ Edge", "msedge.exe",
         ("YouTube", "video", "Some Video", "")),
        ("Some Video - YouTube and 2 more pages - Personal - Microsoft Edge", "msedge.exe",
         ("YouTube", "video", "Some Video", "")),
        ("Stream Midnight City by M83 | Listen online for free on SoundCloud - Brave", "brave.exe",
         ("SoundCloud", "music", "Midnight City", "M83")),
        ("Oppenheimer.2023.1080p.mkv - VLC media player", "vlc.exe", ("VLC", "video", "Oppenheimer.2023.1080p", "")),
        ("track01.flac - VLC media player", "vlc.exe", ("VLC", "music", "track01", "")),
        ("VLC media player", "vlc.exe", None),
        ("Some Film - mpv", "mpv.exe", ("mpv", "video", "Some Film", "")),
        ("Radiohead - Reckoner [foobar2000]", "foobar2000.exe", ("foobar2000", "music", "Reckoner", "Radiohead")),
        ("Stremio", "stremio.exe", None),
        ("brain.py - A.S.H - Visual Studio Code", "Code.exe", None),
        ("The Weeknd - Blinding Lights", "notepad.exe", None),         # a title alone is not a player
        ("The Weeknd - Blinding Lights", "", None),
        ("", "Spotify.exe", None),
    ]:
        check(f"{proc} {title!r}", got(title, proc) == want, str(got(title, proc)))


class World:
    """A desktop the sensor can look at, and a clock it can be told the time by."""

    def __init__(self):
        self.t = NOW
        self.windows = []

    def sensor(self, **kw):
        from src.senses.nowplaying import NowPlayingSensor
        kw.setdefault("interval", 5.0)
        return NowPlayingSensor(windows=lambda: list(self.windows), clock=lambda: self.t, **kw)

    def poll(self, s, secs=5.0):
        self.t += secs
        return s.read()


def t2_sensor():
    print("\n=== N2. The sensor ===")
    w = World()
    s = w.sensor()
    song = lambda a, t: (f"{a} - {t}", "Spotify.exe", False, 100)

    check("nothing open, nothing said", w.poll(s) == [])
    w.windows = [("brain.py - A.S.H - Visual Studio Code", "Code.exe", True, 1), ("Notes", "notepad.exe", False, 2)]
    check("windows that are not players are not looked at", w.poll(s) == [] and not s.slots)

    w.windows = [song("The Weeknd", "Blinding Lights")]
    check("a song seen once is not yet a song (titles flicker)", w.poll(s) == [])
    ev = w.poll(s)
    check("seen twice, it has started", len(ev) == 1 and ev[0].meta["event"] == "started", str(ev))
    m = ev[0].meta
    check("...and says what, where and by whom", (m["item"], m["player"], m["kind"], m["artist"]) ==
          ("Blinding Lights by The Weeknd", "Spotify", "music", "The Weeknd"), str(m))
    check("nothing finished yet", "finished" not in m)
    check("it is not written to the journal", ev[0].persistable is False)
    check("a quiet event, not one to speak up about", ev[0].salience <= 0.15 and ev[0].urgency == 0)
    for _ in range(30):
        w.poll(s)
    check("playing on, nothing more is said", w.poll(s) == [])
    check("the sensor knows what is playing", [p.item for p in s.playing_now()] == ["Blinding Lights by The Weeknd"])

    w.windows = [song("Daft Punk", "One More Time")]
    w.poll(s)
    ev = w.poll(s)
    m = ev[0].meta if ev else {}
    check("a new song: changed, with what just ended", m.get("event") == "changed" and
          m.get("finished") == "Blinding Lights by The Weeknd" and m.get("finished_how") == "changed", str(m))
    check("...played about two and a half minutes", 140 <= m.get("finished_s", 0) <= 170, str(m))
    check("...which is not a skip", m.get("finished_skip") is False)

    for _ in range(3):                                   # 15s in, and the next one
        w.poll(s)
    w.windows = [song("Daft Punk", "Harder Better")]
    w.poll(s)
    ev = w.poll(s)
    m = ev[0].meta if ev else {}
    check("replaced after ~20 seconds is a skip", m.get("finished_skip") is True and m.get("finished_s", 99) < 30,
          str(m))

    w.windows = [("Spotify Premium", "Spotify.exe", False, 100)]
    check("paused for a moment, nothing yet", w.poll(s, 5) == [])
    ev = w.poll(s, 10)
    m = ev[0].meta if ev else {}
    check("paused for good, it has stopped", m.get("event") == "stopped" and m.get("finished_how") == "stopped",
          str(m))
    check("...and a stop is never a skip", m.get("finished_skip") is False)
    check("...and nothing is playing", s.playing_now() == [])

    # A pause and a resume is one listen.
    w2 = World()
    s2 = w2.sensor()
    w2.windows = [song("A", "B")]
    w2.poll(s2)
    w2.poll(s2)
    for _ in range(4):
        w2.poll(s2)                                      # ~25s in
    w2.windows = [("Spotify Premium", "Spotify.exe", False, 100)]
    ev = []
    for _ in range(4):
        ev += w2.poll(s2)
    check("paused at ~25s: a stop, not a skip", ev and ev[0].meta["finished_skip"] is False, str([e.meta for e in ev]))
    w2.windows = [song("A", "B")]
    w2.poll(s2)
    w2.poll(s2)
    for _ in range(20):
        w2.poll(s2)
    w2.windows = []
    ev = []
    for _ in range(4):
        ev += w2.poll(s2)
    check("resumed and played on: the two parts add up (a listen, not two fragments)",
          ev and ev[0].meta["finished_s"] >= 90, str([e.meta for e in ev]))

    # A flicker is not a change.
    w3 = World()
    s3 = w3.sensor()
    w3.windows = [song("A", "B")]
    w3.poll(s3)
    w3.poll(s3)
    w3.windows = [song("X", "Y")]
    r = w3.poll(s3)
    w3.windows = [song("A", "B")]
    r += w3.poll(s3)
    check("a title that flickers for one poll changes nothing", r == [] and s3.playing_now()[0].title == "B", str(r))

    # Privacy: a blocked or private window is not reported, nor is a player's title off the list
    w4 = World()
    from src.senses.screen import ActiveWindowSensor
    ws = ActiveWindowSensor(blocklist=["secret song"], use_default_blocklist=True)
    s4 = w4.sensor(window_sensor=ws)
    w4.windows = [song("Someone", "Secret Song")]
    w4.poll(s4)
    check("a window on the blocked list is not reported", w4.poll(s4) == [] and not s4.playing_now())
    from src.senses.nowplaying import NowPlayingSensor
    check("the sensor says it only reads titles of players", "player" in NowPlayingSensor.capability.description)
    check("...and is marked privacy-sensitive", NowPlayingSensor.capability.privacy_sensitive is True)

    # What the tab sensor would never record is not a "like" either.
    w7 = World()
    s7 = w7.sensor()
    w7.windows = [("Symptoms of anxiety - YouTube - Google Chrome", "chrome.exe", True, 300)]
    w7.poll(s7)
    check("a video whose title is about health is not reported as playing", w7.poll(s7) == [] and not s7.playing_now())
    w7.windows = [("Your recovery key - YouTube - Google Chrome", "chrome.exe", True, 300)]
    w7.poll(s7)
    check("...nor one about a password or recovery key", w7.poll(s7) == [])
    w7.windows = [("Interstellar Docking Scene - YouTube - Google Chrome", "chrome.exe", True, 300)]
    w7.poll(s7)
    check("an ordinary title still is", len(w7.poll(s7)) == 1)

    # Two players at once are two plays.
    w5 = World()
    s5 = w5.sensor()
    w5.windows = [song("A", "B"), ("Dark | Netflix - Google Chrome", "chrome.exe", True, 200)]
    w5.poll(s5)
    ev = w5.poll(s5)
    check("a song and a show at the same time are both seen",
          sorted(e.meta["player"] for e in ev) == ["Netflix", "Spotify"], str([e.meta for e in ev]))

    # Where the platform can say what is making sound, a silent tab is not "playing".
    w6 = World()
    s6 = w6.sensor(audible=lambda: [999])
    w6.windows = [("Some Video - YouTube - Google Chrome", "chrome.exe", True, 200)]
    w6.poll(s6)
    check("an open tab that makes no sound is not playing", w6.poll(s6) == [])
    s7 = w6.sensor(audible=lambda: [200])
    w6.poll(s7)
    check("...one that does is", len(w6.poll(s7)) == 1)


def watch(tt, **m):
    return tt.watch(dict(m), NOW)


def music(secs, how="changed", skip=False, title="Blinding Lights", artist="The Weeknd"):
    return dict(finished=f"{title} by {artist}", finished_title=title, finished_artist=artist,
                finished_kind="music", finished_s=secs, finished_how=how, finished_skip=skip)


def t3_profile():
    print("\n=== N3. The profile ===")
    import tst_taste as tt_

    tt = tt_.taste()
    item = lambda k: tt.kb.tastes.get("khalid", {}).get("items", {}).get(k)

    check("a play heard to the end counts a little, and says nothing yet", watch(tt, **music(200)) == [])
    check("...it is kept, as habit", item("blinding lights by the weeknd")["by"] == "habit"
          and abs(item("blinding lights by the weeknd")["score"] - 0.1) < 1e-6, str(item("blinding lights by the weeknd")))
    check("...and the artist gets half", abs(item("the weeknd")["score"] - 0.05) < 1e-6, str(item("the weeknd")))
    watch(tt, **music(200))
    said = watch(tt, **music(200))
    check("three listens make it something they like", said == ["The user likes Blinding Lights by The Weeknd."],
          str(said))
    check("...and it reads as one in the profile", "Blinding Lights by The Weeknd" in tt.describe("khalid"),
          tt.describe("khalid"))
    for _ in range(12):
        watch(tt, **music(200))
    check("however often it is played, behaviour alone never says 'love'",
          item("blinding lights by the weeknd")["score"] <= 0.6, str(item("blinding lights by the weeknd")))
    check("...and the count is kept", item("blinding lights by the weeknd")["n"] == 15)

    check("a song played for 40 seconds and let go on is neither", watch(tt, **music(40, title="Meh")) == []
          and item("meh by the weeknd") is None)
    watch(tt, **music(12, skip=True, title="Nope", artist="Someone"))
    check("one skip is noted but not yet a dislike", abs(item("nope by someone")["score"] + 0.2) < 1e-6)
    said = watch(tt, **music(9, skip=True, title="Nope", artist="Someone"))
    check("two skips make it something they don't like", said == ["The user dislikes Nope by Someone."], str(said))
    check("...and the artist is not blamed", item("someone") is None)
    check("a stop is not a skip (nothing finished_skip, nothing counted)",
          watch(tt, **music(8, how="stopped", skip=False, title="Paused", artist="Q")) == []
          and item("paused by q") is None)

    # Their own word stands.
    tt.observe("I hate Hotline Bling", "console", NOW)
    before = dict(item("hotline bling"))
    for _ in range(5):
        watch(tt, **music(200, title="Hotline Bling", artist="Drake"))
    check("a person who said they hate it is not argued with by what played",
          item("hotline bling by drake") is not None and item("hotline bling")["score"] == before["score"],
          str(item("hotline bling")))
    tt.observe("I love Interstellar", "console", NOW)
    s0 = item("interstellar")["score"]
    watch(tt, finished="Interstellar", finished_title="Interstellar", finished_artist="", finished_kind="show",
          finished_s=60, finished_how="changed", finished_skip=False)
    check("...nor is something they love, dropped after a minute", item("interstellar")["score"] == s0)

    # Films, shows and videos
    tt2 = tt_.taste()
    it2 = lambda k: tt2.kb.tastes.get("khalid", {}).get("items", {}).get(k)
    for _ in range(3):
        watch(tt2, finished="Dark", finished_title="Dark", finished_artist="", finished_kind="show",
              finished_s=2400, finished_how="stopped", finished_skip=False)
    check("a show watched for long stretches is one they like", it2("dark") and it2("dark")["score"] >= 0.3
          and it2("dark")["kind"] == "show", str(it2("dark")))
    watch(tt2, finished="Pilot", finished_title="Pilot", finished_artist="", finished_kind="show",
          finished_s=50, finished_how="changed", finished_skip=False)
    check("a show dropped after under two minutes counts against it", it2("pilot")["score"] < 0, str(it2("pilot")))
    watch(tt2, finished="Clip", finished_title="Clip", finished_artist="", finished_kind="video",
          finished_s=20, finished_how="changed", finished_skip=False)
    check("hopping between videos says nothing", it2("clip") is None)
    check("a bare play with nothing finished does nothing", watch(tt2, event="started", item="X") == [])

    # Off means off; a guest cannot make it up; undo puts it back.
    tt3 = tt_.taste(from_playing=False)
    check("from_playing off learns nothing from plays", watch(tt3, **music(200)) == [] and not tt3.kb.tastes)
    tt4 = tt_.taste()
    watch(tt4, **music(200))
    watch(tt4, **music(200))
    snap = dict(tt4.kb.tastes["khalid"]["items"]["blinding lights by the weeknd"])
    watch(tt4, **music(200))
    check("the third listen leaves an undo behind", tt4.last_undo is not None)
    tt4.last_undo()
    now = tt4.kb.tastes["khalid"]["items"]["blinding lights by the weeknd"]
    check("...which puts the song back to two listens", now["n"] == snap["n"] and now["score"] == snap["score"],
          str(now))
    tt5 = tt_.taste(enabled=False)
    check("taste off, plays teach nothing", watch(tt5, **music(200)) == [])


def t4_runtime():
    print("\n=== N4. On the runtime ===")
    from src.daemon.taste import Taste, TasteConfig
    from src.senses.base import Modality, SensorEvent
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, taste=True, knowledge_path=os.path.join(d, "knowledge.json"),
                wonder_path=os.path.join(d, "wonders.json"))
    rt.brain.user = "Khalid"
    rt.taste = Taste(rt.wonder.kb, user="Khalid", cfg=TasteConfig(), clock=lambda: rt.t[0])
    items = lambda: rt.wonder.kb.tastes.get("khalid", {}).get("items", {})

    def play(secs, title="Blinding Lights", artist="The Weeknd", skip=False):
        rt.t[0] += 3
        rt.submit(SensorEvent(source="now_playing", modality=Modality.TEXT, text="now playing x", salience=0.1,
                              persistable=False, meta={"event": "changed", **music(secs, skip=skip, title=title,
                                                                                    artist=artist)}))
        rt._tick()

    for _ in range(3):
        play(200)
    check("three listens reach the profile", "blinding lights by the weeknd" in items()
          and items()["blinding lights by the weeknd"]["score"] >= 0.3, str(items()))
    check("...and leave an undo with a label that reads",
          any("Blinding Lights" in lbl for lbl in [s[1] for s in rt.undo._steps]), str(rt.undo._steps))
    n = len(rt.undo._steps)
    play(200, title="Starboy")
    check("a play that changes nothing worth saying leaves no undo step", len(rt.undo._steps) == n)

    rt.say("what do I like?")
    check("asked, it answers from what was played too", "Blinding Lights" in rt.out[-1], rt.out[-1])

    undone = rt.say("undo that")
    check("'undo that' takes the song back out of the profile", "blinding lights by the weeknd" not in items()
          or items()["blinding lights by the weeknd"]["score"] < 0.3, str(items()))

    rt.taste = None
    play(200)                                       # no profile on this runtime: must not raise
    check("with no taste profile, a play is ignored quietly", True)


def t5_wired():
    print("\n=== N5. Wired in ===")
    from src.daemon.service import DEFAULT_CONFIG
    from src.daemon import manual
    from src.senses import build_periphery
    from src.senses.base import PeripheryRegistry
    cfg = DEFAULT_CONFIG
    check("the config names the sensor", "now_playing" in cfg["sensors"], str(cfg["sensors"].keys()))
    check("...and the taste switches", all(k in cfg["taste"] for k in ("from_playing", "listen_s", "watch_s")))
    reg = build_periphery({"sensors": {"now_playing": {"enabled": True}}}, registry=PeripheryRegistry())
    check("the periphery builds it", "now_playing" in reg.sensors, str(list(reg.sensors)))
    off = build_periphery({"sensors": {"now_playing": {"enabled": False}}}, registry=PeripheryRegistry())
    check("...and it can be switched off", off.sensors["now_playing"].state.value in ("disabled", "unavailable"),
          off.sensors["now_playing"].state.value)
    ids = [e[0] for e in manual.ENTRIES]
    check("the manual in core memory has an entry for it", "playing" in ids, str(ids))
    body = [e for e in manual.ENTRIES if e[0] == "playing"][0][2]
    check("...which says where it comes from and what it cannot do", "title" in body.lower() and "pause" in body.lower(),
          body)
    check("...and fits the manual's size limit", len(body) < 760, str(len(body)))
    taste_body = [e for e in manual.ENTRIES if e[0] == "taste"][0][2]
    check("the taste entry no longer says it ignores what you play", "does not yet learn tastes from the screen or "
          "from what you play" not in taste_body, taste_body)


if __name__ == "__main__":
    t1_titles()
    t2_sensor()
    t3_profile()
    t4_runtime()
    t5_wired()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
