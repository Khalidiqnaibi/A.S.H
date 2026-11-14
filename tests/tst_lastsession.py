"""
tst_lastsession.py -- what the last real session, and a review of the code behind it, found.

A session on Nov 14 went: "hit me" (to ASH's own trivia offer, twelve minutes later) answered by the model asking what it
meant; "thats easy" after a trivia answer answered by the model saying it had not done anything; "thank you ash .. tell me a
joke" sent to a 1.5B model that took 78 seconds to mangle one; the trivia read in a different voice from ASH's; then a full
disk, a screen grab that failed, a Windows restart, and at the next start two state files found empty. Separately it took
100% of CPU and RAM while a game ran. A review of the code found the rest.

  L1  Games: "hit me" and its cousins, "that's easy", "another one", "too hard", and the voice
  L2  Jokes and yes, said the way people say them
  L3  Saves that survive the power going: atomic writes, backups, tolerant loads, a full disk said once
  L4  Sensors that come back; a microphone that does not send audio away; a vision model that is not loaded for nothing
  L5  The loop: a cancel that cannot leave the model blocked, one bad event, learning saved on stop
  L6  What a model says that neither JARVIS nor BMO would: taken out, and asked not to be said
  L7  The screen read-out, the journal flood, the console window and the console-close handler

Run:  PYTHONIOENCODING=utf-8 python tests/tst_lastsession.py
"""

import errno
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
T0 = 1_800_000_000.0
from tools import screen_tool as _screen_tool  # noqa: E402,F401  (imported once, before any test swaps modules in and out)


def l1_games():
    print("\n=== L1. Games ===")
    import random
    from src.daemon.play import Play
    from src.daemon.service import DEFAULT_CONFIG

    check("the games speak in ASH's own voice by default", DEFAULT_CONFIG["play"]["voice"] == "", str(DEFAULT_CONFIG["play"]))

    p = Play(clock=lambda: T0, seed=3)
    p.offer("trivia", T0)
    r = p.hear("hit me", T0 + 12 * 60)
    check("'hit me' twelve minutes after the trivia offer is a yes (it went to the model)",
          r is not None and r["say"].startswith("Trivia:"), str(r))
    check("...and the game is on", p.playing)

    for phrase in ("bring it on", "let's go", "go for it", "Hit me!", "hey ash, hit me", "I'm ready", "fire away", "ask me"):
        q = Play(clock=lambda: T0, seed=1)
        q.offer("trivia", T0)
        check(f"'{phrase}' answers the offer", (q.hear(phrase, T0 + 300) or {}).get("say", "").startswith("Trivia:"))

    q = Play(clock=lambda: T0, seed=1)
    q.offer("trivia", T0)
    check("a plain 'yes' ten minutes later is NOT taken (it could be about anything)", q.hear("yes", T0 + 600) is None)
    check("...and the offer is still open for an unmistakable one", q.offered is not None)
    q.offer("trivia", T0)
    check("nor is it taken after 40 minutes", q.hear("hit me", T0 + 2400) is None)
    q.offer("trivia", T0)
    check("'hit the lights' is not a yes", q.hear("hit the lights", T0 + 10) is None)
    q2 = Play(clock=lambda: T0, seed=1)
    check("with nothing offered, 'hit me' starts nothing", q2.hear("hit me", T0) is None)
    q2.offer("trivia", T0)
    check("a yes within two minutes still works", (q2.hear("yeah sure", T0 + 30) or {}).get("say", "").startswith("Trivia:"))

    # After a game
    g = Play(clock=lambda: T0, seed=5)
    g.start("trivia", T0)
    ans = g.game.answers[0]
    r = g.hear(ans, T0 + 5)
    check("a right answer ends with 'Another one?'", r["say"].startswith("Correct!") and r["say"].endswith("Another one?"), str(r))
    check("...and it is an open offer for the same kind of game", g.offered and g.offered[0] == "again:trivia", str(g.offered))
    r2 = g.hear("yes", T0 + 8)
    check("'yes' gives another trivia question, not the day's one again", r2["say"].startswith("Trivia:"), str(r2))

    g = Play(clock=lambda: T0, seed=5)
    g.start("trivia", T0)
    g.hear(g.game.answers[0], T0 + 5)
    g.offered = None
    r = g.hear("thats easy", T0 + 20)
    check("'thats easy' after a trivia answer: a harder-sounding line and a new question (it went to the model, which "
          "said it had done nothing)", r and r["say"].startswith("Easy, was it?") and "Trivia:" in r["say"], str(r))
    for phrase in ("another one", "next", "one more", "again"):
        g = Play(clock=lambda: T0, seed=5)
        g.start("trivia", T0)
        g.hear(g.game.answers[0], T0 + 5)
        g.offered = None
        check(f"'{phrase}' after a game is another one", (g.hear(phrase, T0 + 20) or {}).get("say", "").startswith("Trivia:"))
    g = Play(clock=lambda: T0, seed=5)
    g.start("trivia", T0)
    g.hear("no idea", T0 + 5)
    g.offered = None
    g.last = ("trivia", T0)
    r = g.hear("that was hard", T0 + 10)
    check("'that was hard' gets sympathy and an offer", r and "tough" in r["say"] and g.offered[0] == "again:trivia", str(r))
    g = Play(clock=lambda: T0, seed=5)
    g.last = ("trivia", T0)
    check("five minutes after the game, 'thats easy' is not about the game", g.hear("thats easy", T0 + 400) is None)
    check("with no game before it, nor is 'another one'", Play(clock=lambda: T0).hear("another one", T0) is None)

    from src.daemon import manual


def l2_jokes_and_yes():
    print("\n=== L2. Jokes and yes ===")
    from src.daemon.fun import _JOKE
    from src.daemon import yesno
    for t in ("thank you ash .. tell me a joke", "thanks, tell me a joke", "ok so tell me a joke please", "tell me a joke",
              "cool, another one"):
        check(f"a joke, by rule: {t!r}", bool(_JOKE.match(t)))
    for t in ("thank you ash", "tell me a joke about cats", "the joke is on you"):
        check(f"not one: {t!r}", not _JOKE.match(t))
    for t in ("hit me", "bring it on", "let's go", "fire away", "I'm in", "hit me with it"):
        check(f"'{t}' is a yes", yesno.is_yes(t))
        check(f"...but not a firm one (it cannot send a mail)", not yesno.is_yes(t, strong=True))
    check("'yes' is still firm", yesno.is_yes("yes", strong=True))
    check("'hit the brakes' is not a yes", not yesno.is_yes("hit the brakes"))


def l3_saves():
    print("\n=== L3. Saves that survive ===")
    from src.memory import atomic
    d = tempfile.mkdtemp()
    p = os.path.join(d, "a.json")
    atomic.write_json(p, {"v": 1})
    atomic.write_json(p, {"v": 2})
    check("the second write is the file, the first is the backup", json.load(open(p))["v"] == 2
          and json.load(open(p + ".bak"))["v"] == 1)
    check("no temp file is left behind", not os.path.exists(p + ".tmp"))
    open(p, "w").close()                                  # what a power cut left: an empty file
    got = atomic.load_json(p, default="none")
    check("an empty file: the last good copy is read", got == {"v": 1}, str(got))
    check("...and the empty one is set aside, not left to fail again", os.path.exists(p + ".bad") and not os.path.exists(p))
    check("nothing at all: the default", atomic.load_json(os.path.join(d, "none.json"), default=5) == 5)
    with open(os.path.join(d, "t.json"), "w") as f:
        f.write("{not json")
    check("torn JSON, no backup: the default", atomic.load_json(os.path.join(d, "t.json"), default="d") == "d")

    class Boom:
        pass
    try:
        atomic.write_json(os.path.join(d, "b.json"), {"x": Boom()})
        ok = False
    except TypeError:
        ok = True
    check("an object that cannot be written never touches the file", ok and not os.path.exists(os.path.join(d, "b.json")))

    import logging
    got_logs = []

    class H(logging.Handler):
        def emit(self, rec):
            got_logs.append(rec.getMessage())
    lg = logging.getLogger("tst.atomic")
    lg.addHandler(H())
    e = OSError(errno.ENOSPC, "No space left on device")
    for _ in range(50):
        atomic.log_io_failure(lg, "x-file", e)
    check("a full disk is said once, not fifty times", len(got_logs) == 1 and "disk is full" in got_logs[0], str(got_logs))
    check("...and is recognised", atomic.is_disk_full(e) and not atomic.is_disk_full(ValueError()))

    # dwell and crashes: found empty after a hard stop
    from src.daemon.dwell import Dwell
    dp = os.path.join(d, "dwell.json")
    dw = Dwell(path=dp, clock=lambda: T0) if "path" in Dwell.__init__.__code__.co_varnames else None
    if dw is not None:
        dw.day = {"2026-11-14": {"code.exe": 120.0}}
        dw.save()
        dw.day = {"2026-11-14": {"code.exe": 300.0}}
        dw.save()
        open(dp, "w").close()
        again = Dwell(path=dp, clock=lambda: T0)
        check("dwell: an empty file at start brings back the last good copy instead of starting fresh",
              again.day.get("2026-11-14", {}).get("code.exe") == 120.0, str(again.day))
    from src.daemon.crashwatch import CrashWatch
    cp = os.path.join(d, "crashes.json")
    cw = CrashWatch(path=cp, clock=lambda: T0, reader=lambda: [], background=False)
    cw._save_runs(T0 + 100)
    open(cp + ".runs", "w").close()
    cw2 = CrashWatch(path=cp, clock=lambda: T0 + 500, reader=lambda: [], background=False)
    check("crash history: an empty runs file brings back the last one", len(cw2.runs) >= 2, str(cw2.runs))

    # entities and core rules: a damaged file must not stop the daemon starting
    from src.memory.entity.entity_store import EntityStore
    ep = os.path.join(d, "entities.json")
    open(ep, "w").close()
    es = EntityStore(ep)
    check("entity memory: an empty file is survived (it used to raise at start-up)", es.all() == [])
    open(ep, "w").write('{"a": ')
    check("...and so is a torn one", EntityStore(ep).all() == [])
    from src.memory.core.core_store import CoreStore
    cop = os.path.join(d, "core.json")
    open(cop, "w").close()
    check("core memory: the same", CoreStore(cop).rules == {})

    # episodes: a torn last line, then another append
    from src.memory.episodic.episode_store import EpisodeStore
    from src.memory.episodic.episode_model import Episode
    jp = os.path.join(d, "ep.jsonl")
    st = EpisodeStore(jp)
    e1 = Episode(episode_id="e1", timestamp=T0, summary="one", event_type="x", related_entities=[], importance=0.5)
    st.add(e1)
    with open(jp, "a", encoding="utf-8") as f:
        f.write('{"episode_id": "e2", "summ')             # the power went mid-line
    st2 = EpisodeStore(jp)
    check("episodes: a torn last line is skipped, the rest kept", "e1" in st2.episodes and "e2" not in st2.episodes)
    st2.add(Episode(episode_id="e3", timestamp=T0, summary="three", event_type="x", related_entities=[], importance=0.5))
    st3 = EpisodeStore(jp)
    check("...and the next append is not glued to the torn line", "e3" in st3.episodes and "e1" in st3.episodes,
          str(list(st3.episodes)))

    # the journal on a full disk keeps what it could not write, a little, and says so once
    from src.daemon.journal import Journal
    from src.senses.base import Modality, SensorEvent
    j = Journal(os.path.join(d, "journal"))
    real_open = open

    def full(path, mode="r", *a, **k):
        if "a" in mode and "journal" in str(path):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, mode, *a, **k)
    import builtins
    builtins.open = full
    try:
        for i in range(5):
            j.append(SensorEvent(source="console", modality=Modality.TEXT, text=f"line {i}"))
        j.flush()
    finally:
        builtins.open = real_open
    check("a full disk: the lines are kept for the next flush, not dropped", len(j._buffer) == 5, str(len(j._buffer)))
    j.flush()
    check("...and written when there is room", len(j._buffer) == 0 and any(r["text"] == "line 4" for r in j.read_day()))

    # activity: a failed save waits, it does not retry every tick
    from src.daemon.activity import ActivityMemory
    am = ActivityMemory(path=os.path.join(d, "no", "such dir", "x.json"), clock=lambda: T0)
    am._dirty = True
    builtins.open = full
    try:
        am.save(force=True)
    finally:
        builtins.open = real_open
    check("activity: a failed save is not retried at once", am._saved_at == T0)


def l4_sensors():
    print("\n=== L4. Sensors, microphone, vision ===")
    from src.senses.base import Capability, Sensor, SensorEvent, State

    class Flaky(Sensor):
        capability = Capability(name="flaky")
        default_interval = 2.0
        fail = True

        def read(self):
            if self.fail:
                raise OSError("device busy")
            return [SensorEvent(source="flaky", text="ok")]
    s = Flaky(enabled=True)
    for _ in range(3):
        s.poll()
    check("failing backs the interval off", s.interval > s.base_interval, str(s.interval))
    s.fail = False
    s.poll()
    check("one good read puts the interval back", s.interval == s.base_interval, str(s.interval))
    s.fail = True
    for _ in range(5):
        s.poll()
    check("five failures in a row: resting", s.state == State.ERROR)
    check("...not due while it rests", not s.due(s.errored_at + 10))
    check("...tried again after a couple of minutes", s.due(s.errored_at + s.RETRY_AFTER_S + 1) and s.state == State.ACTIVE)

    from src.senses.mic import MicrophoneSensor
    m = MicrophoneSensor(enabled=False)
    check("local recognition only by default", m.local_only is True)
    import numpy as np
    sent = []
    import types
    fake_sr = types.ModuleType("speech_recognition")

    class Rec:
        def recognize_google(self, data):
            sent.append(1)
            return "what the room said"
    fake_sr.Recognizer = Rec
    fake_sr.AudioData = lambda *a: None
    old = sys.modules.get("speech_recognition")
    sys.modules["speech_recognition"] = fake_sr
    fake_fw = types.ModuleType("faster_whisper")

    def broken(*a, **k):
        raise MemoryError("out of memory")
    fake_fw.WhisperModel = broken
    oldfw = sys.modules.get("faster_whisper")
    sys.modules["faster_whisper"] = fake_fw
    try:
        out = m.transcribe(np.zeros(1600, dtype=np.int16))
        check("when Whisper fails the audio does NOT go to Google (it used to)", out == "" and not sent, f"{out!r} {sent}")
        m2 = MicrophoneSensor(enabled=False, local_only=False)
        out2 = m2.transcribe(np.zeros(1600, dtype=np.int16))
        check("...unless privacy.require_local_asr is switched off", out2 == "what the room said" and sent)
    finally:
        for name, mod in (("speech_recognition", old), ("faster_whisper", oldfw)):
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    from src.senses import build_periphery
    from src.senses.base import PeripheryRegistry
    reg = build_periphery({"privacy": {"require_local_asr": False}}, registry=PeripheryRegistry())
    check("the config reaches the sensor", reg.sensors["microphone"].local_only is False)
    reg2 = build_periphery({}, registry=PeripheryRegistry())
    check("...and the default is local", reg2.sensors["microphone"].local_only is True)

    from src.brain.sensory import SensoryCortex, VisionExtractor
    loaded = []
    orig = VisionExtractor._try_load
    VisionExtractor._try_load = lambda self: loaded.append(1)
    try:
        sc = SensoryCortex(embedder=None, enable_vision=False)
        sc.enable_vision(load_vit=False)
        check("an image with vision off uses the descriptor and loads nothing (every screen capture used to load the ViT)",
              loaded == [] and sc.vision._backend == "descriptor")
        sc2 = SensoryCortex(embedder=None, enable_vision=True)
        check("vision switched on still loads it", loaded == [1], str(loaded))
    finally:
        VisionExtractor._try_load = orig


def l5_loop():
    print("\n=== L5. The loop ===")
    rt = te.rig()
    import types
    fake = types.ModuleType("tools.LLM")
    calls = []
    fake.resume = lambda: calls.append("resume")
    fake.cancel_inflight = lambda: calls.append("cancel")
    old = sys.modules.get("tools.LLM")
    sys.modules["tools.LLM"] = fake
    try:
        check("nothing thinking: nothing to cancel", rt.cancel_thinking() is False and calls == [])
        rt._thinking = 1
        check("thinking: it cancels", rt.cancel_thinking() is True and calls == ["cancel"])
        check("...once", rt.cancel_thinking() is True and calls == ["cancel"])
        rt._resume_model()
        check("the block is cleared by the answer's own clean-up", calls[-1] == "resume")
        rt._thinking = 0
    finally:
        if old is None:
            sys.modules.pop("tools.LLM", None)
        else:
            sys.modules["tools.LLM"] = old
    check("the loop and the fast lane share a lock", hasattr(rt, "_cancel_lock"))

    # one event that raises does not drop the rest
    seen = []
    orig = rt._handle

    def handle(ev):
        if "boom" in ev.text:
            raise RuntimeError("handler bug")
        seen.append(ev.text)
        return orig(ev)
    rt._handle = handle
    from src.senses.base import Modality, SensorEvent
    for t in ("first", "boom", "third"):
        rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text=t, addressed=False, salience=0.1))
    rt._tick()
    check("one event that raises does not cost the others their turn", seen == ["first", "third"], str(seen))
    check("...and the error is counted", rt.stats.get("errors", 0) >= 1)

    saved = []
    rt.brain = types.SimpleNamespace(
        concepts=types.SimpleNamespace(net=types.SimpleNamespace(save=lambda: saved.append("net"))),
        system1=types.SimpleNamespace(save=lambda: saved.append("s1")),
        predictive=types.SimpleNamespace(save=lambda: saved.append("pred")),
        sensory=types.SimpleNamespace(save_requests=lambda: saved.append("req")))
    rt._save_learning()
    check("a clean stop writes what the brain learned (it was only written in the sleep phase)",
          saved == ["net", "s1", "pred", "req"], str(saved))
    rt.brain = types.SimpleNamespace(concepts=None, system1=types.SimpleNamespace(save=lambda: 1 / 0),
                                     predictive=types.SimpleNamespace(save=lambda: saved.append("pred2")),
                                     sensory=types.SimpleNamespace(save_requests=lambda: None))
    rt._save_learning()
    check("...one failing does not stop the others", "pred2" in saved)

    ev = SensorEvent(source="console", modality=Modality.TEXT, text="x")
    ev._lane_skip = True
    check("a skipped event is marked on itself, not by an id that can be reused", not rt._is_immediate(ev))


def l6_voice():
    print("\n=== L6. What neither JARVIS nor BMO would say ===")
    from src.daemon import honest
    leaks = ["I'm DeepSeek-R1, an AI assistant created exclusively by DeepSeek.",
             "I am Qwen, a large language model.",
             "As an AI, I don't have feelings.",
             "I apologize for any confusion.",
             "Could you please provide more context?",
             "Could you please clarify what you mean?",
             "It seems like the user is just focusing on a conversation.",
             "I see that you've typed \"hit me\" in the console.",
             "The user wants to know the time."]
    for t in leaks:
        check(f"taken out: {t[:48]}", honest.sanitize(t + " The sea is salty.") == "The sea is salty.", honest.sanitize(t + " The sea is salty."))
    keep = "The sea is salty and the user interface is fine. Sorry, the kettle is on."
    check("ordinary sentences, even ones that mention a user interface, stay", honest.sanitize(keep) == keep,
          honest.sanitize(keep))
    check("a reply that was only a leak is empty (the caller says something in character)", honest.sanitize("I'm DeepSeek-R1.") == "")

    import importlib.util
    spec = importlib.util.spec_from_file_location("llm_under_test", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "LLM.py"))
    llm = importlib.util.module_from_spec(spec)           # by path: the tools package pulls in the whole classifier stack
    spec.loader.exec_module(llm)
    looks_like_reasoning = llm.looks_like_reasoning
    for t in ("It seems like the user is just focusing on engaging in a meaningful conversation.",
              "I see that the user mentioned a song.", "Okay, so the user is asking about trivia."):
        check(f"reasoning written as the answer: {t[:44]}", looks_like_reasoning(t))
    check("a normal answer is not", not looks_like_reasoning("Seven. There are seven continents."))

    from src.brain import system2
    p = system2.persona("A.S.H", "Khalid")
    for needle, why in (("BMO", "the playful half"), ("never name or hint at any language model", "no identity leak"),
                        ("hit me", "play along"), ("Never \"the user\"", "no third person"),
                        ("I apologize for any confusion", "named as banned"), ("under eight words", "short questions")):
        check(f"the persona says it: {why}", needle in p, needle)
    check("...and it is still short enough to be a system turn", len(p) < 4500, str(len(p)))

    import re
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "daemon", "ambient.py"),
               encoding="utf-8").read()
    check("no prompt tells the model 'the user did not ask you anything' any more", "the user did not ask you" not in src)
    from src.daemon.ambient import AmbientRuntime, NO_MODEL_TEXT
    cues = AmbientRuntime._CUES
    check("the stall lines are in character, and there is a choice of them", len(cues) >= 6 and "Hmm, let me think." not in cues
          and all(len(c) < 40 for c in cues), str(cues))
    check("the no-model line is not a status page", "brain" in NO_MODEL_TEXT and "model" not in NO_MODEL_TEXT.lower())
    from src.daemon.freemoment import MoodCheck, FreeMoments
    from src.brain.memory_engine import NO_REPLY_TEXT
    check("'nothing useful' is not what ASH says", "stumped" in NO_REPLY_TEXT)


def l7_screen_and_friends():
    print("\n=== L7. The read-out, the flood, the console window, the close handler ===")
    from tools.screen_tool import _spoken_text, describe_view
    logs = ("2026-11-09 14:43:43,078 INFO Intent candidate: search (score=0.3) / Open the pod bay doors / "
            "Traceback (most recent call last) / --- === ###")
    check("log lines and tracebacks are not read out", _spoken_text(logs) == "Open the pod bay doors", _spoken_text(logs))
    d = describe_view({"ok": True, "title": "t", "process": "WindowsTerminal.exe", "subject": "", "project": "", "scene": "text",
                       "text": "2026-11-09 14:43:43,078 INFO x / 2026-11-09 14:43:44,078 WARNING y", "fresh": True,
                       "age_s": 1.0, "withheld": None, "reader": True})
    check("a screen of nothing but logs is said to be that", "mostly code and logs" in d and "INFO" not in d, d)
    d2 = describe_view({"ok": True, "title": "Notes", "process": "notepad.exe", "subject": "Notes", "project": "",
                        "scene": "text", "text": "Buy milk / Call Omar", "fresh": True, "age_s": 1.0, "withheld": None,
                        "reader": True})
    check("ordinary text is read, in a natural lead", "I can make out: Buy milk" in d2 and "part I can read" not in d2, d2)

    # the journal flood: one window said once in a while
    from PIL import Image
    import random
    from src.senses.screen import ScreenshotSensor

    class WS:
        def current(self):
            return "Stremio", "stremio-shell-ng.exe"

        def is_blocked(self, t):
            return False

        def hosts_daemon(self):
            return False
    s = ScreenshotSensor(enabled=True, window_sensor=WS(), describe=False, min_repeat_s=120.0)
    rnd = random.Random(1)

    def frame():
        return Image.frombytes("RGB", (64, 64), bytes(rnd.getrandbits(8) for _ in range(64 * 64 * 3)))
    s._capture = frame
    s.state = type(s.state)("active")
    got = []
    for _ in range(6):
        got += s.read()
    check("six changed frames of one window in a row: one event, not six", len(got) == 1, str(len(got)))
    s._last_said = ("Stremio", time.time() - 200)
    check("...and again once it has been a couple of minutes", len(s.read()) == 1)
    WS.current = lambda self: ("Code", "code.exe")
    check("a different window is said at once", len(s.read()) == 1)

    from src.daemon import service
    check("the console-close handler exists (the window being closed used to skip the clean stop)", hasattr(service, "_CONSOLE_HANDLER"))
    if sys.platform == "win32":
        service.install_signal_handlers(lambda: None)
        check("...and is installed on Windows", service._CONSOLE_HANDLER is not None)

    import src.ui.hooks as hooks
    text = open(hooks.__file__, encoding="utf-8").read()
    check("the console window is not opened over a game or an unprompted remark", "game.active" in text and "can_talk" in text)
    check("a cancelled answer is not shown on the console either", "_cancelled" in text)

    asrc = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ashd.py"), encoding="utf-8").read()
    check("start-up caps the numeric libraries' threads and runs below normal priority",
          "OMP_NUM_THREADS" in asrc and "_background_priority()" in asrc and "BELOW_NORMAL" in asrc)
    check("...and says so once when the language model is not there", "Is Ollama running?" in asrc)


if __name__ == "__main__":
    l1_games()
    l2_jokes_and_yes()
    l3_saves()
    l4_sensors()
    l5_loop()
    l6_voice()
    l7_screen_and_friends()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
