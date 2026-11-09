"""
tst_session4.py -- round three of the JARVIS / BMO ideas, one section per item.

  S1  The model is preloaded at start (one token, kept alive, no thinking cost)
  S4  A model claiming it sent/called/ordered something is replaced, unless a tool ran
  S5  Each reply carries how it was made, shown as a tappable card in the console
  S3  Wikipedia facts are on in the daemon unless switched off
  S2  A slow model answer gets a short spoken cue first; a quick one does not

Run:  PYTHONIOENCODING=utf-8 python tst_session4.py
"""
import importlib.util
import time
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402,F401
import tst_wondering as tw  # noqa: E402

check = tw.check


def load_llm():
    spec = importlib.util.spec_from_file_location("ash_llm_s4", os.path.join(HERE, "tools", "LLM.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    return L


def s1():
    print("\n=== S1. Model preload ===")
    L = load_llm()
    sent = []

    class R:
        def __init__(self, code=200, text="{}"):
            self.status_code, self.text = code, text

    orig = L.requests.post
    L.requests.post = lambda url, json=None, timeout=None, **k: (sent.append(dict(json)), R())[1]
    try:
        llm = L.LLM(mode="ollama", ollama_model="m", ollama_url="http://x/api/chat")
        check("warm() reports success", llm.warm() is True)
        p = sent[-1]
        check("one token only", p["options"]["num_predict"] == 1)
        check("kept loaded for hours", p["keep_alive"] == "6h")
        check("same model as real calls", p["model"] == "m")
        L.requests.post = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        check("an unreachable server is False, not a crash", llm.warm() is False)
        off = L.LLM(mode="openrouter", openrouter_key="k")
        check("a non-ollama backend does nothing", off.warm() is False)
    finally:
        L.requests.post = orig
    src = open(os.path.join(HERE, "ashd.py"), encoding="utf-8").read()
    check("the daemon starts the preload in the background", "ash-preload" in src and "ASH_PRELOAD_MODEL" in src)


def s2():
    print("\n=== S2. Thinking cue ===")
    import time
    for wait, work, expect in ((0.05, 0.4, True), (1.0, 0.01, False), (0.0, 0.3, False)):
        rt = te.rig(think_cue_s=wait)
        orig = rt.ash.think

        def slow(q, image=None, context="", orig=orig, work=work):
            time.sleep(work)
            return orig(q, image=image, context=context)
        rt.ash.think = slow
        rt.say("tell me something interesting about owls please")
        time.sleep(0.1)
        cues = [t for t in rt.out if t in rt._CUES]
        check(f"wait={wait} work={work}: cue {'spoken' if expect else 'not spoken'}", bool(cues) == expect, str(rt.out))
        if expect:
            check("the real answer still follows the cue", rt.out[-1] == "Sure." and rt.out.index(cues[0]) < len(rt.out) - 1)
    th = te.rig(think_cue_s=0.05)
    check("an unprompted remark gets no cue", th._thinking_cue(False) is None)


def s3():
    print("\n=== S3. Facts on by default ===")
    src = open(os.path.join(HERE, "ashd.py"), encoding="utf-8").read()
    check("the daemon defaults facts to enabled", 'get("facts", {}).get("enabled", True)' in src)
    from src.daemon.facts import FactsConfig
    check("a bare config stays off (tests, embedding)", FactsConfig().enabled is False)
    check("explicit false still turns it off", FactsConfig.from_dict({"enabled": False}).enabled is False)


def s4():
    print("\n=== S4. Claims of undone actions ===")
    from src.daemon import honest
    for t in ("I've sent the email to Sara.", "Sure, I just called him for you.", "I have set a reminder for 5.",
              "The message has been sent.", "Done! I've ordered the pizza."):
        check(f"caught: {t}", honest.claimed_unperformed(t))
    for t in ("I can't send emails.", "Sara has sent the report to me.", "I've read your email.",
              "Okay. I called it a day.", "It's 5 o'clock."):
        check(f"left alone: {t}", not honest.claimed_unperformed(t))
    seen = []
    rt = te.rig()
    rt.response_sink = lambda *a: None
    rt._last_tool_used = False
    rt.ash.think = None
    out = []
    rt._real_sink = out.append
    meta = {"addressed": True, "pathway": "slow"}
    # drive the real sink path: it hands the final text to the configured response sink
    rt2 = te.rig()
    rt2._last_tool_used = False
    rt2.response_sink("I've sent the email to Sara.", meta)
    check("the sink replaces it", rt2.out[-1] == honest.NOT_DONE, str(rt2.out))
    rt2._last_tool_used = True
    rt2.response_sink("I've sent the email to Sara.", meta)
    check("a tool that really ran is exempt", rt2.out[-1] != honest.NOT_DONE, str(rt2.out))


def s5():
    print("\n=== S5. The 'why' card ===")
    from src.daemon.explain import route_label
    a = {"addressed": True}
    cases = [
        ({**a, "pathway": "slow"}, "Hello.", "", "model"),
        ({**a, "pathway": "slow"}, "It's 5.", "get_time", "tool"),
        ({**a, "pathway": "fast"}, "It's 5.", "get_time", "reflex"),
        ({**a, "pathway": "slow", "outcome": "desktop"}, "Opening Chrome.", "", "model"),
        ({**a, "pathway": "offer", "outcome": "desktop"}, "Mic on.", "", "rules"),
        ({**a}, "Ada Lovelace was a mathematician. (Wikipedia)", "", "wikipedia"),
        ({"addressed": False, "pathway": "offer"}, "Been at it a while.", "", "noticed"),
        ({**a, "degraded": True}, "Model is down.", "", "notice"),
    ]
    for meta, text, tool, want in cases:
        got = route_label(meta, text, tool)
        check(f"{want}: {text[:30]!r}", got["route"] == want and bool(got["detail"]), str(got))
    html = open(os.path.join(HERE, "src", "ui", "static", "console.html"), encoding="utf-8").read()
    check("the console shows a tappable route tag and a card", "tag route" in html and "why-card" in html)
    hooks = open(os.path.join(HERE, "src", "ui", "hooks.py"), encoding="utf-8").read()
    check("the bus record carries the route", "route_detail" in hooks)


def s6():
    print("\n=== S6. The face reacts to what you say, and has idle life ===")
    import random
    from src.face.moods import IdleLife, reaction_to
    cases = [("thank you so much", "love"), ("haha that's funny", "very_happy"), ("good night ash", "sleepy"),
             ("wow", "surprised"), ("you're wrong", "sad"), ("shut up", "annoyed"), ("hello", "wink"),
             ("good morning", "happy"), ("I'm sorry about that", "acknowledge"), ("I'm so tired", "sleepy"),
             ("what time is it", None), ("set a timer for five minutes", None), ("thanksgiving is in November", None)]
    for text, want in cases:
        check(f"{text!r} -> {want}", reaction_to(text) == want, str(reaction_to(text)))
    check("a long paragraph is not read for feeling", reaction_to("thanks " + "word " * 60) is None)

    life = IdleLife(random.Random(7))
    check("nothing while someone just spoke", life.step(0, 5) is None)
    check("nothing while ASH is busy", life.step(100, 300, busy=True) is None)
    seen, t, prev = [], 1000.0, None
    for _ in range(400):
        t += 5
        a = life.step(t, 300)
        if a:
            seen.append(a)
            check("never the same twice in a row", a != prev) if prev else None
            prev = a
    check("it does things, but rarely", 8 <= len(seen) <= 40, str(len(seen)))
    check("only animations that exist", set(seen) <= {"look_around", "double_blink", "curious", "wink"}, str(set(seen)))
    from src.face import BUILTINS
    check("they are all built-in animations", all(n in BUILTINS for n in {"look_around", "double_blink", "curious", "wink",
                                                                         "love", "very_happy", "sleepy", "surprised",
                                                                         "sad", "annoyed", "happy", "acknowledge"}))
    night = IdleLife(random.Random(1))
    check("late at night it gets sleepy", night.step(0, 300, hour=2) == "sleepy")

    import tst_face_react as fr
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.face.bridge import attach_to_daemon
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [time.mktime((2026, 3, 10, 14, 0, 0, 0, 0, -1))]
    out = []
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    rt = AmbientRuntime(ash=tw.Ash(), periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9),
                        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    p = fr.Player()
    bridge = attach_to_daemon(rt, p)
    bridge.idle_life = IdleLife(random.Random(3))
    clock[0] += 3
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="thank you!", addressed=True,
                          salience=0.9, urgency=0.6))
    rt._tick()
    check("'thank you' made the real face show love", "love" in p.played, str(p.played))
    check("...and again once the reply lands, instead of a nod", p.played.count("love") >= 2
          and "acknowledge" not in p.played, str(p.played))
    before = len(p.played)
    for _ in range(60):
        clock[0] += 10
        idle.value = 0.0
        rt._tick()
    check("a quiet room: the face does something on its own", len(p.played) > before, str(p.played[before:]))
    check("...only from the idle set", set(p.played[before:]) <= {"look_around", "double_blink", "curious", "wink"},
          str(set(p.played[before:])))


def s7():
    print("\n=== S7. Quiet till a time ===")
    from src.daemon import quiet
    now = time.mktime((2026, 11, 9, 22, 0, 0, 0, 0, -1))
    cases = [("quiet till 9", 11.0), ("be quiet until 9am", 11.0), ("do not disturb for 2 hours", 2.0),
             ("quiet for an hour", 1.0), ("quiet for 30 minutes", 0.5), ("quiet until noon", 14.0),
             ("ash be quiet until 3:30 pm", 17.5)]
    for text, hrs in cases:
        r = quiet.parse(text, now)
        check(f"{text!r} -> {hrs}h", r is not None and abs((r[0] - now) / 3600 - hrs) < 0.01, str(r))
    for text in ("quiet mode", "quiet till 25", "what time is it", "quiet for 0 minutes", "be quiet"):
        check(f"{text!r} is not a timed quiet", quiet.parse(text, now) is None)

    rt = te.rig()
    rt.say("quiet for 2 hours")
    check("it says until when", rt.out[-1].startswith("Okay, quiet until "), rt.out[-1])
    check("the gate is muted", rt.gate.muted is True)
    rt.t[0] += 3600
    rt._tick()
    check("still quiet after an hour", rt.gate.muted is True)
    rt.t[0] += 3700
    rt._tick()
    check("back on its own after two", rt.gate.muted is False)
    rt.say("quiet for an hour")
    rt.say("wake up")
    check("'wake up' ends it and forgets the timer", rt.gate.muted is False and rt._quiet_until is None)
    rt.say("be quiet till 9")
    rt.say("quiet mode")
    check("an untimed 'quiet mode' still works", rt.gate.muted is True)


def s8():
    print("\n=== S8. Wake ring and mood tint in the console ===")
    html = open(os.path.join(HERE, "src", "ui", "static", "console.html"), encoding="utf-8").read()
    check("the page rings the face when it hears you", "function heard()" in html and "heardring" in html)
    check("the ring is not drawn for quiet button records", "!m.quiet" in html)
    check("the page tints by mood, with reduced-motion respected",
          'data-mood="love"' in html and "prefers-reduced-motion" in html)
    from src.ui import hooks
    rt = te.rig(with_face=True)
    check("no mood at first", hooks._mood(rt) is None)
    rt.face.on_user_text("thank you so much")
    check("a thank-you shows love in the status", hooks._mood(rt) == "love", str(hooks._mood(rt)))
    check("it fades after half a minute", rt.face.shown_mood(time.time() + 40) is None)
    rt.face._mood = "sleepy"
    check("then the slower mood shows", rt.face.shown_mood(time.time() + 40) == "sleepy")
    bare = te.rig()
    check("no face: the status just has no mood", hooks._mood(bare) is None)


def s9():
    print("\n=== S9. Undo buttons on the timeline ===")
    from src.daemon.forget import UndoStack
    t = [100.0]
    u = UndoStack(window_s=600, clock=lambda: t[0])
    log = []
    a = u.push("one", lambda: log.append("one"))
    b = u.push("two", lambda: log.append("two"))
    check("push returns distinct ids", a != b)
    check("undo_id takes back an earlier one, not just the last", u.undo_id(a) == "Undone: one." and log == ["one"])
    check("a step can't be undone twice", u.undo_id(a) is None)
    check("the other is still there", len(u) == 1 and u.undo() == "Undone: two.")
    c = u.push("old", lambda: log.append("old"))
    t[0] += 700
    check("too old: not undone", u.undo_id(c) is None and "old" not in log)

    from src.ui.bus import UIBus
    from src.ui.hooks import attach_ui
    rt = te.rig()
    bus = UIBus()
    on_say, on_control, extra = attach_ui(rt, bus)
    hit = []
    uid = rt.undo.push("Lieth is your friend", lambda: hit.append(1), 100.0)
    rec = [r for r in bus.history() if r["kind"] == "undoable"]
    check("the timeline is told of it", rec and rec[-1]["uid"] == uid and rec[-1]["label"] == "Lieth is your friend", str(rec))
    rt.undo.clock = lambda: 100.0
    res = on_control("undo", str(uid))
    check("the button's command really undoes it", res.get("ok") is True and hit == [1], str(res))
    check("and the page is told", any(r["kind"] == "undone" and r["ok"] for r in bus.history()))
    res = on_control("undo", str(uid))
    check("again: refused, page told 'gone'", res.get("ok") is False and
          any(r["kind"] == "undone" and not r["ok"] for r in bus.history()))
    check("a bad id is refused", on_control("undo", "x").get("ok") is False)
    html = open(os.path.join(HERE, "src", "ui", "static", "console.html"), encoding="utf-8").read()
    check("the page has the undo row", "function addUndoable" in html and '"undone"' in html)


if __name__ == "__main__":
    s1()
    s2()
    s3()
    s4()
    s5()
    s6()
    s7()
    s8()
    s9()
    print("\nALL OK" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
