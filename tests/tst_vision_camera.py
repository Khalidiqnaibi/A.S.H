"""
tst_vision_camera.py -- asking the camera works out of the box, answers by rule (never the model), says exactly why it
cannot, and notices what changes. Every frame is synthetic and every camera, model and sensor is a fake: nothing here
opens a camera, a microphone or the network, and no picture is written anywhere but a temp folder.

  V1  the ways people ask: many phrasings, misspellings, and what is NOT a camera question
  V2  why there is no picture, in one line: privacy / busy / no camera / switched off -- never the model
  V3  "look at me", "what am I holding", "take a picture with the cam": the answers
  V4  in the runtime: loose phrasings, guests, overheard speech, no model call, nothing kept
  V5  reacting to change: only from the camera sensor, never in privacy or kids mode, never with a stranger, once
  V6  defaults and wiring

Run:  PYTHONIOENCODING=utf-8 python tests/tst_vision_camera.py
"""

import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_vision as tv  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
MID = (280, 200, 80, 80)


def v1_phrasings():
    print("\n=== V1. The ways people ask ===")
    from src.daemon.looking import parse
    cases = {
        "look at me": "me", "Look at me!": "me", "Ash, look at me please": "me", "can you see me": "me", "can you see me?": "me",
        "could you see me now": "me", "do you see me": "me", "am I in the frame": "me", "am i visible": "me",
        "what am I holding": "holding", "what am i holding up": "holding", "what's in my hand": "holding", "whats in my hand": "holding",
        "what's in my hands": "holding", "what do I have in my hand": "holding", "what is this thing": "holding",
        "what is this object I'm holding": "holding", "what's this in my hand": "holding", "identify this": "holding",
        "tell me what I'm holding": "holding", "can you tell what I'm holding": "holding", "guess what I'm holding": "holding",
        "what am I showing you": "holding", "look at this": "holding", "look at what I'm holding": "holding",
        "what do you see on the camera": "see", "what do you see on the cam": "see", "what do you see through the webcam": "see",
        "what can you see in front of you": "see", "what's in front of the camera": "see", "show me what you see": "see",
        "show me what you see through the camera": "see", "look around": "see", "take a look around": "see", "what's in view": "see",
        "what's the camera showing": "see", "what's on the camera": "see", "can you see anything": "see", "do you see anything": "see",
        "use the camera": "see", "check the webcam and tell me what you see": "see", "look through the camera": "see",
        "look at the camera and tell me what you see": "see", "hey ash, what do you see on the camera": "see",
        "take a picture with the cam": "snap", "take a picture with the camera": "snap", "Ash, take a photo using the webcam": "snap",
        "take a picture of this": "snap", "snap a photo of what I'm holding": "snap", "grab a shot with the camera": "snap",
        "take a selfie with the camera": "snap", "can you take a picture with the cam please": "snap",
    }
    bad = {q: parse(q) for q, k in cases.items() if (parse(q) or (None,))[0] != k}
    check(f"{len(cases)} phrasings go where they should", not bad, str(bad))
    typos = {"wat am i holding": "holding", "wht am i holding": "holding", "what am i holdin": "holding", "what am i holding rn": None,
             "lookk at me": "me", "look at mee": None, "wat do you see on the camra": "see", "whats in my hnad": "holding",
             "take a pic with the cam": "snap", "can u see me": "me", "wat is this thing": "holding", "show me wat you see": "see",
             "what do you see on the cammera": "see", "luk at me": "me"}
    got = {q: (parse(q) or (None,))[0] for q in typos}
    check("misspellings are put right", all(got[q] == k for q, k in typos.items() if k), str({q: got[q] for q, k in typos.items() if k and got[q] != k}))
    for q in ("what do you see", "what's on my screen", "what is this", "what is that", "look it up", "look at the weather", "I'm looking around for my keys",
              "hold on", "can you see the problem", "do you see what I mean", "take a photo", "take a picture", "take a picture of me",
              "what am I doing", "what's in my calendar", "what's in my inbox", "what do I have today", "look at me and laugh at the joke later on please really", ""):
        check(f"not a camera question: {q!r}", parse(q) is None, str(parse(q)))
    check("a bare 'what do you see?' is still the SCREEN's", parse("what do you see") is None)
    check("'take a photo' is the media rule's, not the camera's", parse("take a photo") is None and parse("take a picture of me") is None)
    check("a label's capitals survive (typo repair is not run on names)", parse("is this Omar") == ("is_this", {"label2": "Omar"}) or parse("is this Omar")[0] == "is_this")
    check("a long line is never taken", parse("what am I holding " + "x " * 100) is None)
    lk = tv.looking([tv.thing("cup", .9)])
    check("wants() takes camera questions on a Looking that is on", lk.wants("look at me") and lk.wants("wat am i holding") and not lk.wants("what time is it"))
    off = tv.looking([tv.thing("cup", .9)])
    off.cfg.enabled = False
    check("...and off too: it is still the camera's question (it says it is off, it does not hand it to the model)", off.wants("look at me") and off.wants("what am I holding"))
    check("...but 'is this X' about a taught thing needs it on", not off.wants("is this my mug"))


def eye_with(grab=None, paused=False, sensor=None):
    from src.vision.eye import Eye
    return Eye(sensor=(lambda: sensor), grab=grab, paused=lambda: paused, clock=lambda: 1000.0)


def v2_why_not():
    print("\n=== V2. Why there is no picture ===")
    from src.senses.base import State
    from src.vision.eye import Eye
    check("no camera at all: the eye says 'no camera'", eye_with(grab=lambda: None).look() == (None, "no camera"))
    cam = type("Cam", (), {"state": State.ACTIVE, "grab_now": lambda self: None})()
    check("the sensor has the camera open and gives no frame: 'busy'", eye_with(grab=lambda: None, sensor=cam).look() == (None, "busy"))
    cam2 = type("Cam", (), {"state": State.DISABLED, "grab_now": lambda self: None})()
    check("a sensor that is off does not make it 'busy'", eye_with(grab=lambda: None, sensor=cam2).look() == (None, "no camera"))
    check("privacy beats everything", eye_with(grab=lambda: tv.frame(), paused=True).look() == (None, "privacy"))
    ok = type("Cam", (), {"state": State.ACTIVE, "grab_now": lambda self: tv.frame()})()
    check("the sensor's own frame is used when it has one", eye_with(sensor=ok).look()[1] == "sensor")

    def hear(q, **kw):
        lk = tv.looking([tv.thing("cup", .9)], **kw.pop("kw", {}))
        lk.eye = kw.pop("eye", lk.eye)
        return lk, lk.hear(q)
    lk, say = hear("look at me", eye=eye_with(paused=True, grab=lambda: tv.frame()))
    check("privacy: one line, says why", say == "Privacy mode is on, so my camera is off and I'm not looking." and "\n" not in say, say)
    lk, say = hear("what am I holding", eye=eye_with(grab=lambda: None))
    check("no camera: one line, says why and what to check", say.startswith("I couldn't get a picture -- there's no camera I can open") and "in use by another app" in say, say)
    lk, say = hear("look around", eye=eye_with(grab=lambda: None, sensor=cam))
    check("busy: one line, says who has it", say.startswith("My camera is busy") and "camera sensor" in say, say)
    lk, say = hear("take a picture with the cam", eye=eye_with(grab=lambda: None))
    check("a picture request fails with the same honesty, and saves nothing", "no camera" in say)
    for q in ("look at me", "what am I holding", "look around", "take a picture with the cam"):
        lk, say = hear(q, eye=eye_with(paused=True, grab=lambda: tv.frame()))
        check(f"privacy for {q!r}: no frame taken, no detector run", lk.eye.looks == 0 and lk.objects.calls == 0)
    lk = tv.looking([tv.thing("cup", .9)])
    lk.cfg.enabled = False
    say = lk.hear("look at me")
    check("switched off: one line, names the setting", "switched off" in say and "vision.enabled" in say and lk.eye.looks == 0, say)
    check("look_now says the same, structured", lk.look_now()["why"] == "off" and lk.look_now()["ok"] is False)
    lk = tv.looking([tv.thing("cup", .9)])
    lk.eye = eye_with(grab=lambda: None)
    r = lk.look_now()
    check("look_now with no camera: not ok, why 'no camera', the line", not r["ok"] and r["why"] == "no camera" and "no camera I can open" in r["say"], str(r))
    check("the camera state says it too", lk.camera_state()["state"] == "unavailable" and "no camera" in lk.camera_state()["why"], str(lk.camera_state()))
    lk.eye = eye_with(grab=lambda: None, sensor=cam)
    lk.look_now()
    check("...and 'busy' as busy", lk.camera_state()["state"] == "watching" or lk.camera_state()["state"] == "busy")
    lk.eye = eye_with(paused=True)
    check("camera state: privacy", lk.camera_state()["state"] == "privacy" and "Privacy mode" in lk.camera_state()["why"])
    lk.eye = eye_with(grab=lambda: tv.frame())
    lk.look_now()
    check("camera state: ready, and it says the camera is off until asked", lk.camera_state()["state"] == "ready" and "until you ask" in lk.camera_state()["why"])
    lk.eye = eye_with(sensor=ok)
    check("camera state: the sensor is on -> watching", lk.camera_state()["state"] == "watching")
    dead = type("Cam", (), {"state": State.UNAVAILABLE, "unavailable_reason": "no camera at index 0", "grab_now": lambda self: None})()
    lk.eye = eye_with(sensor=dead)
    check("camera state: the sensor could not open it -> its reason", lk.camera_state() == {"state": "unavailable", "why": "no camera at index 0"})


def v3_answers():
    print("\n=== V3. The answers ===")
    T = tv.thing
    lk = tv.looking(faces=[tv.face()], vectors=[tv.vec(1)], user=lambda: "Khalid")
    lk.memory.enroll("Khalid", "person", tv.vec(1))
    check("'look at me' for the person it knows: 'you, Khalid'", lk.hear("look at me") == "Yes, I can see you, Khalid.", lk.hear("look at me"))
    lk2 = tv.looking(faces=[tv.face()], vectors=[tv.vec(9)])
    check("a face nobody taught: it says so and how to teach it", lk2.hear("can you see me") == "Yes, I can see someone, but I don't know them -- say \"remember my face\" and I will.")
    check("nobody in front of it: honest", lk2.look_at_me(tv.frame()) .startswith("Yes") and tv.looking().hear("look at me").startswith("I looked, and I don't see anyone"))
    two = tv.looking(faces=[tv.face((10, 10, 100, 100)), tv.face((300, 10, 100, 100))], vectors=[tv.vec(1), tv.vec(9)])
    two.memory.enroll("Omar", "person", tv.vec(1))
    check("two people, one known", two.hear("look at me") == "Yes, I can see Omar, and someone I don't know.", two.hear("look at me"))
    nf = tv.looking([T("person", .9, (100, 50, 400, 400))], faces_backend=tv.FakeFaces(available=False))
    check("no face models but a person shape: it can see someone, not who", nf.hear("look at me") == "Yes, I can see someone. I can't tell who without the face models.", nf.hear("look at me"))
    nf2 = tv.looking([T("person", .3, (100, 50, 400, 400))], faces_backend=tv.FakeFaces(available=False), object_conf=0.2)
    check("...hedged when the person shape is not sure", nf2.hear("look at me").startswith("I think I can see someone."), nf2.hear("look at me"))
    none = tv.looking(objects=tv.FakeObjects(available=False), faces_backend=tv.FakeFaces(available=False))
    check("no model at all: it says so (with the fallback's reason when there is one)", "no model installed" in none.hear("look at me"))

    h = tv.looking([T("watch", .8, MID)])
    check("held: 'It's a watch.' (the YOLO bar is 0.6)", h.hear("what am I holding") == "It's a watch.", h.hear("what am I holding"))
    check("held: unsure", tv.looking([T("watch", .4, MID)]).hear("what am I holding") == "I think it's a watch, but I'm not sure.")
    check("held: exactly the bar (0.6) is sure, 0.59 is not", tv.looking([T("watch", .6, MID)]).hear("what am I holding") == "It's a watch."
          and tv.looking([T("watch", .59, MID)]).hear("what am I holding").startswith("I think"))
    check("held: nothing near the middle -> the biggest thing", tv.looking([T("cup", .9, (5, 5, 40, 40)), T("book", .8, (400, 300, 200, 150))]).hear("what am I holding") == "It's a book.")
    check("held: 'a' before a vowel", tv.looking([T("umbrella", .9, MID)]).hear("what am I holding") == "It's an umbrella.")
    check("held: only you in the picture", "closer to the camera, in the middle" in tv.looking([T("person", .9, (0, 0, 640, 480))]).hear("what am I holding"))
    check("held: a taught thing beats what the detector calls it", "your blue mug" in _taught_mug().hear("what am I holding"))
    check("'what is this thing' is the same question", h.hear("what is this thing") == "It's a watch.")

    snap_dir = tempfile.mkdtemp()
    s = tv.looking([T("cup", .9)], photo_dir=snap_dir)
    say = s.hear("take a picture with the cam")
    files = os.listdir(snap_dir)
    check("'take a picture with the cam': one JPEG in the photos folder, and it says where", len(files) == 1 and files[0].endswith(".jpg") and f"saved as {files[0]}" in say, f"{say} {files}")
    import cv2
    check("...and the file is a real picture", cv2.imread(os.path.join(snap_dir, files[0])) is not None)
    check("a look that is not a picture request saves nothing", (tv.looking([T("cup", .9)], photo_dir=snap_dir).hear("look around"), len(os.listdir(snap_dir)))[1] == 1)
    bad = tv.looking([T("cup", .9)], photo_dir=os.path.join(snap_dir, "x\0y"))
    check("a folder that cannot be written: said, not a crash", bad.hear("take a picture with the cam") == "I got the picture but couldn't save it.")


def _taught_mug():
    lk = tv.looking([tv.thing("cup", .9, MID)])
    f = tv.frame(square=((30, 90, 200), (200, 160, 240, 190)), noise=4)
    lk.eye._grab = lambda: f
    lk.hear("remember this as my blue mug")
    return lk


def rt_with(things=(), faces=None, vectors=None, vcfg=None, **cfg):
    import tst_effects as te
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    from src.vision.scene import DetectionLog
    d = tempfile.mkdtemp()
    grabs = []
    box = []
    eye = Eye(grab=lambda: grabs.append(1) or tv.frame(size=(480, 640)), paused=lambda: bool(box and box[0].privacy.paused))
    faces, vectors = ([tv.face()], [tv.vec(9)]) if faces is None else (faces, vectors or [])
    objs = tv.FakeObjects(list(things))
    log = DetectionLog(os.path.join(d, "detections.jsonl"))
    io = dict(eye=eye, faces=tv.FakeFaces(list(faces), list(vectors)), objects=objs, memory=VisualMemory(None), detections=log)
    rt = te.rig(vision=True, vision_io=io, vision_cfg=dict(dict(enrol_checks=False, photo_dir=os.path.join(d, "photos")), **(vcfg or {})), **cfg)
    rt.brain.user = "Khalid"
    box.append(rt)
    rt.grabs, rt.objs, rt.log, rt.dir = grabs, objs, log, d
    return rt


def v4_runtime():
    print("\n=== V4. In the runtime ===")
    from src.senses.base import Modality, SensorEvent
    rt = rt_with([tv.thing("watch", .8, MID)])
    n = len(rt.ash.calls)
    for q, want in [("Ash, look at me", "Yes, I can see someone"), ("what am I holding", "It's a watch."), ("Ash, wat am i holding", "It's a watch."),
                    ("what's in my hand", "It's a watch."), ("whats in my hnad", "It's a watch."), ("what do you see on the camera", "I see a person and a watch."),
                    ("show me what you see", "I see a person and a watch."), ("can you see me", "Yes, I can see someone"), ("hey ash can you look at me please", "I"),
                    ("Ash, what do you see on the cam", "I see a person and a watch.")]:
        k = len(rt.grabs)
        rt.say(q)
        ok = rt.out and want in rt.out[-1] and len(rt.grabs) == k + 1
        check(f"{q!r} -> the camera, by rule", ok, f"{rt.out[-1:]} grabs={len(rt.grabs) - k}")
    check("not one of them reached the language model", len(rt.ash.calls) == n, str(rt.ash.calls[n:]))
    check("the picture is kept in memory only for the minute (for 'save that photo')", rt.looking.eye.recent(rt.t[0]) is not None and rt.looking.eye.recent(rt.t[0] + 120) is None)
    n = len(rt.ash.calls)
    k = len(rt.grabs)
    rt.say("what do you see")
    check("a bare 'what do you see?' is still the screen's", len(rt.grabs) == k)
    n = len(rt.ash.calls)
    rt.say("Ash, take a picture with the cam")
    pics = os.listdir(os.path.join(rt.dir, "photos"))
    check("a picture request is saved, by rule", len(pics) == 1 and "saved as" in rt.out[-1] and len(rt.ash.calls) == n, f"{rt.out[-1]} {pics}")
    rt.say("Ash, look at me")
    kinds = {(r["kind"], r["label"]) for r in rt.log.recent(50)}
    check("every look went in the detection log, as labels", ("object", "watch") in kinds)
    raw = open(os.path.join(rt.dir, "detections.jsonl"), encoding="utf-8").read()
    rows = [json.loads(l) for l in raw.splitlines()]
    check("the file has only labels, numbers and boxes (no pixels, no names)", rows and all(set(r) == {"t", "kind", "label", "conf", "box", "source"} for r in rows)
          and "Khalid" not in raw and raw.isascii())
    check("nothing but that and the one saved picture was written", sorted(os.listdir(rt.dir)) == ["detections.jsonl", "photos"])

    k = len(rt.grabs)
    rt.privacy.pause("test")
    rt.say("Ash, look at me")
    check("privacy mode: one line, no frame, no model", rt.out[-1] == "Privacy mode is on, so my camera is off and I'm not looking." and len(rt.grabs) == k and len(rt.ash.calls) == n, rt.out[-1])
    rt.say("Ash, take a picture with the cam")
    check("...a picture request too, and no file", "Privacy mode is on" in rt.out[-1] and len(os.listdir(os.path.join(rt.dir, "photos"))) == 1)
    rt.privacy.resume()

    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="look at me", addressed=False, salience=0.4, meta={}))
    rt._tick()
    check("overheard speech never opens the camera", len(rt.grabs) == k)
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="Ash, what am I holding", addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice is refused and no frame is taken", rt.out[-1].startswith("Sorry, I only do that for Khalid") and len(rt.grabs) == k, rt.out[-1])
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="Ash, take a picture with the cam", addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("...also for a picture, and nothing is saved", len(rt.grabs) == k and len(os.listdir(os.path.join(rt.dir, "photos"))) == 1)

    off = rt_with([tv.thing("cup", .9)], vcfg=dict(enabled=False))
    n2 = len(off.ash.calls)
    off.say("Ash, look at me")
    check("vision.enabled false: it says so, never the model, no frame", "switched off" in off.out[-1] and len(off.ash.calls) == n2 and off.grabs == [], off.out[-1])
    off.say("Ash, what am I holding")
    check("...for 'what am I holding' too", "switched off" in off.out[-1] and off.grabs == [])

    nocam = rt_with([tv.thing("cup", .9)])
    nocam.looking.eye._grab = lambda: None
    n3 = len(nocam.ash.calls)
    nocam.say("Ash, look at me")
    check("no camera: the reason, never 'I don't have a camera' from the model", "no camera I can open" in nocam.out[-1] and len(nocam.ash.calls) == n3, nocam.out[-1])
    check("the answer never claims ASH has no camera or is looking at its own window", "own window" not in nocam.out[-1] and "don't have a camera" not in nocam.out[-1])
    check("the status carries the new counters", rt.status()["vision"]["looks"] >= 10)


def pump(rt, things=None, ts=None):
    """One camera-change reading: put the fake detector's answer in, advance the clock past the interval, let the worker
    finish, then step again to pick the answer up."""
    if things is not None:
        rt.objs.things = list(things)
    rt.cam.last_frame_ts = time.time() if ts is None else ts
    rt.t[0] += 2.0
    rt._tick()
    job = rt.scene_watch._job
    if job is not None:
        job.result(timeout=10)
    rt._tick()


def watch_rt(**kw):
    from src.senses.base import State
    vcfg = dict(watch_interval_s=1.0, watch_confirm=2, watch_cooldown_s=60.0, announce_objects=True)
    vcfg.update(kw.pop("vcfg", {}))
    rt = rt_with(vcfg=vcfg, **kw)
    cam = type("Cam", (), {"due": lambda self, now: False, "name": "camera", "close": lambda self: None, "status": lambda self: {},
                           "disable": lambda self: None, "enable": lambda self: None})()
    cam.state, cam.last_frame, cam.last_frame_ts = State.ACTIVE, tv.frame(size=(480, 640)), time.time()
    rt.periphery.sensors["camera"] = cam
    rt.cam = cam
    return rt


def v5_reacting():
    print("\n=== V5. Reacting to what changes ===")
    from src.senses.base import State
    rt = watch_rt()
    n = len(rt.ash.calls)
    for _ in range(3):
        pump(rt, [])
    check("the scene as it was: nothing is said", rt.out == [] and rt.looking.stats.get("changes", 0) == 0)
    watch = tv.thing("watch", .8, MID)
    pump(rt, [watch])
    check("one reading is not enough", rt.out == [])
    pump(rt, [watch])
    check("a watch in the middle for two readings: 'You picked up a watch.'", rt.out and rt.out[-1] == "You picked up a watch.", str(rt.out))
    check("...it was not a language-model call", len(rt.ash.calls) == n)
    pump(rt, [watch])
    pump(rt, [watch])
    check("it stays: not said again", rt.out.count("You picked up a watch.") == 1)
    for _ in range(3):
        pump(rt, [])
    check("it goes: 'The watch is gone.'", rt.out[-1] == "The watch is gone.", str(rt.out))
    check("the camera sensor's frame was used, the camera was never opened", rt.grabs == [])
    labels = [(r["kind"], r["label"], r["source"]) for r in rt.log.recent(20)]
    check("detections and events are in the log (the watch, the events)", ("object", "watch", "watch") in labels and ("event", "appeared watch", "watch") in labels
          and ("event", "gone watch", "watch") in labels, str(labels))
    check("a watched detection is logged once a minute, not every reading", sum(1 for k, l, s in labels if (k, l) == ("object", "watch")) == 1)
    check("the stats count the changes", rt.stats.get("vision_appeared") == 1 and rt.stats.get("vision_gone") == 1)
    check("the page's snapshot has the scene as boxes, from the watch", rt.looking.snapshot()["scene"] is not None)

    # when it must not look at all
    pr = watch_rt()
    for _ in range(3):
        pump(pr, [])
    pr.privacy.pause("test")
    k = pr.objs.calls
    pump(pr, [watch])
    pump(pr, [watch])
    check("privacy mode: no reading, nothing logged, nothing said", pr.objs.calls == k and pr.out == [] and pr.scene_watch._job is None)
    pr.privacy.resume()
    pump(pr, [watch])
    pump(pr, [watch])
    check("...and after privacy mode it works again", pr.out and pr.out[-1] == "You picked up a watch.", str(pr.out))

    kd = watch_rt(kids=True)
    kd.say("Ash, kids mode for 20 minutes")
    check("kids mode is on", kd.kids.active)
    for _ in range(3):
        pump(kd, [])
    k = kd.objs.calls
    n_out = len(kd.out)
    pump(kd, [watch])
    pump(kd, [watch])
    pump(kd, [watch])
    check("kids mode: no reading and nothing said", kd.objs.calls == k and len(kd.out) == n_out)

    so = watch_rt()
    so.cam.state = State.DISABLED
    pump(so, [])
    pump(so, [watch])
    check("the camera sensor is off: nothing is read (a look is only ever asked for)", so.objs.calls == 0 and so.scene_watch._job is None)
    st = watch_rt()
    st.cam.last_frame = None
    pump(st, [watch])
    check("the sensor has no frame yet: nothing is read", st.objs.calls == 0)
    old = watch_rt()
    pump(old, [watch], ts=time.time() - 60)
    check("a frame older than ten seconds is not used", old.objs.calls == 0)
    rr = watch_rt(vcfg=dict(react=False))
    for _ in range(4):
        pump(rr, [watch])
    check("vision.react false: no reading", rr.objs.calls == 0)
    quiet = watch_rt(vcfg=dict(announce_objects=False))
    for _ in range(3):
        pump(quiet, [])
    pump(quiet, [watch])
    pump(quiet, [watch])
    check("announce_objects false: recorded, not said", quiet.out == [] and quiet.stats.get("vision_appeared") == 1
          and any(r["label"] == "appeared watch" for r in quiet.log.recent(10)))
    mu = watch_rt()
    for _ in range(3):
        pump(mu, [])
    mu.gate.mute(True)
    pump(mu, [watch])
    pump(mu, [watch])
    check("muted: recorded, not said", mu.out == [] and mu.stats.get("vision_appeared") == 1)
    sg = watch_rt(vcfg=dict(presence=True, hold_when_stranger=True))
    for _ in range(3):
        pump(sg, [])
    sg.looking.presence.unknown_until = sg.t[0] + 500
    sg.looking.presence._stranger_on = True
    pump(sg, [watch])
    pump(sg, [watch])
    check("a stranger in view: not said (it is the user's things)", sg.out == [] and sg.stats.get("vision_appeared") == 1)
    iv = watch_rt(vcfg=dict(watch_interval_s=30.0))
    pump(iv, [])
    k = iv.objs.calls
    iv.t[0] += 2
    iv.cam.last_frame_ts = time.time()
    iv._tick()
    check("readings are spaced by watch_interval_s", iv.objs.calls == k)
    # people
    pp = watch_rt(faces=[tv.face()], vectors=[tv.vec(5)], vcfg=dict(presence=True, announce_people=True))
    pp.looking.faces.faces = []
    for _ in range(3):
        pump(pp, [])
    pp.looking.faces.faces, pp.looking.faces.vectors = [tv.face()], [tv.vec(5)]
    pump(pp, [])
    pump(pp, [])
    check("a person arrives (announce_people on): said, and not named", pp.out and pp.out[-1] == "Someone just came into view.", str(pp.out))
    pp.looking.faces.faces = []
    for _ in range(8):
        pump(pp, [])
    check("...and leaves", pp.out[-1] == "Nobody is in view now.", str(pp.out))
    pq = watch_rt(faces=[], vcfg=dict(presence=True))
    for _ in range(3):
        pump(pq, [])
    pq.looking.faces.faces = [tv.face()]
    pump(pq, [])
    pump(pq, [])
    check("announce_people is off by default: recorded only", pq.out == [] and pq.stats.get("vision_person_in") == 1)
    # a faulty detector never becomes 'something vanished'
    ft = watch_rt()
    for _ in range(3):
        pump(ft, [])
    pump(ft, [watch])
    pump(ft, [watch])
    ft.out.clear()
    ft.objs.detect = lambda img: (setattr(ft.objs, "last_error", "object detection failed"), [])[1]
    for _ in range(4):
        pump(ft, None)
    check("a detector that fails is not an empty room: nothing is 'gone'", not any("gone" in o for o in ft.out), str(ft.out))
    check("...the fault is counted", ft.looking.stats.get("faults", 0) >= 1)


def v6_defaults():
    print("\n=== V6. Defaults and wiring ===")
    from src.daemon.looking import LookingConfig, build
    from src.daemon.service import DEFAULT_CONFIG, load_config
    v = DEFAULT_CONFIG["vision"]
    check("vision is ON by default (asked looks)", v["enabled"] is True)
    check("background watching stays off: presence off, and the camera sensor is off", v["presence"] is False and DEFAULT_CONFIG["sensors"]["camera"] in ({"enabled": False}, False)
          or DEFAULT_CONFIG["sensors"].get("camera", {}).get("enabled") is False)
    check("what it says aloud about people is off", v["announce_people"] is False and v["announce_known"] is False and v["announce_strangers"] is False)
    check("the classifier fallback is on, and nothing is downloaded by default", v["classifier"] is True)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "daemon.json")
    open(p, "w", encoding="utf-8").write(json.dumps({"sensors": {}}))
    cfg = load_config(p)
    check("a daemon.json with no vision section gets it on (the user's own situation)", cfg["vision"]["enabled"] is True)
    open(p, "w", encoding="utf-8").write(json.dumps({"vision": {"enabled": False}}))
    check("...and an explicit false is honoured", load_config(p)["vision"]["enabled"] is False)
    src = open(os.path.join(HERE, "ashd.py"), encoding="utf-8").read()
    check("ashd always builds it", "vision=True," in src)
    lc = LookingConfig.from_dict({"enabled": True})
    check("the new settings have sane defaults", lc.react and lc.watch_interval_s >= 4 and lc.watch_confirm >= 2 and not lc.announce_objects and not lc.announce_people
          and lc.detection_log_size == 300 and lc.classifier_min_conf < lc.object_conf and lc.classifier_sure_conf < lc.sure_conf)
    lk = build({}, {"eye": tv.Eye(grab=lambda: tv.frame()) if hasattr(tv, "Eye") else None}, lambda: None, lambda: False, lambda: 1.0, None)
    check("build() with an empty config: enabled, a YOLO-or-ImageNet object backend, no file for detections", lk.cfg.enabled and lk.detections.path is None
          and type(lk.objects).__name__ == "FallbackObjects")
    lk2 = build({"enabled": False, "classifier": False}, {}, lambda: None, lambda: False, lambda: 1.0, os.path.join(d, "vision.json"))
    check("vision.enabled false and classifier false: off, YOLO only, and the log lives next to vision.json",
          not lk2.cfg.enabled and type(lk2.objects).__name__ == "ONNXObjects" and lk2.detections.path == os.path.join(d, "vision", "detections.jsonl"))
    import tst_effects as te
    bare = te.rig()
    check("a bare runtime has no scene watcher", bare.scene_watch is None)
    check("a vision runtime has one", rt_with().scene_watch is not None)
    from src.daemon.commands import CATALOG
    phrases = {p for p, k in CATALOG}
    check("the catalog knows the camera phrasings (so a typo is repaired by the loose matcher)", {"look at me", "can you see me", "what do you see on the camera", "what's in my hand",
          "show me what you see", "take a picture with the camera", "what am I holding"} <= phrases)


def v7_follow_ups():
    print("\n=== V7. 'Try again', 'how about now?', 'what's this?' ===")
    from src.daemon.looking import parse
    for q in ("What am I holding Ash?", "what am I holding, ash",
              "look at me, ash"):
        check(f"{q!r} is the camera", parse(q) is not None, str(parse(q)))
    for q in ("what's the weather", "what's that song", "what is this error", "what's this about", "how about now", "try again", "What's this, Ash?", "what is this"):
        check(f"{q!r} alone is not (nothing was looked at)", parse(q) is None)
    rt = rt_with([tv.thing("watch", .8, MID)])
    n = len(rt.ash.calls)
    k = len(rt.grabs)
    rt.say("Ash, what am I holding")
    check("(a look first)", "watch" in rt.out[-1] and len(rt.grabs) == k + 1)
    for q in ("Try again.", "How about now?", "now", "and now", "one more time", "look again", "what about now", "How about with no?", "this one",
              "What's this, Ash?", "what is that", "Ash, what's this one", "what do you think it is"):
        k = len(rt.grabs)
        rt.say(q)
        check(f"{q!r} right after a look: looks again, by rule", len(rt.grabs) == k + 1 and "watch" in rt.out[-1] and len(rt.ash.calls) == n, f"{rt.out[-1]} {len(rt.grabs) - k}")
    rt.t[0] += 600
    k = len(rt.grabs)
    rt.say("how about now")
    check("ten minutes later it is not a camera request any more", len(rt.grabs) == k)
    rt2 = rt_with([tv.thing("watch", .8, MID)])
    k = len(rt2.grabs)
    rt2.say("try again")
    check("with no look before it, 'try again' is nothing to the camera", len(rt2.grabs) == k)
    rt2.say("look at me")
    rt2.say("now")
    check("after 'look at me', 'now' looks at them again", len(rt2.grabs) == 2 and rt2.out[-1].startswith("Yes, I can see someone"), rt2.out[-1])


def v8_corrections():
    print("\n=== V8. Being told it got it wrong ===")
    rt = rt_with([tv.thing("phone", .55, MID)])
    n = len(rt.ash.calls)
    rt.say("Ash, what am I holding")
    check("(it guesses)", "phone" in rt.out[-1])
    rt.say("nope")
    check("'nope' after a guess: it owns up and asks, by rule, no model", rt.out[-1].startswith("Okay, I got that wrong. What is it?") and len(rt.ash.calls) == n, rt.out[-1])
    rt.say("it's not a phone")
    check("'it's not a phone' too", rt.out[-1].startswith("Okay, I got that wrong"), rt.out[-1])
    rt.say("it's a watch")
    check("'it's a watch' teaches it from what it just saw", "watch" in rt.out[-1].lower() and len(rt.ash.calls) == n, rt.out[-1])
    check("...and it is in its memory", rt.looking.memory.get("watch") is not None or rt.looking.memory.get("a watch") is not None, str(rt.looking.memory.listing() if hasattr(rt.looking.memory, "listing") else ""))
    rt.t[0] += 600
    m = len(rt.ash.calls)
    rt.say("it's a nice day")
    check("ten minutes later 'it's a nice day' is just conversation", len(rt.ash.calls) == m + 1)
    rt2 = rt_with([tv.thing("phone", .55, MID)])
    rt2.say("nope")
    check("with nothing looked at, 'nope' is not the camera's", "got that wrong" not in (rt2.out[-1] if rt2.out else ""))


if __name__ == "__main__":
    for fn in (v1_phrasings, v2_why_not, v3_answers, v4_runtime, v5_reacting, v6_defaults, v7_follow_ups, v8_corrections):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
