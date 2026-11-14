"""
tst_vision_scene.py -- naming things without a download, and noticing what changes. No camera, no model files, no network:
frames are synthetic arrays and the classifier is a scripted fake.

  S1  ImageNet's verbose names in plain words; worn things read as a person
  S2  the held thing: nearest the middle, else the biggest; never a person or furniture
  S3  the classifier backend: the crops it looks at, what it makes of the answers, the bars, the way it degrades
  S4  the YOLO file is preferred when there is one
  S5  honest wording: "it's a watch" only when sure, "I think it's a watch" below the bar; the bar is the backend's own
  S6  the detection log: ring of ~300, jsonl with a cap, labels and numbers only
  S7  change watching: picked up, put down, flicker, warm-up, cool-down, people

Run:  PYTHONIOENCODING=utf-8 python tests/tst_vision_scene.py
"""

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_vision as tv  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check


def scripted(table, calls=None):
    """A fake classifier: crop i of a call gets table[i % len(table)] -- a list of (raw label, probability)."""
    def classify(crops):
        if calls is not None:
            calls.append(len(crops))
        return [list(table[i % len(table)]) for i in range(len(crops))]
    return classify


def s1_plain_words():
    print("\n=== S1. Plain words ===")
    from src.vision.scene import PLAIN, is_worn, plain_label
    for raw, want in [("digital watch", "watch"), ("analog clock", "clock"), ("wall clock", "clock"),
                      ("cellular telephone, cellular phone, cellphone, cell, mobile phone", "phone"),
                      ("coffee mug", "mug"), ("pop bottle, soda bottle", "bottle"), ("water bottle", "water bottle"),
                      ("laptop, laptop computer", "laptop"), ("notebook, notebook computer", "laptop"),
                      ("computer keyboard, keypad", "keyboard"), ("mouse, computer mouse", "mouse"),
                      ("remote control, remote", "remote"), ("ballpoint, ballpoint pen, ballpen, Biro", "pen"),
                      ("sunglasses, dark glasses, shades", "sunglasses"), ("stopwatch, stop watch", "stopwatch"),
                      ("Windsor tie", "tie"), ("teddy, teddy bear", "teddy bear"), ("reflex camera", "camera"),
                      ("loupe, jeweler's loupe", "magnifying glass"), ("book_jacket", "book"),
                      ("tabby, tabby cat", "tabby"), ("Granny Smith", "granny smith"), ("", "")]:
        check(f"{raw!r} -> {want!r}", plain_label(raw) == want, plain_label(raw))
    check("a name the table does not know is its first synonym, lower case", plain_label("Arabian camel, dromedary, Camelus dromedarius") == "arabian camel")
    check("underscores read as spaces", plain_label("wooden_spoon") == "wooden spoon")
    check("jersey / suit / mask are worn, a mug is not", is_worn("jersey, T-shirt, tee shirt") and is_worn("suit, suit of clothes") and is_worn("mask")
          and not is_worn("coffee mug") and not is_worn("digital watch"))
    check("every plain word is lower case and short", all(v == v.lower() and len(v) < 30 for v in PLAIN.values()))


def s2_held():
    print("\n=== S2. The held thing ===")
    from src.vision.scene import in_centre, pick_held
    T = tv.thing
    W, H = 640, 480
    check("nothing -> None", pick_held([], W, H) is None)
    check("only a person -> None", pick_held([T("person", .9, (100, 50, 400, 400))], W, H) is None)
    check("only furniture (skipped) -> None", pick_held([T("laptop", .9, (250, 190, 140, 100))], W, H, skip={"laptop"}) is None)
    near = T("bottle", .5, (290, 180, 70, 120))
    far = T("cup", .95, (10, 10, 60, 60))
    check("the one nearest the middle wins over a more confident one at the edge", pick_held([far, near], W, H) is near)
    a, b = T("cup", .6, (270, 200, 100, 80)), T("book", .6, (330, 220, 100, 80))
    check("two near the middle: the more central", pick_held([a, b], W, H) is a)
    check("equal distance: confidence breaks the tie", pick_held([T("cup", .4, (270, 200, 100, 80)), T("book", .9, (270, 200, 100, 80))], W, H).label == "book")
    c1, c2 = T("cup", .6, (5, 5, 50, 50)), T("book", .5, (0, 300, 260, 180))
    check("nothing near the middle: the biggest box, wherever it is", pick_held([c1, c2], W, H) is c2)
    check("a person in the middle is ignored for the thing beside it", pick_held([T("person", .99, (0, 0, 640, 480)), near], W, H) is near)
    tile = T("watch", .4, (0, 0, 320, 240), )
    centre = T("watch", .4, (128, 72, 384, 336))
    check("a whole-frame-ish centre crop beats a corner tile", pick_held([tile, centre], W, H) is centre)
    check("in_centre: the middle is, a corner is not", in_centre((290, 210, 60, 60), W, H) and not in_centre((0, 0, 60, 60), W, H))
    check("in_centre respects the fraction", in_centre((400, 210, 60, 60), W, H, 0.3) and not in_centre((400, 210, 60, 60), W, H, 0.1))


def s3_classifier():
    print("\n=== S3. The classifier backend ===")
    from src.vision.classify import ClassifierObjects, crops_of
    f = tv.frame(size=(480, 640))
    cs = crops_of(f)
    check("one centre crop and nine tiles", len(cs) == 10 and cs[0][2] == "centre" and sum(1 for c in cs if c[2] == "tile") == 9)
    check("the centre crop is the embedder's own box (a thing taught by it is found by it)", cs[0][1] == (128, 72, 384, 336))
    check("every tile is half the frame and they overlap, covering all of it", all(c[1][2:] == (320, 240) for c in cs[1:])
          and {c[1][:2] for c in cs[1:]} == {(x, y) for x in (0, 160, 320) for y in (0, 120, 240)})
    check("every crop is the size its box says", all(c[0].shape[:2] == (c[1][3], c[1][2]) for c in cs))
    check("the light version is the middle only", len(crops_of(f, tiles=False)) == 1)
    check("a tiny frame gives no crops, not a crash", crops_of(np.zeros((8, 8, 3), np.uint8)) == [])

    calls = []
    watch = [[("digital watch", 0.72), ("analog clock", 0.1)]] + [[("desk", 0.05)]] * 9
    c = ClassifierObjects(scripted(watch, calls))
    got = c.detect(f)
    check("a watch in the middle crop is a Thing named in plain words", got and got[0].label == "watch" and abs(got[0].score - 0.72) < 1e-6, str(got))
    check("...with the centre crop's box, from the centre", got[0].box == (128, 72, 384, 336) and got[0].src == "centre")
    check("all ten crops were classified in ONE batch", calls == [10])
    check("what is below the floor is not reported", all(t.score >= 0.15 for t in got) and not any(t.label == "desk" for t in got))
    check("the second guess is the middle's only, and only above the floor", not any(t.label == "clock" for t in got))
    check("the light call classifies one crop", (calls.clear(), c.detect_light(f), calls == [1])[2])

    rep = ClassifierObjects(scripted([[("coffee mug", 0.4)]]))
    r = rep.detect(f)
    check("the same label from many tiles is ONE thing (not '10 mugs')", len(r) == 1 and r[0].label == "mug")
    check("the centre wins a near tie, because that is where a held thing is", r[0].src == "centre", str(r))

    mixed = ClassifierObjects(scripted([[("jersey, T-shirt, tee shirt", 0.8)], [("coffee mug", 0.5)]]))
    r = {t.label: t for t in mixed.detect(f)}
    check("a worn thing is reported as a person, discounted (ImageNet has no person class)", "person" in r and abs(r["person"].score - 0.64) < 1e-6, str(r))
    check("...and the mug beside it is still a mug", "mug" in r)

    floor = ClassifierObjects(scripted([[("digital watch", 0.29)]]))
    check("just below the floor (0.29 < 0.30): nothing", floor.detect(f) == [])
    edge = ClassifierObjects(scripted([[("digital watch", 0.30)]]))
    check("exactly the floor: reported", [t.label for t in edge.detect(f)] == ["watch"])
    lo = ClassifierObjects(scripted([[("digital watch", 0.3)]]), min_conf=0.5)
    check("the floor is configurable", lo.detect(f) == [])

    check("sorted most sure first", [t.score for t in ClassifierObjects(scripted([[("digital watch", .3)], [("coffee mug", .6)], [("water bottle", .45)]])).detect(f)] == sorted(
        [t.score for t in ClassifierObjects(scripted([[("digital watch", .3)], [("coffee mug", .6)], [("water bottle", .45)]])).detect(f)], reverse=True))
    check("a None frame is nothing, not a crash", c.detect(None) == [])
    check("an injected classifier is available, with no model", c.available() and c.why_not() == "")

    def boom(crops):
        raise RuntimeError("model blew up")
    bad = ClassifierObjects(boom)
    check("a classifier that raises: no things, and it says it broke (not 'nothing there')", bad.detect(f) == [] and bad.last_error == "object detection failed")
    check("...and a good call clears the flag", (setattr(c, "last_error", "x"), c.detect(f), c.last_error)[2] is None)

    missing = ClassifierObjects(None, name="no-such/model-anywhere", clock=lambda: 0.0)
    check("no cached model: not available, with the reason", not missing.available() and "not in the local Hugging Face cache" in missing.why_not()
          and "never downloaded" in missing.why_not(), missing.why_not())
    check("...and detect returns nothing without trying to load", missing.detect(f) == [] and missing.calls == 0)
    check("it checks the local cache only (local_files_only), by reading the source",
          "local_files_only=True" in open(os.path.join(HERE, "src", "vision", "classify.py"), encoding="utf-8").read())


def s4_fallback():
    print("\n=== S4. YOLO when there is one ===")
    from src.vision.classify import ClassifierObjects, FallbackObjects
    f = tv.frame(size=(480, 640))
    yolo = tv.FakeObjects([tv.thing("cup", 0.9)])
    net = ClassifierObjects(scripted([[("digital watch", 0.7)]]))
    fb = FallbackObjects(yolo, net)
    check("the detector file's answer is used when it is there", [t.label for t in fb.detect(f)] == ["cup"] and fb.kind == "yolo")
    check("...and the classifier is not touched", net.calls == 0)
    check("YOLO's bars are the config's (the backend names none)", fb.min_conf is None and fb.sure_conf is None)
    yolo._ok = False
    check("no detector file: the classifier answers", [t.label for t in fb.detect(f)] == ["watch"] and fb.kind == "imagenet")
    check("...with its own bars", fb.min_conf == 0.30 and fb.sure_conf == 0.55)
    check("detect_light goes to the classifier's cheap call", len(fb.detect_light(f)) == 1)
    check("available while either is", fb.available())
    net2 = ClassifierObjects(None, name="none/none", clock=lambda: 0.0)
    both = FallbackObjects(tv.FakeObjects(available=False), net2)
    check("neither: unavailable, kind none, and why", not both.available() and both.kind == "none" and "cache" in both.why_not())
    fb.last_error = "x"
    check("last_error is shared both ways", fb.last_error == "x" and (setattr(fb, "last_error", None), fb.last_error)[1] is None)


def s5_wording():
    print("\n=== S5. Honest wording and the bars ===")
    from src.vision.classify import ClassifierObjects, FallbackObjects

    def lk(p, table=None, **kw):
        net = ClassifierObjects(scripted(table or [[("digital watch", p)]] + [[("desk", 0.01)]] * 9), **kw)
        return tv.looking(objects=net)
    check("sure (0.72): 'It's a watch.'", lk(0.72).hear("what am I holding") == "It's a watch.")
    check("exactly the bar (0.55): sure", lk(0.55).hear("what am I holding") == "It's a watch.")
    check("just under (0.54): 'I think it's a watch'", lk(0.54).hear("what am I holding") == "I think it's a watch, but I'm not sure.")
    check("a low-ish classifier probability is still hedged, not dropped (0.32)", lk(0.32).hear("what am I holding") == "I think it's a watch, but I'm not sure.")
    check("below the classifier's floor: nothing to say", "closer to the camera" in lk(0.1).hear("what am I holding"))
    check("the YOLO bars do not apply (0.4 is below YOLO's 0.6 'sure' but above the classifier's floor)", lk(0.4).hear("look at this") != "It's a watch.")
    check("an 'a' / 'an' that reads right", lk(0.8, [[("analog clock", 0.8)]] + [[("desk", .01)]] * 9).hear("what am I holding") == "It's a clock."
          and lk(0.8, [[("iPod", 0.8)]] + [[("desk", .01)]] * 9).hear("what am I holding") == "It's a music player.")
    check("'look around' with the classifier says what is in view, hedged when unsure",
          lk(0.3).hear("look around") == "I see a watch -- though I'm not very sure." and lk(0.8).hear("look around") == "I see a watch.")
    yolo = tv.looking(objects=FallbackObjects(tv.FakeObjects([tv.thing("cup", 0.45, (290, 200, 60, 80))]), ClassifierObjects(scripted([[("x", 0.9)]]))))
    check("with YOLO the config's bars are used (0.45 is below 0.6)", yolo.hear("look at this") == "I think it's a cup, but I'm not sure.")
    check("thing_threshold is the embedder's, not the detector's: a classifier does not change it",
          tv.looking(objects=ClassifierObjects(scripted([[("x", .9)]]))).cfg.thing_threshold == 0.88)
    # a taught thing is still found by the box the classifier uses
    t = tv.looking(objects=ClassifierObjects(scripted([[("coffee mug", 0.6)]] + [[("desk", .01)]] * 9)))
    f = tv.frame(square=((30, 90, 200), (200, 160, 240, 190)), noise=4)
    t.eye._grab = lambda: f
    check("teaching by showing works with the classifier's centre box", "Okay, I'll recognise my blue mug (it looks like a mug)" in t.hear("remember this as my blue mug"))
    check("...and it is named back by the same box", t.hear("what am I holding").startswith("That's your blue mug"), t.hear("what am I holding"))


def s6_log():
    print("\n=== S6. The detection log ===")
    from src.vision.scene import DetectionLog
    d = tempfile.mkdtemp()
    path = os.path.join(d, "vision", "detections.jsonl")
    t = [1000.0]
    log = DetectionLog(path, size=300, max_lines=100, clock=lambda: t[0])
    rec = log.add("object", "watch", 0.7234567, (1, 2, 3, 4), "asked")
    check("a record is time, kind, label, confidence, box, source -- nothing else", sorted(rec) == ["box", "conf", "kind", "label", "source", "t"]
          and rec["conf"] == 0.723 and rec["box"] == [1, 2, 3, 4])
    line = open(path, encoding="utf-8").read().splitlines()
    check("it was appended to the file, one JSON line", len(line) == 1 and json.loads(line[0]) == rec)
    for i in range(400):
        t[0] += 1
        log.add("object", f"thing{i % 7}", 0.5, (0, 0, 10, 10), "asked")
    check("the ring keeps the last 300", len(log) == 300 and log.recent(1)[0]["label"] == "thing0" and log.total == 401)
    check("recent is newest first", [r["t"] for r in log.recent(5)] == sorted([r["t"] for r in log.recent(5)], reverse=True))
    n = len(open(path, encoding="utf-8").read().splitlines())
    check("the file is capped (100 lines, trimmed when 25% over)", n <= 125, str(n))
    check("...keeping the newest", json.loads(open(path, encoding="utf-8").read().splitlines()[-1])["t"] == t[0])
    again = DetectionLog(path, size=300, max_lines=100, clock=lambda: t[0])
    check("a restart reloads what the file has", 0 < len(again) <= n and again.recent(1)[0]["t"] == t[0])
    old = DetectionLog(os.path.join(d, "old.jsonl"), max_lines=10, max_age_s=100.0, clock=lambda: t[0])
    for i in range(20):
        old.add("object", "x", .5, None, "asked", now=t[0] - 1000 + i)
    for i in range(10):
        old.add("object", "y", .5, None, "asked", now=t[0] + i)
    rows = [json.loads(l) for l in open(os.path.join(d, "old.jsonl"), encoding="utf-8")]
    check("a trim also drops what is older than the age cap", all(r["label"] == "y" for r in rows) and rows, str(len(rows)))
    blob = open(path, encoding="utf-8").read()
    check("no pixels in the file: it is JSON text of numbers and labels", blob.isascii() and "\x00" not in blob and all(json.loads(l) for l in blob.splitlines()))
    log.add("object", "w" * 500, 7.0, (1, 2), "x" * 50, 5.0)
    last = log.recent(1)[0]
    check("a long label is cut, a confidence is clamped to 0..1, a short box is None", len(last["label"]) == 60 and last["conf"] == 1.0 and last["box"] is None and len(last["source"]) <= 12)
    c = DetectionLog(None)
    for lab, conf in (("cup", .4), ("cup", .6), ("watch", .9)):
        c.add("object", lab, conf, None, "asked")
    c.add("event", "appeared watch", 1.0, None, "watch")
    rows = {r["label"]: r for r in c.counts()}
    check("per-label counts and average confidence (events are not counted)", rows["cup"]["count"] == 2 and rows["cup"]["avg"] == 0.5
          and rows["watch"]["count"] == 1 and "appeared watch" not in rows)
    check("most often first", c.counts()[0]["label"] == "cup")
    check("no path: memory only, nothing on disk", DetectionLog(None).add("object", "a", .5, None, "asked") and True)
    c.clear()
    check("clear empties it, and deletes the file", len(c) == 0 and (log.clear(), not os.path.exists(path))[1])
    bad = DetectionLog(os.path.join(d, "nul\0bad", "x.jsonl"))
    check("an unwritable path does not break adding", bad.add("object", "a", .5, None, "asked")["label"] == "a" and len(bad) == 1)
    # through the Looking object: only labels and numbers, with the log injected
    lg = DetectionLog(None)
    lk = tv.looking([tv.thing("cup", 0.8), tv.thing("person", 0.9, (0, 0, 640, 480))], faces=[tv.face()], vectors=[tv.vec(3)])
    lk.detections = lg
    lk.hear("look around")
    kinds = sorted((r["kind"], r["label"]) for r in lg.recent(20))
    check("an asked look logs each thing, the person and the face (as 'face', never a name)", kinds == [("face", "face"), ("object", "cup"), ("person", "person")], str(kinds))
    check("...and the last scene is boxes, no pixels", lk.last_scene and set(lk.last_scene) == {"t", "w", "h", "source", "things", "faces"}
          and lk.last_scene["w"] == 640 and not any(isinstance(v, np.ndarray) for v in lk.last_scene.values()))
    check("the scene is JSON-clean", json.dumps(lk.last_scene) and json.dumps(lk.snapshot()))


def s7_changes():
    print("\n=== S7. Change watching ===")
    from src.vision.scene import ChangeWatcher, event_line
    T = tv.thing
    mid = (280, 200, 80, 80)
    corner = (0, 0, 60, 60)
    shape = (480, 640)
    w = ChangeWatcher(confirm=3, miss=3, warmup=3, cooldown_s=120.0)
    ev = []
    t = 0.0

    def feed(things, people=0):
        nonlocal t
        t += 8
        got = w.observe(things, people, shape, t)
        ev.extend(got)
        return got
    for _ in range(3):
        feed([])
    check("warm-up readings are the scene as it was: nothing is news", ev == [])
    check("one reading of a watch is not enough", feed([T("watch", .7, mid)]) == [] and feed([T("watch", .7, mid)]) == [])
    check("the third in a row: 'appeared'", feed([T("watch", .7, mid)]) == [("appeared", "watch", 0.7)])
    check("it stays: said once", feed([T("watch", .7, mid)]) == [] and feed([T("watch", .7, mid)]) == [])
    check("one missed reading is not 'gone'", feed([]) == [] and feed([T("watch", .7, mid)]) == [])
    check("three missed readings in a row: 'gone'", feed([]) == [] and feed([]) == [] and feed([]) == [("gone", "watch", 0.7)])
    check("...once", feed([]) == [])
    w2 = ChangeWatcher(confirm=3, miss=3, warmup=3, cooldown_s=120.0)
    t = 0.0
    for _ in range(3):
        w2.observe([T("lamp", .8, mid)], 0, shape, (t := t + 8))
    check("what was there from the start is not an event, and its leaving is not announced",
          w2.observe([], 0, shape, (t := t + 8)) == [] and w2.observe([], 0, shape, (t := t + 8)) == [] and w2.observe([], 0, shape, (t := t + 8)) == [])
    w3 = ChangeWatcher(confirm=3, miss=3, warmup=0, cooldown_s=120.0, min_conf=0.3)
    t = 0.0
    seq = [[T("mug", .29, mid)]] * 5
    check("below the watch bar (0.29 < 0.30) nothing is claimed", all(w3.observe(s, 0, shape, (t := t + 8)) == [] for s in seq))
    check("the bar itself (0.30) counts", [w3.observe([T("mug", .3, mid)], 0, shape, (t := t + 8)) for _ in range(3)][-1] == [("appeared", "mug", 0.3)])
    w4 = ChangeWatcher(confirm=2, miss=2, warmup=0)
    t = 0.0
    check("something at the edge of the frame is not 'picked up' (the middle only)", all(w4.observe([T("mug", .9, corner)], 0, shape, (t := t + 8)) == [] for _ in range(4)))
    check("furniture and persons are never things picked up", all(ChangeWatcher(confirm=1, warmup=0, skip={"laptop"}).observe([T("laptop", .9, mid), T("person", .9, mid)], 0, shape, 1.0) == [] for _ in range(2)))
    w5 = ChangeWatcher(confirm=2, miss=2, warmup=0, cooldown_s=100.0)
    t = 0.0
    out = []
    for step in range(60):                        # a thing flickering in and out
        out += w5.observe([T("mug", .8, mid)] if (step // 3) % 2 == 0 else [], 0, shape, (t := t + 8))
    ups = [e for e in out if e[0] == "appeared"]
    check("a flickering thing is said once per cool-down (100 s = 12 readings), not every time", 1 <= len(ups) <= 5, str(len(ups)))
    check("the label cool-down is per label", len({e[1] for e in out}) == 1)
    w6 = ChangeWatcher(confirm=2, miss=2, warmup=0, cooldown_s=0.0)
    t = 0.0
    both = []
    for _ in range(2):
        both += w6.observe([T("mug", .8, mid), T("watch", .8, (300, 220, 40, 40))], 0, shape, (t := t + 8))
    check("two things at once: both", sorted(e[1] for e in both) == ["mug", "watch"])
    # people
    p = ChangeWatcher(confirm=2, miss=2, warmup=2, person_cooldown_s=300.0)
    t = 0.0
    r = []
    for people in (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1):
        r += p.observe([], people, shape, (t := t + 8))
    check("someone there from the start is the scene as it was", all(e[0] != "person_in" or e[2] for e in r) and r.count(("person_in", "", 1.0)) <= 1)
    check("a person who leaves: said after enough empty readings", ("person_out", "", 1.0) in r, str(r))
    q = ChangeWatcher(confirm=2, miss=2, warmup=0, person_cooldown_s=300.0)
    t = 0.0
    seq = []
    for people in (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1):
        seq += q.observe([], people, shape, (t := t + 8))
    check("arrive, leave, arrive again: the second arrival is inside the cool-down so not repeated", seq.count(("person_in", "", 1.0)) == 1 and seq.count(("person_out", "", 1.0)) == 1, str(seq))
    q.reset()
    check("reset forgets everything", q.readings == 0 and not q.present)
    check("a person is counted, never named: the event carries no name", all(len(e) == 3 and e[1] == "" for e in seq))
    check("lines: sure", event_line("appeared", "watch", .8) == "You picked up a watch." and event_line("gone", "watch", .8) == "The watch is gone.")
    check("lines: unsure is hedged", event_line("appeared", "watch", .3) == "Is that a watch you picked up? I'm not sure."
          and event_line("gone", "watch", .3) == "I don't see the watch any more.")
    check("lines: an 'an' and people", event_line("appeared", "umbrella", .9) == "You picked up an umbrella."
          and event_line("person_in", "", 1) == "Someone just came into view." and event_line("person_out", "", 1) == "Nobody is in view now." and event_line("nonsense", "x", 1) == "")


if __name__ == "__main__":
    for fn in (s1_plain_words, s2_held, s3_classifier, s4_fallback, s5_wording, s6_log, s7_changes):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
