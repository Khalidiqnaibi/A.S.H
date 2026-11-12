"""
tst_vision.py -- recognising people and things through the camera, with no real camera and no model files.

Every section is one step and was added with it. The camera is a fake that hands back synthetic frames, and the model
backends are fakes too: what is tested is what ASH does with what they say. Nothing here opens a real camera.

Run:  PYTHONIOENCODING=utf-8 python tst_vision.py
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
SECTIONS = []


def section(fn):
    SECTIONS.append(fn)
    return fn


def vec(seed, dim=128, noise=0.0, base=None):
    """A reproducible unit vector; `base` plus noise is 'the same thing seen again'."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32) if base is None else np.asarray(base, dtype=np.float32).copy()
    if noise:
        v = v + noise * np.random.RandomState(seed + 1000).randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@section
def step0_visual_memory():
    print("\n=== STEP 0: the visual memory -- labels and numbers, cautious matching ===")
    from src.vision.memory import VisualMemory, norm_label
    t = [1000.0]
    path = os.path.join(tempfile.mkdtemp(), "vision.json")
    m = VisualMemory(path, clock=lambda: t[0])
    omar, mug, keys = vec(1), vec(2), vec(3)
    check("labels are normalised: articles and case do not make another label", norm_label("My Blue Mug") == "blue mug"
          and norm_label("the  blue   mug!") == "blue mug" and norm_label("Omar") == "omar")
    v = m.enroll("Omar", "person", omar)
    check("a person is taught", v is not None and v.label == "Omar" and v.kind == "person" and v.seen == 1)
    m.enroll("my blue mug", "thing", mug)
    check("...and a thing", m.get("blue mug") is not None and m.status() == {"people": 1, "things": 1})
    check("the same label seen again adds a look, not a second item", m.enroll("omar", "person", vec(1, noise=0.1, base=omar)) is not None
          and len(m.list("person")) == 1 and len(m.get("Omar").embeds) == 2 and m.get("Omar").seen == 2)
    check("a person and a thing cannot share a label", m.enroll("Omar", "thing", vec(9)) is None)
    check("a bad label, a bad kind or an empty vector is refused", m.enroll("", "person", omar) is None
          and m.enroll("x", "animal", omar) is None and m.enroll("x", "thing", np.zeros(128)) is None
          and m.enroll("x", "thing", np.array([np.nan] * 128)) is None)

    got = m.match(vec(1, noise=0.15, base=omar), threshold=0.6)
    check("the same person a little differently: matched, with the score", got and got.label == "Omar" and got.score > 0.9, str(got))
    check("a stranger is not matched", m.match(vec(77), threshold=0.6) is None)
    check("a thing is not matched as a person when asked only for people", m.match(mug, kind="person", threshold=0.6) is None
          and m.match(mug, kind="thing", threshold=0.6).label == "my blue mug")
    check("below the threshold: not sure", m.match(vec(1, noise=2.0, base=omar), threshold=0.95) is None)

    twin = vec(5, base=omar, noise=0.02)                           # nearly the same face as Omar's
    m.enroll("Ali", "person", twin)
    check("two labels that look alike: not sure, rather than a guess", m.match(vec(1, noise=0.01, base=omar), kind="person", threshold=0.6, margin=0.05) is None)
    check("...though with the margin lowered it names the closer", m.match(vec(1, noise=0.01, base=omar), kind="person", threshold=0.6, margin=0.0) is not None)
    check("scores are listed, highest first", [s[0] for s in m.scores(omar, "person")][0] in ("Omar", "Ali"))

    t[0] = 2000.0
    m.seen_now("Omar", 2000.0)
    check("seeing someone updates when and how often", m.get("Omar").last_seen == 2000.0 and m.get("Omar").seen == 3)
    old = VisualMemory(path)
    check("it survives a restart", old.get("Omar") is not None and old.get("blue mug").kind == "thing" and len(old.get("Omar").embeds) == 2)
    for i in range(20):
        m.enroll("keys", "thing", vec(100 + i))
    check("a label keeps only its latest twelve looks", len(m.get("keys").embeds) == 12)

    out = os.path.join(tempfile.mkdtemp(), "x.json")
    n = m.export(out)
    import json
    data = json.load(open(out, encoding="utf-8"))
    check("the export is readable and has no numbers in it", n == 4 and "embeds" not in json.dumps(data) and data["items"][0]["label"])

    check("forgetting one label", m.forget("Omar") == 1 and m.get("Omar") is None and m.get("blue mug") is not None)
    check("...a kind", m.forget(kind="person") == 1 and m.list("person") == [])
    check("...and everything", m.forget() == 2 and m.list() == [] and VisualMemory(path).list() == [])
    check("forgetting what is not there says zero", m.forget("nobody") == 0)
    photo = os.path.join(tempfile.mkdtemp(), "p.jpg")
    open(photo, "wb").write(b"x")
    m.enroll("desk", "thing", vec(8), photo=photo)
    m.forget("desk")
    check("a kept photo is the user's file: forgetting removes the pointer, not the file", os.path.exists(photo))


def frame(color=(200, 200, 200), square=None, size=(480, 640), noise=0, seed=0):
    """A BGR test frame: a grey background and, if given, a filled square of `square=(bgr, (x, y, w, h))`."""
    f = np.full((size[0], size[1], 3), color, dtype=np.uint8)
    if square:
        c, (x, y, w, h) = square
        f[y:y + h, x:x + w] = c
    if noise:
        f = np.clip(f.astype(int) + np.random.RandomState(seed).randint(-noise, noise + 1, f.shape), 0, 255).astype(np.uint8)
    return f


@section
def step0_backends_and_installer():
    print("\n=== STEP 0: the models behind their interfaces, and an installer that asks first ===")
    from PIL import Image
    from src.vision import backends as bk, models

    red_rgb = Image.new("RGB", (4, 4), (255, 0, 0))
    check("a PIL image (RGB) becomes BGR", tuple(bk.as_bgr(red_rgb)[0, 0]) == (0, 0, 255))
    check("an array is taken to be BGR already; grey and float frames are handled", bk.as_bgr(np.zeros((4, 4), np.uint8)).shape == (4, 4, 3)
          and bk.as_bgr(np.ones((4, 4, 3), np.float32)).max() == 255 and bk.as_bgr(frame()).shape == (480, 640, 3))
    check("anything else is None, not a crash", bk.as_bgr(None) is None and bk.as_bgr("x") is None and bk.as_bgr(np.zeros((3,))) is None)

    # an object detector's raw output, with a known geometry: a 1280x720 frame is letterboxed to 640 (ratio 0.5, pad 0,140)
    canvas, ratio, pad = bk.letterbox(np.zeros((720, 1280, 3), np.uint8), 640)
    check("letterbox keeps the aspect ratio and says how to undo it", canvas.shape == (640, 640, 3) and ratio == 0.5 and pad == (0, 140))
    out = np.zeros((1, 84, 6), np.float32)

    def put(i, cx, cy, w, h, cls, score):
        out[0, 0:4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    put(0, 320, 320, 100, 100, 41, 0.90)         # a cup
    put(1, 322, 322, 100, 100, 41, 0.60)         # the same cup, found twice
    put(2, 100, 300, 80, 160, 0, 0.80)           # a person
    put(3, 500, 500, 50, 50, 63, 0.20)           # a laptop it is not sure of
    things = bk.parse_yolov8(out, ratio, pad, (720, 1280))
    check("duplicates are merged and the unsure one is dropped: a cup and a person, best first",
          [t.label for t in things] == ["cup", "person"] and abs(things[0].score - 0.9) < 1e-5, str(things))
    check("boxes are mapped back to the original frame (x, y, w, h)", things[0].box == (540, 260, 200, 200), str(things[0].box))
    check("the detector's [1, anchors, 4+classes] layout works too", [t.label for t in bk.parse_yolov8(out.transpose(0, 2, 1), ratio, pad, (720, 1280))] == ["cup", "person"])
    check("nothing over the threshold is nothing", bk.parse_yolov8(np.zeros((1, 84, 6), np.float32), ratio, pad, (720, 1280)) == [])
    check("a box off the edge is clipped to the frame", all(0 <= t.box[0] and t.box[0] + t.box[2] <= 1280 for t in bk.parse_yolov8(out, ratio, pad, (720, 1280))))
    check("80 class names, with spaces", len(bk.COCO_LABELS) == 80 and "cell phone" in bk.COCO_LABELS and "dining table" in bk.COCO_LABELS)

    # things: the light embedder tells colour and shape apart
    emb = bk.ThingEmbedder()
    blue = frame(square=((200, 80, 20), (220, 140, 200, 200)))
    blue2 = frame(square=((205, 85, 25), (225, 150, 190, 190)), noise=6, seed=3)
    red = frame(square=((30, 30, 210), (220, 140, 200, 200)))
    e1, e2, e3 = emb.embed(blue), emb.embed(blue2), emb.embed(red)
    check("it makes a unit vector of the documented size", e1.shape == (bk.ThingEmbedder.DIM,) and abs(float(np.linalg.norm(e1)) - 1) < 1e-5)
    check("the same blue thing, a little moved, resized and noisy, is close (0.97)", float(np.dot(e1, e2)) > 0.9, str(float(np.dot(e1, e2))))
    green = frame(square=((40, 200, 60), (220, 140, 200, 200)))
    black, white = frame(square=((20, 20, 20), (220, 140, 200, 200))), frame(square=((245, 245, 245), (220, 140, 200, 200)))
    check("a red thing is not (0.26 against blue)", float(np.dot(e1, e3)) < 0.5, str(float(np.dot(e1, e3))))
    check("...nor a green one, and red and green are apart", float(np.dot(e1, emb.embed(green))) < 0.5 and float(np.dot(e3, emb.embed(green))) < 0.5)
    check("a black thing and a white one are apart too (colour without a hue still counts)", float(np.dot(emb.embed(black), emb.embed(white))) < 0.5)
    boxed = emb.embed(red, (220, 140, 200, 200))
    check("a box picks the crop: a tight crop of the red thing is nearer red than blue",
          boxed is not None and float(np.dot(boxed, e3)) > float(np.dot(boxed, e1)) + 0.3)
    check("an empty or tiny crop is None", emb.embed(red, (0, 0, 3, 3)) is None and emb.embed(None) is None)

    # faces: the interface is there; without model files it says so and does nothing
    import cv2
    check("this OpenCV has the face detector and recogniser", hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF"))
    none = bk.OpenCVFaces("/no/such/yunet.onnx", "/no/such/sface.onnx")
    check("without model files: not available, and detect/embed are empty, not errors",
          not none.available() and none.detect(frame()) == [] and none.embed(frame(), bk.Face((0, 0, 1, 1), 1.0)) is None)
    check("the base classes are inert", not bk.FaceBackend().available() and not bk.ObjectBackend().available()
          and not bk.ONNXObjects("/no/such.onnx").available() and bk.ONNXObjects("/no/such.onnx").detect(frame()) == [])

    # the installer: a plan unless told to, and only what is missing
    folder = tempfile.mkdtemp()
    calls = []
    plan = models.plan(folder)
    check("the plan names what and from where, with sizes", [p["name"] for p in plan] == ["face_detector", "face_recognizer"]
          and all(p["url"].startswith("https://github.com/opencv/opencv_zoo/") for p in plan) and plan[1]["approx_mb"] > 30)

    def fake(url, to):
        calls.append(url)
        os.makedirs(os.path.dirname(to), exist_ok=True)
        open(to, "wb").write(b"x" * 4096)
    r = models.install(folder, fetch=fake)
    check("without a yes, NOTHING is fetched", calls == [] and r["done"] == [] and len(r["plan"]) == 2)
    r = models.install(folder, yes=True, fetch=fake)
    check("with a yes, both are fetched", len(calls) == 2 and r["done"] == ["face_detector", "face_recognizer"] and models.missing(folder) == [])
    calls.clear()
    check("and a second time there is nothing left to fetch", models.install(folder, yes=True, fetch=fake)["plan"] == [] and calls == [])
    folder2 = tempfile.mkdtemp()

    def boom(url, to):
        raise OSError("no network")
    r = models.install(folder2, yes=True, fetch=boom)
    check("a failed download is reported and leaves nothing half-written", len(r["failed"]) == 2 and models.missing(folder2) == list(models.MODELS)
          and not any(f.endswith(".part") for f in os.listdir(folder2) if os.path.isdir(folder2)))

    def tiny(url, to):
        os.makedirs(os.path.dirname(to), exist_ok=True)
        open(to, "wb").write(b"<html>")
    r = models.install(tempfile.mkdtemp(), yes=True, fetch=tiny)
    check("a file that is obviously not a model (an error page) is refused and removed", len(r["failed"]) == 2 and r["done"] == [])


@section
def step0_the_eye():
    print("\n=== STEP 0: one frame on request -- in memory for a minute, on disk only if asked ===")
    import cv2
    from src.vision.eye import Eye, capture_once
    from src.senses.devices import CameraSensor

    class FakeCap:
        opened, released = [], []

        def __init__(self, index=0):
            self.index, self.n = index, 0
            FakeCap.opened.append(index)

        def isOpened(self):
            return True

        def set(self, *a):
            return True

        def read(self):
            self.n += 1
            return True, frame(color=(self.n * 10, self.n * 10, self.n * 10), size=(48, 64))

        def release(self):
            FakeCap.released.append(self.index)
    real = cv2.VideoCapture
    cv2.VideoCapture = FakeCap                                    # nothing below can reach a real camera
    try:
        f = capture_once(0, warmup=4)
        check("capture_once skips frames so the exposure settles, keeps the last, and lets the device go",
              f is not None and int(f[0, 0, 0]) == 40 and FakeCap.released == [0], f"{None if f is None else f[0, 0]} {FakeCap.released}")

        t = [1000.0]
        paused = [False]
        shots = []

        def grab():
            shots.append(1)
            return frame(color=(10, 20, 30), size=(48, 64))
        eye = Eye(grab=grab, paused=lambda: paused[0], clock=lambda: t[0])
        fr, how = eye.look()
        check("with no sensor running, it opens the camera for a moment", fr is not None and how == "camera" and len(shots) == 1)
        check("...and keeps that frame in memory only, for the next question", eye.recent() is fr)
        t[0] += 59
        check("for a minute", eye.recent() is not None)
        t[0] += 2
        check("...then it is gone", eye.recent() is None and eye._frame is None)
        eye.look()
        paused[0] = True
        check("privacy mode drops it at once", eye.recent() is None)
        fr, why = eye.look()
        check("...and nothing is looked at while it is on", fr is None and why == "privacy" and len(shots) == 2)
        paused[0] = False
        none = Eye(grab=lambda: None, clock=lambda: t[0])
        check("no camera: said, not a crash", none.look() == (None, "no camera"))
        eye.drop()
        check("a frame can be dropped once used", eye.recent() is None)

        s = CameraSensor(enabled=True)
        check("the camera sensor (on a fake device) is active", s.state.value == "active", s.state.value)
        s.read()
        check("it keeps its latest frame in memory", s.last_frame is not None and s.last_frame_ts > 0)
        e2 = Eye(sensor=lambda: s, grab=lambda: (_ for _ in ()).throw(AssertionError("a second capture must not be opened")))
        before = len(FakeCap.opened)
        fr, how = e2.look()
        check("with the sensor running, a look uses the sensor's own device: no second capture", fr is not None and how == "sensor" and len(FakeCap.opened) == before)
        s.last_frame_ts = 0.0
        fr, how = e2.look()
        check("a stale latest frame is replaced by a fresh read from the same device", fr is not None and how == "sensor")
        s.disable()
        check("turning the camera off drops the picture with the light", s.last_frame is None)
        s2 = CameraSensor(enabled=False)
        check("a sensor that is off is not asked: the eye opens the camera itself", Eye(sensor=lambda: s2, grab=grab).look()[1] == "camera")
    finally:
        cv2.VideoCapture = real

    # saving: the only way a frame reaches the disk
    folder = tempfile.mkdtemp()
    eye = Eye(clock=lambda: 1_800_000_000.0)
    path = eye.save(frame(color=(10, 90, 200), size=(48, 64)), folder, "Blue Mug!")
    check("a saved photo is a JPEG with the label in its name", path and path.endswith("-blue-mug.jpg") and os.path.isfile(path), str(path))
    check("...and it is the picture", cv2.imread(path) is not None and cv2.imread(path).shape[:2] == (48, 64))
    check("nothing else was written", os.listdir(folder) == [os.path.basename(path)])
    check("a bad frame saves nothing and does not crash", eye.save(None, folder) is None)


# ----- fakes for everything above the models --------------------------------------------------------------------------
class FakeObjects:
    def __init__(self, things=(), available=True):
        self.things, self._ok, self.calls = list(things), available, 0

    def available(self):
        return self._ok

    def detect(self, img):
        self.calls += 1
        return list(self.things)


class FakeFaces:
    """detect() returns Face boxes; embed() returns the vector for whoever `who` says is in the frame."""

    def __init__(self, faces=(), vectors=(), available=True):
        self.faces, self.vectors, self._ok, self.calls = list(faces), list(vectors), available, 0

    def available(self):
        return self._ok

    def detect(self, img):
        self.calls += 1
        return list(self.faces)

    def embed(self, img, face):
        i = next((k for k, f in enumerate(self.faces) if f is face), -1)          # identity: Face holds an array
        return self.vectors[i] if 0 <= i < len(self.vectors) else None


def thing(label, score, box=(280, 200, 80, 80)):
    from src.vision.backends import Thing
    return Thing(label, score, box)


def face(box=(200, 100, 120, 140)):
    from src.vision.backends import Face
    return Face(box=box, score=0.95, row=np.zeros(15, np.float32))


def looking(things=(), faces=(), vectors=(), eye_paused=False, grab=True, frames=None, **kw):
    """A Looking with a fake eye (a synthetic frame, or the `frames` in turn), fake models, an in-memory visual memory."""
    from src.daemon.looking import Looking, LookingConfig
    from src.vision.backends import ThingEmbedder
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    t = [1000.0]
    seq = list(frames) if frames else None
    n = [0]

    def grab_fn():
        if seq:
            n[0] += 1
            return seq[(n[0] - 1) % len(seq)]
        return frame(size=(480, 640))
    eye = Eye(grab=grab_fn if grab else (lambda: None), paused=lambda: eye_paused, clock=lambda: t[0])
    objs = kw.pop("objects", FakeObjects(things))
    fcs = kw.pop("faces_backend", FakeFaces(faces, vectors))
    mem = kw.pop("memory", VisualMemory(None, clock=lambda: t[0]))
    user = kw.pop("user", lambda: "")
    kw.setdefault("enrol_checks", False)          # the synthetic frames are flat grey; the quality checks have their own tests
    lk = Looking(eye, fcs, objs, ThingEmbedder(), mem, LookingConfig(enabled=True, **kw), clock=lambda: t[0], user=user)
    lk.t = t
    return lk


@section
def step1_look_around_and_what_am_i_holding():
    print("\n=== STEP 1: 'look around' and 'what am I holding?' -- from the labels, no model ===")
    from src.daemon.looking import parse
    ok = {"look around": "see", "what's in front of the camera": "see", "can you see me": "see", "use the camera": "see",
          "what do you see through the camera": "see", "check the webcam and tell me what you see": "see",
          "what am I holding": "holding", "look at this": "holding", "what's in my hand": "holding", "what is this thing I'm holding": "holding",
          "hey ash, look around please": "see"}
    check("the ways people ask", all((parse(q) or (None,))[0] == k for q, k in ok.items()), str({q: parse(q) for q in ok if (parse(q) or (None,))[0] != ok[q]}))
    check("bare 'what do you see?' is the SCREEN's and stays so", parse("what do you see") is None and parse("what's on my screen") is None)
    check("ordinary talk is not taken for it", all(parse(q) is None for q in ("look at the weather", "I'm looking around for my keys",
                                                                              "what is this", "hold on", "look it up")))

    lk = looking([thing("cup", 0.8), thing("cup", 0.7), thing("laptop", 0.55), thing("person", 0.9)])
    say = lk.hear("look around", 1000.0)
    check("what is in view: people first, then things by count", say == "I see a person, 2 cups and a laptop.", say)
    check("one frame was taken, and nothing was kept of it but the minute in memory", lk.eye.looks == 1 and lk.eye.recent() is not None)
    check("people are 'people', and more than five things are not read out one by one",
          looking([thing("person", .9), thing("person", .8), thing("bottle", .9), thing("book", .9), thing("clock", .9), thing("vase", .9),
                   thing("mouse", .9), thing("cat", .9)]).hear("look around").startswith("I see 2 people, ")
          and "and 3 other things" in looking([thing(c, .9) for c in ("a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1")]).hear("look around"))
    check("when it is not sure of anything, it says so", looking([thing("cup", 0.4)]).hear("look around") == "I see a cup -- though I'm not very sure.")
    check("below the confidence floor is not reported at all", looking([thing("cup", 0.2)]).hear("look around") == "I don't see anything I recognise.")
    check("faces count as people even when the object model does not say so", looking([thing("cup", .9)], faces=[face(), face()]).hear("look around")
          == "I see 2 people and a cup.")
    check("nothing in view", looking([]).hear("look around") == "I don't see anything I recognise.")
    nm = looking(objects=FakeObjects(available=False), faces_backend=FakeFaces(available=False))
    check("with no model installed it says so and how to install", "no model installed" in nm.hear("look around") and "vision install" in nm.hear("look around"))

    hold = looking([thing("person", .95, (0, 0, 640, 480)), thing("laptop", .9, (0, 200, 300, 200)), thing("cup", .8, (20, 20, 60, 60)),
                    thing("bottle", .85, (290, 180, 70, 120))])
    check("what is held is the thing nearest the middle that is not a person or furniture", hold.hear("what am I holding") == "That looks like a bottle.")
    check("unsure: hedged", looking([thing("cup", 0.45, (290, 200, 60, 80))]).hear("look at this") == "That might be a cup, but I'm not sure.")
    check("nothing to hold: it says to hold it closer", "holding it closer" in looking([thing("person", .9)]).hear("what am I holding"))
    check("without an object model it says what is missing", "object model" in looking(objects=FakeObjects(available=False)).hear("what am I holding"))
    check("with faces but no object model it says so honestly", "no object model" in looking(objects=FakeObjects(available=False)).hear("what am I holding"))

    check("privacy mode: it does not look, and says why", looking(eye_paused=True).hear("look around") == "I'm paused for privacy, so I'm not looking.")
    check("no camera: said", "no camera I can open" in looking(grab=False).hear("look around"))
    off = looking([thing("cup", .9)], )
    off.cfg.enabled = False
    check("switched off: nothing happens and no frame is taken", off.hear("look around") is None and off.eye.looks == 0)
    check("an unrelated sentence takes no frame", looking([thing("cup", .9)]).hear("what time is it") is None)

    # on the runtime
    import tst_effects as te
    from src.senses.base import Modality, SensorEvent
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    eye = Eye(grab=lambda: frame(size=(480, 640)))
    io = dict(eye=eye, faces=FakeFaces(), objects=FakeObjects([thing("cup", .8), thing("laptop", .7)]), memory=VisualMemory(None))
    rt = te.rig(vision=True, vision_io=io, grammar=True, vision_cfg=dict(enrol_checks=False))
    rt.brain.user = "Khalid"
    rt.say("look around")
    check("by voice: answered by rule", rt.out[-1] == "I see a cup and a laptop.", rt.out[-1])
    n = eye.looks
    rt.say("what do you see")
    check("bare 'what do you see?' does not open the camera", eye.looks == n, str(rt.out[-1:]))
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="look around", addressed=False, salience=0.4, meta={}))
    rt._tick()
    check("overheard speech never opens the camera", eye.looks == n)
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="look around", addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice is refused, and no frame is taken", rt.out[-1].startswith("Sorry, I only do that for Khalid") and eye.looks == n, rt.out[-1])
    check("the status carries it", rt.status()["vision"]["enabled"] and rt.status()["vision"]["objects"] and rt.status()["vision"]["looks"] == 1)
    from src.senses import address
    check("without the name, 'look around' is at least 'probably for ASH' (a command a handler understands)",
          address.score("look around").confidence >= address.LIKELY_AT, str(address.score("look around").reasons))
    bare = te.rig()
    check("with vision off a bare runtime has no eyes", bare.looking is None and bare.status()["vision"] is None)
    bare.say("look around")
    check("...and the sentence is left to the rest of ASH", "I see" not in bare.out[-1])


@section
def step2_teach_by_showing():
    print("\n=== STEP 2: 'remember this as my blue mug' / 'this is Omar, remember his face' -- numbers, not pictures ===")
    from src.daemon.looking import parse
    ok = {"remember this as my blue mug": ("teach_thing", "my blue mug"), "remember this is my blue mug": ("teach_thing", "my blue mug"),
          "this is my blue mug, remember it": ("teach_thing", "my blue mug"), "remember this person as Omar": ("teach_person", "Omar"),
          "remember Omar's face": ("teach_person", "Omar"), "this is Omar, remember his face": ("teach_person", "Omar"),
          "learn this face as Ali": ("teach_person", "Ali"), "remember my face": ("teach_me", None)}
    bad = {q: parse(q) for q, (k, lab) in ok.items() if (parse(q) or (None, {}))[0] != k or (lab and parse(q)[1].get("label") != lab)}
    check("the ways people teach it", not bad, str(bad))
    check("the photo clause is understood, with or without the comma", parse("remember this as my mug, and save the photo")[1].get("photo") == "1"
          and parse("learn my face and save the photo")[1].get("photo") == "1")
    check("'this is X' alone, and other 'remember's, are ordinary talk", all(parse(q) is None for q in (
        "this is great", "this is my blue mug", "remember to call mom", "remember this", "remember Omar is my cousin", "remember that I like coffee")))

    # ---- people
    omar, ali = vec(1), vec(2)
    lk = looking(faces=[face()], vectors=[omar])
    say = lk.hear("remember this person as Omar", 1000.0)
    check("a face is taught: numbers, not the picture, only here, and a nudge for another angle",
          say.startswith("Okay, I'll recognise Omar. I keep numbers describing the face, not the picture") and "another angle" in say, say)
    v = lk.memory.get("Omar")
    check("...kept as a person with one look", v.kind == "person" and len(v.embeds) == 1 and v.photo == "")
    lk.faces.vectors = [vec(1, noise=0.1, base=omar)]
    check("a second look adds to it", lk.hear("remember Omar's face", 1001.0) == "Got another look at Omar (2 now).")
    lk.faces.vectors = [vec(1, noise=0.15, base=omar)]
    say = lk.hear("remember this person as Ali", 1002.0)
    check("the same face under a second name is refused, with how to fix it", say.startswith("That looks like Omar, who I already know.")
          and 'say "forget Omar" first' in say and lk.memory.get("Ali") is None, say)
    lk.faces.vectors = [ali]
    check("a different face is a different person", lk.hear("remember this person as Ali", 1003.0).startswith("Okay, I'll recognise Ali.")
          and len(lk.memory.list("person")) == 2)

    nof = looking(faces=[], vectors=[])
    check("no face in view: it says so", nof.hear("remember Omar's face") == "I can't see a face. Face the camera and say it again.")
    two = looking(faces=[face((10, 10, 100, 100)), face((300, 10, 110, 110))], vectors=[omar, ali])
    check("two faces of similar size: it will not guess which", "more than one face" in two.hear("remember Omar's face") and two.memory.list() == [])
    near = looking(faces=[face((10, 10, 60, 60)), face((300, 10, 200, 200))], vectors=[omar, ali])
    near.hear("remember Ali's face")
    check("one clearly the nearest: that is the one (a passer-by behind is not enrolled)", near.memory.get("Ali") is not None
          and near.memory.match(ali, "person", 0.9) is not None and near.memory.match(omar, "person", 0.9) is None)
    check("without the face models: it says how to install them", "face models aren't installed" in
          looking(faces_backend=FakeFaces(available=False)).hear("remember Omar's face"))
    check("an unreadable face is not stored", "couldn't read that face" in looking(faces=[face()], vectors=[]).hear("remember Omar's face"))
    me = looking(faces=[face()], vectors=[omar], user=lambda: "Khalid")
    check("'remember my face' uses your name", me.hear("remember my face").startswith("Okay, I'll recognise Khalid.") and me.memory.get("Khalid") is not None)
    check("...and asks if it does not know it", "don't know your name yet" in looking(faces=[face()], vectors=[omar]).hear("remember my face"))
    check("a name that is not a name is refused", "I need a name" in looking(faces=[face()], vectors=[omar]).hear("remember this person as it"))

    # ---- things
    blue = frame(square=((200, 80, 20), (220, 140, 200, 200)))
    blue2 = frame(square=((205, 85, 25), (225, 150, 190, 190)), noise=5, seed=2)
    red = frame(square=((30, 30, 210), (220, 140, 200, 200)))
    th = looking([thing("cup", 0.8, (220, 140, 200, 200))], frames=[blue, blue2, red])
    say = th.hear("remember this as my blue mug", 1000.0)
    check("a thing is taught, with what the detector took it for", say == "Okay, I'll recognise my blue mug (it looks like a cup). "
          "Show it to me again in a different position and I'll be surer.", say)
    check("...kept as a thing", th.memory.get("blue mug").kind == "thing")
    check("a second look adds to it", th.hear("this is my blue mug, remember it", 1001.0) == "Got another look at my blue mug (2 now).")
    th.objects.things = [thing("cup", 0.8, (220, 140, 200, 200))]
    say = th.hear("remember this as my red mug", 1002.0)
    check("a differently coloured one is a different thing", say.startswith("Okay, I'll recognise my red mug") and len(th.memory.list("thing")) == 2, say)
    again = looking([thing("cup", 0.8, (220, 140, 200, 200))], frames=[blue], memory=th.memory)
    say = again.hear("remember this as my tea cup", 1003.0)
    check("the same thing under a second name is refused", say.startswith("That looks like my blue mug, which I already know.") and th.memory.get("tea cup") is None, say)
    nomodel = looking(objects=FakeObjects(available=False), frames=[blue])
    check("with no object model it still learns it, from the middle of the frame, without claiming to know what it is",
          nomodel.hear("remember this as my blue mug") == "Okay, I'll recognise my blue mug. Show it to me again in a different position and I'll be surer.")
    check("a thing cannot take a person's name", "already a person" in looking([thing("cup", .8)], faces=[face()], vectors=[omar], memory=lk.memory).hear("remember this as Omar"))

    # ---- photos, privacy, nothing kept on failure
    folder = tempfile.mkdtemp()
    ph = looking([thing("cup", 0.8, (220, 140, 200, 200))], frames=[blue], photo_dir=folder)
    say = ph.hear("remember this as my blue mug, and save the photo")
    saved = [f for f in os.listdir(folder)]
    check("a photo is saved only when asked, named for the thing, and the pointer is kept",
          len(saved) == 1 and saved[0].endswith("-my-blue-mug.jpg") and saved[0] in say and ph.memory.get("blue mug").photo.endswith(saved[0]), f"{saved} {say}")
    folder2 = tempfile.mkdtemp()
    nop = looking([thing("cup", 0.8, (220, 140, 200, 200))], frames=[blue], photo_dir=folder2)
    nop.hear("remember this as my blue mug")
    check("without the ask, nothing is written to disk", os.listdir(folder2) == [])
    paused = looking(faces=[face()], vectors=[omar], eye_paused=True)
    check("in privacy mode nothing is looked at or learned", paused.hear("remember Omar's face") == "I'm paused for privacy, so I'm not looking."
          and paused.memory.list() == [])

    # ---- on the runtime
    import tst_effects as te
    from src.senses.base import Modality, SensorEvent
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    mem = VisualMemory(None)
    io = dict(eye=Eye(grab=lambda: blue), faces=FakeFaces([face()], [omar]), objects=FakeObjects([thing("cup", .8, (220, 140, 200, 200))]), memory=mem)
    rt = te.rig(vision=True, vision_io=io, vision_cfg=dict(enrol_checks=False))
    rt.brain.user = "Khalid"
    rt.say("remember this as my blue mug")
    check("by voice: taught", rt.out[-1].startswith("Okay, I'll recognise my blue mug") and mem.get("blue mug") is not None, rt.out[-1])
    rt.say("remember my face")
    check("'remember my face' uses the name the runtime knows", mem.get("Khalid") is not None, rt.out[-1])
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="remember this person as Eve", addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice cannot teach it anyone", rt.out[-1].startswith("Sorry, I only do that for Khalid") and mem.get("Eve") is None, rt.out[-1])
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="remember this person as Eve", addressed=False, salience=0.4, meta={}))
    rt._tick()
    check("overheard speech teaches it nothing", mem.get("Eve") is None)


def near(base, cos, seed=5):
    """A unit vector whose cosine similarity to `base` is `cos`."""
    r = np.random.RandomState(seed).randn(len(base)).astype(np.float32)
    r -= float(np.dot(r, base)) * base
    r /= np.linalg.norm(r)
    return (cos * base + np.sqrt(1 - cos * cos) * r).astype(np.float32)


@section
def step3_recognise_who_and_what():
    print("\n=== STEP 3: 'who's this?', 'is this my mug?', 'where are my keys?' -- cautious, and honest about close calls ===")
    import time as _t
    from src.daemon.looking import parse
    omar, ali = vec(1), vec(2)
    check("the ways people ask", all((parse(q) or (None,))[0] == k for q, k in (
        ("who's this", "who"), ("who is in front of the camera", "who"), ("do you recognize me", "who"), ("is this my blue mug", "is_this"),
        ("is this Omar", "is_this"), ("where are my keys", "where"), ("where did I leave my keys", "where"))))
    check("'who am I' is the memory's, not the camera's; everyday 'is this...' / 'where is...' are not it",
          parse("who am I") is None and parse("is this correct") is None and parse("is this the right way") is None)

    def people(**kw):
        lk = looking(faces=[face()], vectors=[omar], **kw)
        lk.memory.enroll("Omar", "person", omar)
        lk.memory.enroll("Ali", "person", ali)
        return lk
    lk = people()
    check("a known face is named", lk.hear("who's this", 1000.0) == "That's Omar.")
    lk.t[0] = 5000.0
    lk.hear("who's this", 5000.0)
    check("...and seeing someone updates when and how often", lk.memory.get("Omar").last_seen == 5000.0 and lk.memory.get("Omar").seen == 3)
    lk.faces.vectors = [vec(9)]
    check("an unknown face is not named, and it says how to teach it", lk.hear("who is that").startswith("I don't know them.") and "remember this person as <name>" in lk.hear("who is that"))
    lk.faces.vectors = [near(omar, 0.35)]
    check("a face that is nearly someone: hedged, not claimed", lk.hear("who's this") == "That might be Omar, but I'm not sure.")
    lk.faces.vectors = [near(omar, 0.30)]
    check("...but not too nearly (0.30 is nobody)", lk.hear("who's this").startswith("I don't know them."))
    twin = people()
    twin.memory.enroll("Twin", "person", near(omar, 0.97, seed=7))
    check("two people who look alike: not sure rather than a guess (the margin)", not twin.hear("who's this").startswith("That's "), twin.hear("who's this"))
    check("nobody in view", looking(faces=[], vectors=[]).hear("who's this") == "I don't see anyone.")
    check("without the face models: it says so", "face models aren't installed" in looking(faces_backend=FakeFaces(available=False)).hear("who's this"))
    two = looking(faces=[face((10, 10, 100, 100)), face((300, 10, 100, 100))], vectors=[omar, ali])
    two.memory.enroll("Omar", "person", omar)
    two.memory.enroll("Ali", "person", ali)
    check("two known people", two.hear("who do you see") == "I see Omar and Ali.")
    mixed = looking(faces=[face((10, 10, 100, 100)), face((300, 10, 100, 100)), face((500, 10, 90, 90))], vectors=[omar, vec(8), vec(9)])
    mixed.memory.enroll("Omar", "person", omar)
    check("known and unknown together", mixed.hear("who do you see") == "I see Omar, and 2 people I don't know.")
    allunk = looking(faces=[face((10, 10, 100, 100)), face((300, 10, 100, 100))], vectors=[vec(8), vec(9)])
    check("nobody known", allunk.hear("who do you see") == "I see 2 people and I don't know any of them.")

    # is this X
    check("'is this Omar?' when Omar is not someone it was taught is left alone, and no frame is taken",
          people().hear("is this Zed") is None and not people().wants("is this Zed") and people().eye.looks == 0)
    check("...and 'is this correct' never takes a frame", people().hear("is this correct") is None)
    lk = people()
    check("yes", lk.hear("is this Omar") == "Yes, that's Omar.")
    lk.faces.vectors = [ali]
    check("no -- it is someone else it knows", lk.hear("is this Omar") == "No -- that looks like Ali.")
    lk.faces.vectors = [vec(9)]
    check("no -- a stranger", lk.hear("is this Omar") == "No, that doesn't look like Omar.")
    lk.faces.vectors = [near(omar, 0.35)]
    check("might be", lk.hear("is this Omar") == "It might be Omar, but I'm not sure.")
    check("no face in view", looking(faces=[], vectors=[]).hear("is this Omar") is None)

    # things
    blue = frame(square=((200, 80, 20), (220, 140, 200, 200)))
    red = frame(square=((30, 30, 210), (220, 140, 200, 200)))
    cup = [thing("cup", 0.8, (220, 140, 200, 200))]

    def shelf(*frames, things_=cup, **kw):
        lk = looking(list(things_), frames=list(frames), **kw)
        return lk
    lk = shelf(blue, red, blue)
    lk.hear("remember this as my blue mug")
    lk.hear("remember this as my red mug")
    check("holding something it was taught: it says which, ahead of what a detector calls it", lk.hear("what am I holding") == "That's your blue mug.")
    check("is this my blue mug: yes (blue is in front)", lk.hear("is this my blue mug") is not None)
    lk.eye._grab = lambda: red
    check("...no, it looks like the red one", lk.hear("is this my blue mug") == "No -- that looks like your red mug.")
    lk.eye._grab = lambda: frame(square=((60, 160, 60), (220, 140, 200, 200)))
    check("something it was never shown: not that one, and says so", lk.hear("is this my blue mug") == "I don't think so -- it doesn't look like your blue mug.")
    lk.eye._grab = lambda: blue
    check("what is in view names what it knows and describes the rest", lk.hear("look around") == "I see your blue mug.")
    lk.objects.things = cup + [thing("laptop", .8, (10, 300, 100, 100)), thing("person", .9, (0, 0, 100, 200))]
    check("...with people and things together", lk.hear("look around") == "I see a person, your blue mug and a laptop.")
    lk.faces.faces, lk.faces.vectors = [face()], [omar]
    lk.memory.enroll("Omar", "person", omar)
    check("...by name when the person is known", lk.hear("look around") == "I see Omar, your blue mug and a laptop.")

    # where
    now0 = _t.mktime((2026, 6, 15, 12, 0, 0, 0, 0, -1))
    keys = shelf(blue, things_=cup)
    keys.t[0] = now0 - 3 * 3600
    keys.memory.clock = lambda: keys.t[0]
    keys.hear("remember this as my keys", keys.t[0])
    check("where is it, with the camera on it right now", keys.hear("where are my keys", now0) == "Your keys are in view right now.")
    keys.eye._grab = lambda: red
    keys.memory.get("keys").last_seen = now0 - 3 * 3600
    keys.memory.get("keys").seen = 3
    check("not in view: when it was last seen, today", keys.hear("where are my keys", now0) == "I last saw your keys at 09:00 today. I can only say what the camera has shown me, not where things are in the room.")
    for ago, phrase in ((30, "just now"), (600, "10 minutes ago"), (86400, "yesterday at 12:00"), (5 * 86400, "on 10 Jun at 12:00")):
        keys.memory.get("keys").last_seen = now0 - ago
        check(f"...{phrase}", keys.hear("where did I leave my keys", now0).startswith(f"I last saw your keys {phrase}."), keys.hear("where did I leave my keys", now0))
    only = shelf(blue, things_=cup)
    only.memory.enroll("my wallet", "thing", vec(4, dim=224))
    check("only known from being shown: it says so", only.hear("where is my wallet", now0).startswith("I only know your wallet from when you showed me"), str(only.hear("where is my wallet", now0)))
    om = people()
    check("a person in view", om.wants("where is Omar") and om.hear("where is Omar", now0) == "Omar is in view right now.")
    om.faces.vectors = [vec(9)]
    om.memory.get("Omar").last_seen = now0 - 7200
    om.memory.get("Omar").seen = 4
    check("a person not in view: when they were last seen (at 10:00 today)", om.hear("where is Omar", now0).startswith("I last saw Omar at 10:00 today."),
          str(om.hear("where is Omar", now0)))
    check("'where is the nearest cafe' takes no frame", shelf(blue).hear("where is the nearest cafe") is None)

    # runtime
    import tst_effects as te
    from src.senses.base import Modality, SensorEvent
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    mem = VisualMemory(None)
    mem.enroll("Omar", "person", omar)
    eye = Eye(grab=lambda: frame(size=(480, 640)))
    rt = te.rig(vision=True, vision_io=dict(eye=eye, faces=FakeFaces([face()], [omar]), objects=FakeObjects([]), memory=mem), grammar=True, vision_cfg=dict(enrol_checks=False))
    rt.brain.user = "Khalid"
    rt.say("who's this")
    check("by voice", rt.out[-1] == "That's Omar.", rt.out[-1])
    n = eye.looks
    rt.say("is this correct")
    check("an everyday 'is this...' does not open the camera", eye.looks == n)
    rt.say("is this Omar")
    check("'is this Omar' does", rt.out[-1] == "Yes, that's Omar." and eye.looks == n + 1, rt.out[-1])
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="who's this", addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest cannot ask who is in front of the camera", rt.out[-1].startswith("Sorry, I only do that for Khalid") and eye.looks == n + 1, rt.out[-1])
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="where is the nearest cafe", addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("...but an everyday question from a guest is not refused", not rt.out[-1].startswith("Sorry, I only do that"), rt.out[-1])


@section
def step4_presence_and_the_stranger_hold():
    print("\n=== STEP 4: who is here, from the camera's own frames -- a state, never a recording ===")
    from src.vision.presence import Presence
    from src.daemon.looking import LookingConfig as LookingConfigDefaults
    p = Presence(away_after_s=90.0, hold_s=120.0, confirm=2)
    check("someone seen is an arrival", p.update(["Omar"], 100.0) == [("arrived", "Omar")] and p.here(100.0) == ["Omar"])
    check("...seen again is not another", p.update(["Omar"], 108.0) == [] and p.here(110.0) == ["Omar"])
    check("not seen for a while is not yet gone (a head turned away)", p.update([], 150.0) == [] and p.here(150.0) == ["Omar"])
    check("gone after the away window, said once", p.update([], 200.0) == [("left", "Omar")] and p.here(200.0) == [] and p.update([], 210.0) == [])
    check("...and back is an arrival again", p.update(["Omar"], 400.0) == [("arrived", "Omar")])
    two = Presence()
    check("two at once, in name order", two.update(["Omar", "Ali"], 10.0) == [("arrived", "Ali"), ("arrived", "Omar")])

    s = Presence(away_after_s=90.0, hold_s=120.0, confirm=2)
    check("one reading of an unknown face is not a stranger (a bad angle)", s.update([None], 10.0) == [] and not s.stranger(10.0))
    check("...two in a row is", s.update([None], 18.0) == [("stranger", "")] and s.stranger(18.0))
    check("...said once, while they stay", s.update([None], 26.0) == [] and s.update([None], 34.0) == [])
    check("the hold outlasts the last sighting by the hold time", s.stranger(34.0 + 119) and not s.stranger(34.0 + 121))
    check("a flicker (unknown, nobody, unknown) is not a stranger", Presence().update([None], 1.0) == [] and
          (lambda q: (q.update([None], 1.0), q.update([], 9.0), q.update([None], 17.0))[2] == [])(Presence()))
    check("a stranger who went and a different one later: a new sighting", s.update([], 400.0) == [] and s.update([None], 410.0) == []
          and s.update([None], 418.0) == [("stranger", "")])
    m = Presence()
    check("a known person and an unknown one together", m.update(["Omar", None], 1.0) == [("arrived", "Omar")] and m.update(["Omar", None], 9.0) == [("stranger", "")])
    check("it reports whether there is a recent reading to answer from", m.fresh(10.0, 24.0) and not m.fresh(100.0, 24.0) and not Presence().fresh(5.0, 24.0))
    m.clear()
    check("clearing forgets everyone", m.here(9.0) == [] and not m.stranger(9.0))
    check("the state holds names and times only", all(isinstance(k, str) and isinstance(v, float) for k, v in p.present.items()))

    # Looking.watch
    omar, ali = vec(1), vec(2)
    lk = looking(faces=[face()], vectors=[omar], presence=True)
    lk.memory.enroll("Omar", "person", omar)
    check("a reading: Omar arrives, and 'last seen' follows", lk.watch(frame(), 2000.0) == [("arrived", "Omar")] and lk.memory.get("Omar").last_seen == 2000.0)
    lk.faces.vectors = [vec(9)]
    check("an unknown face twice: a stranger", lk.watch(frame(), 2008.0) == [] and lk.watch(frame(), 2016.0) == [("stranger", "")])
    check("the hold is on", lk.stranger_present(2016.0) and not lk.stranger_present(2016.0 + 130))
    lk.cfg.hold_when_stranger = False
    check("...unless it was switched off", not lk.stranger_present(2016.0))
    off = looking(faces=[face()], vectors=[omar])
    off.memory.enroll("Omar", "person", omar)
    check("presence off: watch does nothing", off.watch(frame(), 1.0) == [] and not off.stranger_present(1.0))
    check("no face models: watch does nothing", looking(faces_backend=FakeFaces(available=False), presence=True).watch(frame(), 1.0) == [])
    check("announcements are off unless asked", lk.announcement("arrived", "Omar") == "" and lk.announcement("stranger", "") == "")
    lk.cfg.announce_known = lk.cfg.announce_strangers = True
    check("...and then plain", lk.announcement("arrived", "Omar", "Khalid") == "Omar just came in." and lk.announcement("arrived", "Khalid", "Khalid") == "Welcome back."
          and lk.announcement("stranger", "") == "Someone I don't know just came into view." and lk.announcement("left", "Omar") == "")
    st = looking(faces=[face()], vectors=[omar], presence=True)
    st.memory.enroll("Omar", "person", omar)
    check("who is here, with nothing read yet: a new picture is taken", st.hear("who's here", 3000.0) == "That's Omar." and st.eye.looks == 1)
    st.watch(frame(), 3010.0)
    n = st.eye.looks
    check("who is here, once it is watching: answered from that, no new capture", st.hear("who's here", 3015.0) == "Omar is here." and st.eye.looks == n)
    st.faces.vectors = [omar, vec(9)]
    st.faces.faces = [face((10, 10, 100, 100)), face((300, 10, 100, 100))]
    st.watch(frame(), 3020.0)
    st.watch(frame(), 3028.0)
    check("...with a stranger in it", st.hear("who's here", 3030.0) == "I see Omar, and someone I don't know.")
    check("...stale, it looks again", st.hear("who's here", 3030.0 + 600) is not None and st.eye.looks == n + 1)

    # on the runtime: the camera sensor's own frame, a hold on private things, announcements
    import tst_effects as te
    from src.senses.base import State
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    import time as _t
    mem = VisualMemory(None)
    mem.enroll("Omar", "person", omar)
    faces = FakeFaces([face((10, 10, 100, 100))], [omar])

    def runtime(**cfg):
        io = dict(eye=Eye(grab=lambda: frame()), faces=faces, objects=FakeObjects([]), memory=mem)
        rt = te.rig(vision=True, vision_io=io, vision_cfg=dict({"presence": True, "presence_interval_s": 1.0, "presence_background": False}, **cfg.pop("vcfg", {})),
                    clipboard=True, clipboard_io=(lambda: "some private text", lambda t: True), briefings=True, **cfg)
        rt.brain.user = "Khalid"
        cam = type("Cam", (), {"due": lambda self, now: False, "name": "camera", "close": lambda self: None,
                               "status": lambda self: {}, "disable": lambda self: None, "enable": lambda self: None})()
        cam.state, cam.last_frame, cam.last_frame_ts = State.ACTIVE, frame(), _t.time()
        rt.periphery.sensors["camera"] = cam
        rt.cam = cam
        return rt

    def pulse(rt, secs=5):
        rt.t[0] += secs
        rt.cam.last_frame_ts = _t.time()
        rt._tick()
    rt = runtime()
    pulse(rt)
    check("the runtime reads the camera sensor's frame on a tick: Omar is here", rt.looking.presence.here(rt.t[0]) == ["Omar"] and rt.status()["vision"]["here"] == ["Omar"],
          str(rt.status()["vision"]))
    n = faces.calls
    rt.t[0] += 0.2
    rt._tick()
    check("...and not more often than the interval", faces.calls == n)
    rt.say("what's on my clipboard")
    check("no stranger: private things are read out", "some private text" in rt.out[-1], rt.out[-1])
    faces.faces, faces.vectors = [face((10, 10, 100, 100)), face((300, 10, 100, 100))], [omar, vec(9)]
    pulse(rt)
    pulse(rt)
    check("a stranger in view", rt.looking.stranger_present(rt.t[0]) and rt.status()["vision"]["stranger"])
    rt.say("what's on my clipboard")
    check("the clipboard is NOT read out while they are there", rt.out[-1] == rt.HOLD_LINE and "some private text" not in rt.out[-1], rt.out[-1])
    rt.say("summarize my clipboard")
    check("...nor sent to the model", rt.out[-1] == rt.HOLD_LINE)
    rt.say("brief me")
    check("...nor the day's briefing", rt.out[-1] == rt.HOLD_LINE, rt.out[-1])
    check("a hold is a plain line to say, not a silence", rt._maybe_hold("secret", rt.t[0]) == rt.HOLD_LINE)
    faces.faces, faces.vectors = [face((10, 10, 100, 100))], [omar]
    rt.t[0] += 400
    pulse(rt)
    check("when they have gone for the hold time, private things are read again", not rt.looking.stranger_present(rt.t[0]))
    rt.say("what's on my clipboard")
    check("...", "some private text" in rt.out[-1], rt.out[-1])

    # the reading runs on a worker thread: a slow detector never holds the loop
    import threading as _th
    gate_ev = _th.Event()
    slow = FakeFaces([face((10, 10, 100, 100))], [omar])
    real_detect = slow.detect
    slow.detect = lambda fr: (gate_ev.wait(5), real_detect(fr))[1]
    faces_keep, faces = faces, slow
    bg = runtime(vcfg=dict(presence_background=True))
    t0 = _t.time()
    pulse(bg)
    check("a slow detector does not hold the tick", _t.time() - t0 < 1.0 and bg.looking.presence.here(bg.t[0]) == [], str(_t.time() - t0))
    pulse(bg)
    check("...and only one reading is in flight", slow.calls == 0 and bg._presence_job is not None)
    gate_ev.set()
    bg._presence_job.result(5)
    pulse(bg)
    check("the answer is picked up on a later tick", bg.looking.presence.here(bg.t[0]) == ["Omar"], str(bg.status()["vision"]))
    bg._presence_exec.shutdown(wait=True)
    faces = faces_keep

    quiet = runtime(vcfg=dict(hold_when_stranger=False))
    faces.faces, faces.vectors = [face((10, 10, 100, 100)), face((300, 10, 100, 100))], [omar, vec(9)]
    pulse(quiet)
    pulse(quiet)
    quiet.say("what's on my clipboard")
    check("hold_when_stranger off: no hold", "some private text" in quiet.out[-1], quiet.out[-1])

    spoken = runtime(vcfg=dict(announce_strangers=True, announce_known=True))
    faces.faces, faces.vectors = [face((10, 10, 100, 100)), face((300, 10, 100, 100))], [omar, vec(9)]
    n_out = len(spoken.out)
    pulse(spoken)
    pulse(spoken)
    check("announcing on: it says so aloud", "Omar just came in." in spoken.out[n_out:] and "Someone I don't know just came into view." in spoken.out[n_out:], str(spoken.out[n_out:]))
    silent = runtime()
    n_out = len(silent.out)
    pulse(silent)
    pulse(silent)
    check("announcing off (the default): nothing is said, it shows on the console and answers when asked", silent.out[n_out:] == [])
    muted = runtime(vcfg=dict(announce_strangers=True))
    muted.gate.muted = True
    n_out = len(muted.out)
    pulse(muted)
    pulse(muted)
    check("muted: nothing is said even then", muted.out[n_out:] == [])
    paused = runtime()
    paused.privacy.paused = True
    n = faces.calls
    pulse(paused)
    check("privacy mode: the camera frame is not even looked at", faces.calls == n)
    off = runtime()
    off.cam.state = State.DISABLED
    n = faces.calls
    pulse(off)
    check("camera sensor off: nothing is read", faces.calls == n)
    old = runtime()
    old.cam.last_frame_ts = 1.0
    n = faces.calls
    old.t[0] += 5
    old._tick()
    check("a frame that is old is not used (the camera has stopped)", faces.calls == n)
    no_presence = runtime()
    no_presence.looking.cfg.presence = False
    n = faces.calls
    pulse(no_presence)
    check("presence off: a runtime with vision on but no presence reads nothing", faces.calls == n)
    check("...and presence is off unless asked for", LookingConfigDefaults().presence is False)


@section
def step5_manage_what_it_knows():
    print("\n=== STEP 5: what it knows, forgetting it, saving a photo -- and where it shows ===")
    import json
    import time as _t
    from src.daemon.looking import parse
    omar, ali = vec(1), vec(2)
    ok = {"what have you saved": "list", "what faces do you know": "list", "who do you know by sight": "list", "what have I shown you": "list",
          "export the faces": "export", "export what you've seen": "export", "save that photo": "save_photo", "save the last picture": "save_photo",
          "forget all the faces": "forget_kind", "forget things": "forget_kind", "forget everything you've seen": "forget_kind",
          "forget Omar's face": "forget", "forget my blue mug": "forget"}
    check("the ways people ask", all((parse(q) or (None,))[0] == k for q, k in ok.items()), str({q: parse(q) for q in ok if (parse(q) or (None,))[0] != ok[q]}))

    def shelf():
        lk = looking(faces=[face()], vectors=[omar], presence=True)
        lk.memory.enroll("Omar", "person", omar)
        lk.memory.enroll("Ali", "person", ali)
        lk.memory.enroll("my blue mug", "thing", vec(3, dim=224))
        lk.memory.get("Omar").last_seen = 1000.0 - 600
        lk.memory.get("Ali").last_seen = 1000.0 - 90000
        return lk
    lk = shelf()
    say = lk.hear("what have you saved", 1000.0)
    check("the list: people then things, most recently seen first, with when, and that they are numbers", say.startswith("I know 2 people: Omar (last seen 10 minutes ago) and Ali (last seen ")
          and "1 thing: your blue mug (last seen just now)" in say and say.endswith("I keep numbers describing them, not pictures."), say)
    check("nothing taught yet: it says how", "haven't been taught anyone or anything yet" in looking().hear("what have you saved"))
    many = looking()
    for i in range(10):
        many.memory.enroll(f"Person {chr(65 + i)}", "person", vec(100 + i))
    check("a long list is cut and counted", "10 people:" in many.hear("what have you saved") and ", and 2 more" in many.hear("what have you saved"))
    check("a list takes no picture", lk.eye.looks == 0)

    # forgetting
    lk.memory.get("my blue mug").photo = "C:/Users/x/Pictures/ASH/ASH-1-my-blue-mug.jpg"
    lk.presence.update(["Omar"], 1000.0)
    say = lk.hear("forget Omar's face", 1000.0)
    check("a person: gone, the numbers say so, and what was said in conversation is separate",
          say.startswith("Forgotten: Omar. I've deleted the numbers I recognised them by.") and 'forget what I said about Omar' in say and lk.memory.get("Omar") is None, say)
    check("...and no longer counted as here", "Omar" not in lk.presence.here(1001.0))
    say = lk.hear("forget my blue mug", 1000.0)
    check("a thing: gone, and the saved photo is the user's file", say.startswith("Forgotten: my blue mug. I've deleted the numbers I recognised it by.") and "still where it was" in say and lk.memory.get("blue mug") is None, say)
    check("...forgetting something not taught is left alone (and no frame)", not lk.wants("forget Zed") and lk.hear("forget Zed") is None)
    lk = shelf()
    say = lk.hear("forget all the faces", 1000.0)
    check("all faces", say == "Forgotten: 2 people -- the numbers I recognised them by are gone." and lk.memory.list("person") == [] and len(lk.memory.list("thing")) == 1, say)
    check("...none left to forget", lk.hear("forget the faces") == "I don't have any people to forget.")
    check("all things", lk.hear("forget things") == "Forgotten: 1 thing -- the numbers I recognised them by are gone.")
    lk = shelf()
    check("everything", lk.hear("forget everything you've seen").startswith("Forgotten: 3 entries") and lk.memory.list() == [])

    # photos: the picture it just took, for a minute, only if asked
    folder = tempfile.mkdtemp()
    ph = looking([thing("cup", .8)], photo_dir=folder)
    check("no picture taken: it says so", ph.hear("save that photo").startswith("I don't have a picture from the last minute"))
    ph.hear("look around", 2000.0)
    say = ph.hear("save that photo", 2001.0)
    files = os.listdir(folder)
    check("after a look: the photo is saved, named by the time", len(files) == 1 and files[0].startswith("ASH-") and files[0].endswith(".jpg") and files[0] in say, f"{files} {say}")
    check("...and the picture is dropped once saved", ph.hear("save that photo", 2002.0).startswith("I don't have a picture"))
    ph.hear("look around", 3000.0)
    ph.t[0] = 3000.0 + 90
    check("a minute later it is gone", ph.hear("save the picture", 3000.0 + 90).startswith("I don't have a picture") and len(os.listdir(folder)) == 1)
    pp = looking([thing("cup", .8)], photo_dir=tempfile.mkdtemp())
    pp.hear("look around", 4000.0)
    pp.eye.paused = lambda: True
    check("privacy mode drops it before it can be saved", pp.hear("save that photo", 4001.0).startswith("I don't have a picture"))

    # export: labels and times, never the numbers
    lk = shelf()
    say = lk.hear("export the faces", 1000.0)
    path = say.split("Saved 3 entries to ")[1].split(". It lists")[0]
    data = json.load(open(path, encoding="utf-8"))
    check("an export is readable, lists who and when, and holds no numbers", sorted(i["label"] for i in data["items"]) == ["Ali", "Omar", "my blue mug"]
          and "embeds" not in json.dumps(data) and all("taught" in i and "last_seen" in i for i in data["items"]), say)
    os.remove(path)

    # the panel and the cards
    import tst_effects as te
    from src.daemon import headsup, learning
    from src.senses.base import Modality, SensorEvent
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    mem = VisualMemory(None)
    mem.enroll("Omar", "person", omar)
    mem.enroll("my blue mug", "thing", vec(3, dim=224))
    rt = te.rig(vision=True, vision_io=dict(eye=Eye(grab=lambda: frame()), faces=FakeFaces([face()], [omar]), objects=FakeObjects([]), memory=mem),
                vision_cfg=dict(presence=True))
    rt.brain.user = "Khalid"
    learning._CACHE.clear()
    lines = learning.snapshot(rt, 1000.0).get("lines", [])
    check("the panel lists what it can recognise, and says it is numbers", any(l == "Faces and things I can recognise: 1 person (Omar); 1 thing (my blue mug) -- kept as numbers, not pictures." for l in lines), str(lines))
    rt.looking.presence.update(["Omar"], 1000.0)
    learning._CACHE.clear()
    check("...and who has been in view", any(l == "In view lately: Omar" for l in learning.snapshot(rt, 1001.0).get("lines", [])))
    cards = [c for c in headsup.cards(rt, 1001.0) if c["kind"] == "presence"]
    check("a card says who is here", [c["title"] for c in cards] == ["Omar is here"], str(cards))
    rt.looking.presence.update([None], 1010.0)
    rt.looking.presence.update([None], 1018.0)
    cards = [c["title"] for c in headsup.cards(rt, 1019.0) if c["kind"] == "presence"]
    check("...and a stranger gets one, saying private things are held", "Someone I don't know is in view" in cards, str(cards))
    bare = te.rig()
    check("with vision off there are no cards and no panel line", [c for c in headsup.cards(bare, 1.0) if c["kind"] == "presence"] == []
          and not any("recognise" in l for l in learning.snapshot(bare, 1.0).get("lines", [])))

    # on the runtime: the camera's own commands win over the memory's, only for what was taught
    rt.say("what have you saved")
    check("by voice: the list", rt.out[-1].startswith("I know 1 person: Omar"), rt.out[-1])
    rt.say("forget Zed")
    check("'forget Zed' (never taught) goes to the memory as it always did", "Forgotten: Zed" not in rt.out[-1] and "numbers I recognised" not in rt.out[-1], rt.out[-1])
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="forget Omar's face", addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest cannot make it forget anyone", rt.out[-1].startswith("Sorry, I only do that for Khalid") and mem.get("Omar") is not None, rt.out[-1])
    rt.say("forget Omar's face")
    check("by voice: forgotten", mem.get("Omar") is None and "numbers I recognised them by" in rt.out[-1], rt.out[-1])


class FakeReply:
    def __init__(self, text="", status=200):
        self.status_code, self._text = status, text

    def json(self):
        return {"message": {"content": self._text}}


@section
def step6_open_questions_opt_in():
    print("\n=== STEP 6: 'what does this label say?' -- a local vision model, only if switched on ===")
    import base64
    import cv2
    from src.daemon.looking import parse
    from src.vision.vlm import OllamaVision, is_local, jpeg_b64, INSTRUCTION
    ok = {"what does this label say": "read", "what's written on this": "read", "read this label": "read", "describe what you see": "describe",
          "describe this in detail": "describe", "look and tell me if the stove is on": "ask", "use the camera to check whether the door is open": "ask"}
    check("the ways people ask", all((parse(q) or (None,))[0] == k for q, k in ok.items()))
    check("ordinary talk is not taken for it", all(parse(q) is None for q in ("read this file", "what does this mean", "describe the problem", "what does this say about me")))
    check("the question is passed on as asked", parse("look and tell me if the stove is on")[1] == {"q": "if the stove is on"})

    check("only this machine: loopback addresses", is_local("http://127.0.0.1:11434/api/chat") and is_local("http://localhost:11434/x")
          and not is_local("http://192.168.1.5:11434/api/chat") and not is_local("https://api.example.com/v1") and not is_local("nonsense"))
    big = frame(size=(1080, 1920), square=((30, 30, 210), (800, 400, 300, 300)))
    b64 = jpeg_b64(big)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    check("a picture goes as one JPEG, downscaled (longest edge 768)", img is not None and max(img.shape[:2]) == 768 and img.shape[:2] == (432, 768), str(None if img is None else img.shape))
    check("...and a bad frame is None, not a crash", jpeg_b64(None) is None)

    sent = []

    def post(url, json=None, timeout=None):
        sent.append((url, json, timeout))
        return FakeReply("It says: Best before 12 March 2027.")
    v = OllamaVision("llava", "http://127.0.0.1:11434/api/chat", post=post)
    check("available with a model and a local address", v.available() and not OllamaVision("").available()
          and not OllamaVision("llava", "http://10.0.0.9:11434/api/chat").available())
    ans = v.ask(frame(size=(240, 320)), "What does it say?")
    p = sent[0][1]
    check("the answer comes back", ans == "It says: Best before 12 March 2027.")
    check("one message, one image, not streamed, low temperature, a short answer", p["model"] == "llava" and p["stream"] is False and len(p["messages"]) == 1
          and len(p["messages"][0]["images"]) == 1 and p["options"]["temperature"] <= 0.3 and p["options"]["num_predict"] <= 256)
    check("the model is told to stay with what is visible, say when it cannot tell, and never identify a person by their face",
          INSTRUCTION in p["messages"][0]["content"] and "Do not identify any person by their face" in INSTRUCTION and "What does it say?" in p["messages"][0]["content"])
    remote = OllamaVision("llava", "https://api.example.com/chat", post=post)
    n = len(sent)
    check("a remote address is refused, and nothing is sent", remote.ask(frame(), "x") is None and len(sent) == n)
    check("errors, empty answers and a bad status are None", OllamaVision("llava", post=lambda *a, **k: (_ for _ in ()).throw(OSError("down"))).ask(frame(), "x") is None
          and OllamaVision("llava", post=lambda *a, **k: FakeReply("")).ask(frame(), "x") is None
          and OllamaVision("llava", post=lambda *a, **k: FakeReply("x", 500)).ask(frame(), "x") is None)
    check("no frame is None", v.ask(None, "x") is None)

    # through Looking
    def with_vlm(reply="A white box with a blue label.", **kw):
        vl = OllamaVision("llava", post=lambda *a, **k: FakeReply(reply))
        lk = looking(**kw)
        lk.vlm = vl
        lk.cfg.open_questions = True
        return lk
    off = looking()
    say = off.hear("what's written on this", 1000.0)
    check("off by default: it says what is missing, and takes NO picture", "needs a vision model" in say and "vision.vlm_model" in say and off.eye.looks == 0, say)
    flag_only = looking()
    flag_only.cfg.open_questions = True
    check("switched on but no model named: the same", "needs a vision model" in flag_only.hear("describe what you see") and flag_only.eye.looks == 0)
    lk = with_vlm()
    say = lk.hear("describe what you see", 1000.0)
    check("on: the model's words, and that they are the model's reading", say == "A white box with a blue label. That's the vision model's reading, so check it if it matters.", say)
    check("...from one picture, kept for a minute in memory only", lk.eye.looks == 1 and lk.vlm.calls == 1 and lk.eye.recent() is not None)
    check("read", with_vlm("Best before 12 March 2027").hear("what does this label say").startswith("Best before 12 March 2027. That's the vision model's reading"))
    check("a long answer is cut", len(with_vlm("word " * 400).hear("describe what you see")) < 520)
    sent.clear()
    q = looking()
    q.vlm = OllamaVision("llava", post=post)
    q.cfg.open_questions = True
    q.hear("look and tell me if the stove is on")
    check("a free question reaches the model as asked", "if the stove is on" in sent[0][1]["messages"][0]["content"])
    check("no answer from the model: said", "couldn't get an answer" in with_vlm("").hear("describe what you see"))
    pr = with_vlm(eye_paused=True)
    check("in privacy mode nothing is looked at and nothing is sent", pr.hear("describe what you see") == "I'm paused for privacy, so I'm not looking." and pr.vlm.calls == 0)
    check("no camera: said", "no camera I can open" in with_vlm(grab=False).hear("describe what you see"))

    # the runtime
    import tst_effects as te
    from src.senses.base import Modality, SensorEvent
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    vl = OllamaVision("llava", post=lambda *a, **k: FakeReply("A cup on a desk."))
    eye = Eye(grab=lambda: frame())
    rt = te.rig(vision=True, vision_cfg=dict(open_questions=True, vlm_model="llava"),
                vision_io=dict(eye=eye, faces=FakeFaces(), objects=FakeObjects([]), memory=VisualMemory(None), vlm=vl))
    rt.brain.user = "Khalid"
    rt.say("describe what you see")
    check("by voice", rt.out[-1].startswith("A cup on a desk."), rt.out[-1])
    n = eye.looks
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="describe what you see", addressed=True, salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest's voice is refused: no picture, no model", rt.out[-1].startswith("Sorry, I only do that for Khalid") and eye.looks == n and vl.calls == 1, rt.out[-1])
    rt.t[0] += 5
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="describe what you see", addressed=False, salience=0.4, meta={}))
    rt._tick()
    check("overheard speech neither", eye.looks == n and vl.calls == 1)
    from src.daemon.looking import build
    lk = build(dict(open_questions=True, vlm_model="llava"), dict(eye=eye, faces=FakeFaces(), objects=FakeObjects([])), lambda: None, lambda: False, lambda: 1.0, None,
               ollama_url="http://127.0.0.1:4444/api/chat")
    check("built from the config: it uses the language model's own Ollama address", lk.vlm is not None and lk.vlm.url == "http://127.0.0.1:4444/api/chat" and lk.vlm.model == "llava")
    nobody = build(dict(open_questions=False, vlm_model="llava"), dict(eye=eye, faces=FakeFaces(), objects=FakeObjects([])), lambda: None, lambda: False, lambda: 1.0)
    check("...and nothing is built unless it was switched on", nobody.vlm is None and not nobody.can_open_questions())


@section
def wiring_config_control_and_installer():
    print("\n=== WIRING: off by default, the control command, and an installer that asks ===")
    import subprocess
    from src.daemon.service import DEFAULT_CONFIG
    from src.vision import models
    v = DEFAULT_CONFIG["vision"]
    check("vision is off by default, and so is everything that watches or sends", v["enabled"] is False and v["presence"] is False
          and v["announce_known"] is False and v["announce_strangers"] is False and v["open_questions"] is False and v["vlm_model"] == "")
    check("...with the private-things hold on", v["hold_when_stranger"] is True)
    src = open(os.path.join(HERE, "ashd.py"), encoding="utf-8").read()
    check("the daemon builds it from the config", 'vision=bool((cfg.get("vision") or {}).get("enabled", False))' in src and "vision_cfg=cfg.get" in src)

    # the installer's command line
    out, folder, calls = [], tempfile.mkdtemp(), []

    def fake(url, to):
        calls.append(url)
        os.makedirs(os.path.dirname(to), exist_ok=True)
        open(to, "wb").write(b"x" * 4096)
    rc = models.cli([], folder, fake, out.append)
    text = "\n".join(out)
    check("without --yes: it says what would come, from where and how big, and downloads nothing",
          rc == 0 and calls == [] and "nothing has been" in text and "opencv_zoo" in text and "37 MB" in text and "--yes" in text, text)
    check("...and that the object detector is yours to supply", "YOLOv8 ONNX" in text)
    out.clear()
    rc = models.cli(["--yes"], folder, fake, out.append)
    check("with --yes it fetches both", rc == 0 and len(calls) == 2 and out == ["installed face_detector", "installed face_recognizer"], str(out))
    out.clear()
    check("when they are there it says so", models.cli([], folder, fake, out.append) == 0 and out == ["The face models are already installed."])
    out.clear()

    def boom(url, to):
        raise OSError("no network")
    check("a failure is reported and is a non-zero exit", models.cli(["--yes"], tempfile.mkdtemp(), boom, out.append) == 1 and any(l.startswith("FAILED") for l in out))

    # the real command line, with nothing installed and no --yes: must not reach for the network
    env = dict(os.environ, ASH_STATE_DIR=tempfile.mkdtemp(), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, os.path.join(HERE, "ashctl.py"), "vision", "install"], cwd=HERE, env=env, capture_output=True, text=True, timeout=120)
    check("`ashctl vision install` runs on its own, says what it would do, and exits 0", r.returncode == 0 and "nothing has been" in r.stdout, r.stdout[-300:] + r.stderr[-200:])

    # the daemon's control command
    import tst_effects as te
    from src.daemon.service import ControlServer
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    mem = VisualMemory(None)
    mem.enroll("Omar", "person", vec(1))
    mem.enroll("my blue mug", "thing", vec(3, dim=224))
    rt = te.rig(vision=True, vision_io=dict(eye=Eye(grab=lambda: frame()), faces=FakeFaces(), objects=FakeObjects([]), memory=mem))
    cs = ControlServer.__new__(ControlServer)
    cs.runtime = rt
    got = cs.dispatch("vision")
    check("`ashctl vision` lists what it can recognise, with the status", got.get("ok") and "I know 1 person: Omar" in got["report"] and got["status"]["enabled"], str(got)[:200])
    got = cs.dispatch("vision forget Omar")
    check("...forgets one", got == {"ok": True, "report": "forgot 1 visual entry"} and mem.get("Omar") is None and mem.get("blue mug") is not None, str(got))
    got = cs.dispatch("vision forget")
    check("...or everything", got == {"ok": True, "report": "forgot 1 visual entry"} and mem.list() == [], str(got))
    bare = te.rig()
    cs.runtime = bare
    got = cs.dispatch("vision")
    check("...and says so when vision is off", got.get("ok") is False and "vision.enabled" in got.get("error", ""), str(got))


@section
def broken_model_is_not_an_empty_room():
    print("\n=== A model that breaks is said to be broken, not 'nobody there' ===")
    from src.vision.backends import FaceBackend

    class Broken(FakeFaces):
        broken = True

        def detect(self, img):
            self.calls += 1
            self.last_error = "face detection failed" if self.broken else None
            return [] if self.broken else list(self.faces)
    bf = Broken([face()], [vec(1)])
    lk = looking(faces_backend=bf)
    out = lk.hear("who is this")
    check("a failing face model is reported, not 'I don't see anyone'", "stopped working" in out and "don't see anyone" not in out, out)
    check("...and counted in the status", lk.status()["fault"] == "face model" and lk.status()["faults"] == 1, str(lk.status()))
    bf.broken = False
    out = lk.hear("who is this")
    check("once it works again the fault goes away", "stopped working" not in out and lk.status()["fault"] == "", out + str(lk.status()))
    check("the base backends carry the flag too", FaceBackend.last_error is None)
    pl = looking(faces_backend=Broken([face()], [vec(1)]), presence=True)
    pl.t[0] = 5000.0
    pl.presence.update(["Omar"], pl.t[0])
    pl.t[0] += 5
    got = pl.watch(frame(size=(480, 640)), pl.t[0])
    check("a broken reading leaves who-is-here as it was, instead of reporting everyone gone",
          got == [] and pl.presence.here(pl.t[0]) == ["Omar"], str(pl.presence.here(pl.t[0])))


@section
def enrolment_refuses_a_bad_look():
    print("\n=== Learning a face needs a good look: small, dim, blurry or cut-off ones are refused ===")
    from src.vision.backends import Face, face_quality
    sharp = frame(color=(120, 120, 120), noise=40, size=(480, 640))
    big = face((200, 100, 160, 200))

    check("a big, lit, detailed, centred face is fine", face_quality(sharp, big) == "", face_quality(sharp, big))
    check("a small face: move closer", "move closer" in face_quality(sharp, face((300, 200, 40, 50))))
    check("a face cut by the edge: move to the middle", "cut off" in face_quality(sharp, face((560, 100, 160, 200))))
    check("a low detector score: not a clear view", "clear view" in face_quality(sharp, Face(box=(200, 100, 160, 200), score=0.5, row=np.zeros(15, np.float32))))
    check("a dim face", "dim" in face_quality(frame(color=(20, 20, 20), noise=15), big))
    check("a washed-out face", "washed out" in face_quality(frame(color=(245, 245, 245), noise=8), big))
    check("a blurry (flat) face", "blurry" in face_quality(frame(color=(120, 120, 120)), big))
    check("an unreadable picture", "not readable" in face_quality(None, big))

    lk = looking(faces=[big], vectors=[vec(1)], enrol_checks=True, frames=[frame(color=(120, 120, 120))])
    out = lk.hear("remember this person as Omar")
    check("teaching from a blurry look is refused, with what to change", "blurry" in out and "bad look" in out and lk.memory.get("Omar") is None, out)
    lk = looking(faces=[big], vectors=[vec(1)], enrol_checks=True, frames=[sharp])
    out = lk.hear("remember this person as Omar")
    check("a good look is learned", lk.memory.get("Omar") is not None, out)
    lk = looking(faces=[big], vectors=[vec(1)], enrol_checks=False, frames=[frame(color=(120, 120, 120))])
    lk.hear("remember this person as Omar")
    check("...and the check can be switched off", lk.memory.get("Omar") is not None)
    lk = looking(faces=[big], vectors=[vec(1)], enrol_checks=True, frames=[frame(color=(120, 120, 120))])
    lk.memory.enroll("Omar", "person", vec(1))
    check("recognising is never blocked by it", "Omar" in lk.hear("who is this"), lk.hear("who is this"))


@section
def models_are_never_used_by_two_threads_at_once():
    print("\n=== The presence worker and a spoken question never use the models at the same time ===")
    import threading as th
    import time as tm

    class Probe(FakeFaces):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.inside, self.worst = 0, 0
            self.guard = th.Lock()

        def detect(self, img):
            with self.guard:
                self.inside += 1
                self.worst = max(self.worst, self.inside)
            tm.sleep(0.01)
            with self.guard:
                self.inside -= 1
            return list(self.faces)
    pf = Probe([face()], [vec(1)])
    lk = looking(faces_backend=pf, presence=True)
    errs = []

    def work(fn):
        try:
            for _ in range(15):
                fn()
        except Exception as e:                       # noqa: BLE001
            errs.append(repr(e))
    ts = [th.Thread(target=work, args=(lambda: lk.watch(frame(size=(480, 640)), lk.t[0]),)),
          th.Thread(target=work, args=(lambda: lk.hear("who is this"),))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    check("never two detections at once", pf.worst == 1, f"worst={pf.worst}")
    check("...and no errors from either side", not errs, str(errs))


@section
def doctor_says_whether_it_will_work():
    print("\n=== `ashctl vision doctor`: one picture, each model tested, nothing kept ===")
    lk = looking(things=[thing("cup", 0.9)], faces=[face((200, 100, 160, 200))], vectors=[vec(1)])
    out = lk.doctor()
    check("a healthy setup: camera, faces and objects each reported with timing", all(x in out for x in ("camera: ok, 640x480", "1 face(s) in", "face description: ok", "1 object(s) in")), out)
    check("...and the doctor's picture is not kept for 'save that photo'", lk.eye.recent() is None)
    check("...nothing was learned", lk.memory.list() == [])
    check("it says what teaching would complain about", "teaching would say" not in out, out)       # checks are off in this helper
    lk = looking(faces=[face((200, 100, 160, 200))], vectors=[vec(1)], enrol_checks=True)
    check("with checks on, a flat grey test picture is flagged as blurry", "teaching would say: the picture is blurry" in lk.doctor(), lk.doctor())
    out = looking(eye_paused=True, faces=[face()], vectors=[vec(1)]).doctor()
    check("paused for privacy: it does not look", "paused for privacy" in out and "face detection" not in out, out)
    out = looking(grab=False, faces=[face()], vectors=[vec(1)]).doctor()
    check("no camera: says so", "could not open a camera" in out, out)
    out = looking(faces_backend=FakeFaces(available=False), objects=FakeObjects([], available=False)).doctor()
    check("no usable model: says how to get one, and takes no picture", "NOT usable" in out and "Nothing to test" in out and "camera:" not in out, out)

    class Broken(FakeFaces):
        def detect(self, img):
            self.last_error = "face detection failed"
            return []
    out = looking(faces_backend=Broken([face()], [vec(1)])).doctor()
    check("a model that errors is reported as FAILED", "face detection: FAILED" in out, out)
    out = looking(faces_backend=FakeFaces([], [])).doctor()
    check("an empty room: tells you to step in front of the camera", "no face in this picture" in out, out)


def bg_runtime(faces, **vcfg):
    """A runtime whose presence reading runs on the worker thread, with a camera sensor stub holding a fresh frame."""
    import tst_effects as te
    from src.senses.base import State
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory
    import time as _t
    io = dict(eye=Eye(grab=lambda: frame()), faces=faces, objects=FakeObjects([]), memory=VisualMemory(None))
    rt = te.rig(vision=True, vision_io=io, vision_cfg=dict({"presence": True, "presence_interval_s": 1.0, "enrol_checks": False}, **vcfg))
    cam = type("Cam", (), {"due": lambda self, now: False, "name": "camera", "close": lambda self: None,
                           "status": lambda self: {}, "disable": lambda self: None, "enable": lambda self: None})()
    cam.state, cam.last_frame, cam.last_frame_ts = State.ACTIVE, frame(), _t.time()
    rt.periphery.sensors["camera"] = cam
    rt.cam = cam
    return rt


@section
def stopping_does_not_wait_on_a_stuck_detector():
    print("\n=== Stopping the daemon never waits on a detector that is stuck ===")
    import threading as th
    import time as tm
    release = th.Event()
    stuck = FakeFaces([face()], [vec(1)])
    stuck.detect = lambda fr: (release.wait(10), [])[1]
    rt = bg_runtime(stuck)
    rt.t[0] += 5
    rt.cam.last_frame_ts = tm.time()
    rt._tick()
    tm.sleep(0.2)
    check("a reading is in flight and stuck", rt._presence_job is not None and not rt._presence_job.done())
    t0 = tm.time()
    rt.stop(timeout=1.0)
    check("stop() returns promptly and lets go of the worker", tm.time() - t0 < 3.0 and rt._presence_exec is None, f"{tm.time() - t0:.2f}s")
    release.set()


@section
def a_stuck_reading_is_abandoned_and_flagged():
    print("\n=== A reading that never finishes is abandoned, flagged, and does not end presence for good ===")
    import threading as th
    import time as tm
    release = th.Event()
    mode = {"stuck": True}
    f = FakeFaces([face()], [vec(1)])
    real = f.detect
    f.detect = lambda fr: (release.wait(10), real(fr))[1] if mode["stuck"] else real(fr)
    rt = bg_runtime(f, presence_timeout_s=0.3, busy_wait_s=0.2)
    rt.looking.memory.enroll("Omar", "person", vec(1))
    rt.t[0] += 5
    rt.cam.last_frame_ts = tm.time()
    rt._tick()
    tm.sleep(0.5)
    check("a question meanwhile is told to ask again, not left hanging", "still working" in (rt.looking.hear("who is this") or ""))
    rt.t[0] += 5
    rt.cam.last_frame_ts = tm.time()
    rt._tick()
    check("past the timeout the reading is abandoned and counted", rt.presence_stuck == 1 and rt.status()["vision"]["presence_stuck"] == 1, str(rt.status()["vision"]))
    mode["stuck"] = False
    release.set()
    for _ in range(40):
        rt.t[0] += 5
        rt.cam.last_frame_ts = tm.time()
        rt._tick()
        if rt._presence_job is not None:
            try:
                rt._presence_job.result(3)
            except Exception:
                pass
        if rt.looking.presence.here(rt.t[0]):
            break
        tm.sleep(0.1)
    check("once it frees up, presence works again", rt.looking.presence.here(rt.t[0]) == ["Omar"], str(rt.status()["vision"]))
    rt.stop(timeout=1.0)


if __name__ == "__main__":
    for fn in SECTIONS:
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
