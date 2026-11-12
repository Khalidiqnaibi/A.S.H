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


if __name__ == "__main__":
    for fn in SECTIONS:
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
