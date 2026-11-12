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


if __name__ == "__main__":
    for fn in SECTIONS:
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
