"""
tst_photobooth.py -- the fun camera. Frames are synthetic and the frame source is a fake: no camera is ever opened.

  B1  the ways people ask, the filter named in it, and what is a plain photo (left to media.py)
  B2  every filter: same size, really changed, a valid image file written
  B3  the countdown, then one frame, filtered, saved in the photos folder; the plain frame is not kept
  B4  in the runtime: not in privacy mode, not for a guest, not for overheard speech; "send it to my phone" sends it

Run:  PYTHONIOENCODING=utf-8 python tests/tst_photobooth.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.photobooth import FILTERS, PhotoBooth, apply, filter_of  # noqa: E402

check = tw.check


def face_frame():
    rng = np.random.RandomState(0)
    f = np.full((240, 320, 3), (90, 140, 200), np.uint8)
    f[60:200, 110:210] = (60, 90, 170)
    f[100:120, 130:150] = 20
    f[100:120, 170:190] = 20
    return np.clip(f.astype(int) + rng.randint(-12, 13, f.shape), 0, 255).astype(np.uint8)


def b1_asking():
    print("\n=== B1. Asking ===")
    for said, want in [("take a silly photo", ""), ("Ash, photo booth!", ""), ("let's take a fun picture", ""),
                       ("take a photo with the cartoon filter", "cartoon"), ("take a picture of me in sketch style", "sketch"),
                       ("make a comic of me", "comic"), ("take a selfie with the bmo frame", "bmo"),
                       ("take a photo with a retro effect", "retro"), ("take a picture with the mirror filter", "mirror")]:
        check(f"{said!r} -> {want!r}", filter_of(said) == want, str(filter_of(said)))
    for said in ("take a photo", "take a picture of me", "a silly question", "photo of the booth", "take a photo with Omar"):
        check(f"not the booth: {said!r}", filter_of(said) is None, str(filter_of(said)))


def b2_filters():
    print("\n=== B2. Filters ===")
    import cv2
    f = face_frame()
    d = tempfile.mkdtemp()
    for name in FILTERS:
        out = apply(f, name)
        changed = float(np.abs(out.astype(int) - f.astype(int)).mean())
        p = os.path.join(d, name + ".jpg")
        ok = cv2.imwrite(p, out) and cv2.imread(p) is not None
        check(f"{name}: same size, changed, a real image", out.shape == f.shape and out.dtype == np.uint8 and changed > 3 and ok, f"{changed:.1f}")
    m = apply(f, "mirror")
    check("mirror: the right half is the left half flipped", np.array_equal(m[:, 160:], f[:, :160][:, ::-1]))


def b3_take():
    print("\n=== B3. Taking one ===")
    d = tempfile.mkdtemp()
    b = PhotoBooth(grab=face_frame, folder=d, clock=lambda: 1_800_000_000.0)
    check("the countdown names the filter", b.countdown("cartoon") == "Photo booth! A cartoon one. Three... two... one... cheese!")
    out = b.take("sketch")
    files = os.listdir(d)
    check("saved, filtered, in the photos folder", out.startswith("Got it! Saved as booth_") and files == [os.path.basename(b.last_path)]
          and files[0].endswith("_sketch.jpg"), f"{out} {files}")
    check("only the filtered picture is kept (one file)", len(files) == 1)
    check("a random filter when none was named", b.pick("") in FILTERS and b.pick("cartoon") == "cartoon")
    none = PhotoBooth(grab=lambda: None, folder=d)
    check("no camera: says so", none.take("cartoon").startswith("I couldn't get a picture"))


def b4_runtime():
    print("\n=== B4. In the runtime ===")
    import tst_effects as te
    from tst_phone import FakeCLI
    from src.senses.base import Modality, SensorEvent
    d = tempfile.mkdtemp()
    grabs = []
    cli = FakeCLI()
    rt = te.rig(photobooth=True, photobooth_dir=d, photobooth_delay_s=0, photobooth_grab=lambda: grabs.append(1) or face_frame(),
                phone=True, phone_background=False, phone_io={"cli": "kdeconnect-cli", "runner": cli})
    n = len(rt.ash.calls)
    rt.say("Ash, take a silly photo")
    check("countdown, then saved -- by rule, no model", any("cheese" in o for o in rt.out[-2:]) and rt.out[-1].startswith("Got it!")
          and len(os.listdir(d)) == 1 and len(rt.ash.calls) == n, str(rt.out[-2:]))
    rt.say("Ash, send it to my phone")
    check("'send it to my phone' sends the photo", cli.calls[-1][-2] == "--share" and cli.calls[-1][-1].endswith(".jpg"), str(cli.calls[-1]))
    k = len(grabs)
    rt.privacy.pause("test")
    rt.say("Ash, take a silly photo")
    check("privacy mode: no picture", len(grabs) == k and "privacy" in rt.out[-1], rt.out[-1])
    rt.privacy.resume()
    rt.t[0] += 3
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text="Ash, take a silly photo", addressed=True,
                          salience=0.9, urgency=0.6, meta={"guest": True}))
    rt._tick()
    check("a guest: no picture", len(grabs) == k)
    rt.say("take a silly photo", addressed=False)
    check("overheard: no picture", len(grabs) == k)
    check("status", rt.status()["photobooth"]["taken"] == 1)


if __name__ == "__main__":
    for fn in (b1_asking, b2_filters, b3_take, b4_runtime):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
