"""
tst_media.py -- music, sounds and photos.

  M1  Music: each command to its media key, and what is not one
  M2  Photos: taken on request, named by time, remembered; not while paused;
      a camera that fails is said plainly
  M3  Sounds: a chime on cue, and the switch
  M4  The runtime: music and photos without the model; a guest cannot take a
      photo; a reminder chimes
  M5  The real executor does not crash here (no keys pressed, no camera opened)

Run:  PYTHONIOENCODING=utf-8 python tst_media.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.media import Media, MediaConfig, MediaExecutor  # noqa: E402

check = tw.check
T0 = time.mktime((2026, 3, 10, 14, 5, 9, 0, 0, -1))


class FakeEx(MediaExecutor):
    def __init__(self):
        self.keys, self.chimes, self.photos = [], [], []
        self.camera_ok = True

    def key(self, name):
        self.keys.append(name)
        return True

    def chime(self, kind):
        self.chimes.append(kind)
        return True

    def photo(self, path, index=0):
        if not self.camera_ok:
            return False
        self.photos.append(path)
        return True


class Epi:
    def __init__(self):
        self.added = []

    def add_episode(self, summary, event_type, related_entities=None, importance=0.5):
        self.added.append((summary, event_type))


def mrig(**cfg):
    ex, ep = FakeEx(), Epi()
    m = Media(MediaConfig.from_dict({"photo_dir": tempfile.mkdtemp(), **cfg}), executor=ex, episodic=ep,
              clock=lambda: T0)
    m.ex_, m.ep_ = ex, ep
    return m


def m1_music():
    print("\n=== M1. Music ===")
    m = mrig()
    for text, key, say in (("pause the music", "play_pause", "Paused."), ("pause", "play_pause", "Paused."),
                           ("play", "play_pause", "Playing."), ("resume the music", "play_pause", "Playing."),
                           ("next song", "next", "Next one."), ("skip this track", "next", "Next one."),
                           ("previous track", "previous", "Going back."), ("turn it up", "volume_up", "Louder."),
                           ("volume down", "volume_down", "Quieter."), ("mute the sound", "mute", "Done.")):
        r = m.hear(text)
        check(f"{text!r} -> {key}", r == say and m.ex_.keys[-1] == key, str(r))
    n = len(m.ex_.keys)
    for text in ("play the number game", "pause for a second and think", "next week", "turn it up to 11 in the pitch",
                 "", None, "pause " * 30):
        check(f"not music: {str(text)[:30]!r}", m.hear(text) is None)
    check("(no keys pressed for those)", len(m.ex_.keys) == n)
    check("music off: not taken", mrig(music=False).hear("pause the music") is None)


def m2_photos():
    print("\n=== M2. Photos ===")
    m = mrig()
    r = m.hear("take a photo")
    check("taken and saved by time", r == f"Got it -- saved as 2026-03-10_140509.jpg in {os.path.basename(m.cfg.photo_dir)}."
          and m.ex_.photos[-1].endswith("2026-03-10_140509.jpg"), r)
    check("and remembered", m.ep_.added and "take a photo" in m.ep_.added[-1][0] and m.ep_.added[-1][1] == "observation")
    for text in ("can you take a picture of me", "snap a selfie", "hey ash, take a pic please"):
        check(f"also: {text!r}", m.hear(text) is not None and len(m.ex_.photos) >= 2)
    for text in ("I took a photo yesterday", "take a look at this", "photo of the cat"):
        check(f"not a request: {text!r}", m.hear(text) is None)
    check("not while paused", m.hear("take a photo", privacy_paused=True) == "I'm paused for privacy, so no photos right now.")
    m = mrig()
    m.ex_.camera_ok = False
    check("a camera that fails is said plainly", m.hear("take a photo") == "I couldn't get a picture from the camera."
          and m.ep_.added == [])
    check("photos off: not taken", mrig(photos=False).hear("take a photo") is None)


def m3_sounds():
    print("\n=== M3. Sounds ===")
    m = mrig()
    check("a chime on cue", m.cue("reminder") and m.ex_.chimes == ["reminder"])
    check("sounds off: none", not mrig(sounds=False).cue("reminder"))


def m4_runtime():
    print("\n=== M4. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    rt = AmbientRuntime(
        ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, media=True,
                             media_cfg={"photo_dir": tempfile.mkdtemp()}, reminders=True),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    rt.reminders.clock = lambda: clock[0]
    ex = FakeEx()
    rt.media.ex = ex

    def say(text, **meta):
        clock[0] += 2
        rt.submit(SensorEvent(source="console" if not meta else "microphone", modality=Modality.TEXT, text=text,
                              addressed=True, salience=0.9, urgency=0.6, meta=meta))
        rt._tick()

    say("pause the music")
    check("music without the model", out[-1][0] == "Paused." and ex.keys == ["play_pause"] and ash.calls == [])
    say("take a photo")
    check("a photo without the model", out[-1][0].startswith("Got it -- saved as") and len(ex.photos) == 1)
    rt.privacy.pause("test")
    say("take a photo")
    check("not while privacy is paused", len(ex.photos) == 1)
    rt.privacy.resume()
    say("take a photo", guest=True)
    check("a guest cannot take a photo", len(ex.photos) == 1 and "only do that for" in out[-1][0], str(out[-1]))
    say("set a timer for 1 minutes")
    clock[0] += 61
    rt._tick()
    check("a reminder going off chimes", ex.chimes == ["reminder"] and out[-1][0] == "Your 1-minute timer is done.")
    check("status carries media", rt.status()["media"]["photos"] == 1)


def m5_executor():
    print("\n=== M5. The real executor ===")
    ex = MediaExecutor()
    check("an unknown key is refused, not pressed", ex.key("nope") is False)
    check("a camera index that does not exist is a plain False",
          ex.photo(os.path.join(tempfile.mkdtemp(), "x.jpg"), index=99) is False)


if __name__ == "__main__":
    for fn in (m1_music, m2_photos, m3_sounds, m4_runtime, m5_executor):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
