"""
tst_voicectl.py -- controlling ASH by voice.

Phrasings from the session of 2026-11-06, where every one of these went to
the language model, which answered as if it had done them.

  V1  Reading the commands -- including whisper's "CBU" -- and leaving
      ordinary sentences alone
  V2  Doing them: privacy, the mic, the camera, screen capture, quiet mode,
      a face, the list of faces, the real status
  V3  Who may: turning things off from any line, on only from a line meant
      for ASH; nothing back on while privacy mode is on
  V4  Shutting down: asked first, done on a yes, said plainly if it can't be
  V5  A model answer that claims one of these is replaced by the truth
  V6  Quiet mode looks sleepy

Run:  PYTHONIOENCODING=utf-8 python tst_voicectl.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import voicectl  # noqa: E402
from src.daemon.voicectl import claimed_action, parse  # noqa: E402

check = tw.check
T0 = tw.T0


def v1_parse():
    print("\n=== V1. Reading the commands ===")
    cases = {
        "privacy mode": "privacy_on", "Go to privacy mode.": "privacy_on", "Go to privacy mode ash": "privacy_on",
        "go private": "privacy_on", "privacy off": "privacy_off", "resume": "privacy_off",
        "Turn off mic": "mic_off", "stop listening": "mic_off", "turn on the mic": "mic_on",
        "activate camera": "camera_on", "turn the camera off": "camera_off",
        "Turn on the screen capture": "screen_on", "turn off screen capture": "screen_off",
        "Go to sleep ash": "sleep", "quiet mode": "sleep", "wake up": "wake",
        "Ash turn off.": "shutdown", "shut down": "shutdown",
        "what faces can you make?": "faces", "show me your expressions": "faces",
        "Ash, what's your status?": "status", "What's your CBU status as?": "status", "system status": "status",
        "how's your memory usage": "status",
    }
    for text, want in cases.items():
        got = parse(text)
        check(f"{text!r} -> {want}", got is not None and got.name == want, str(got))
    for text, anim in (("Smile for me, Ash.", "happy"), ("look surprised", "surprised"), ("make a sad face", "sad"),
                       ("wink at me", "wink"), ("give me a big smile", "very_happy"), ("look sleepy", "sleepy")):
        got = parse(text)
        check(f"{text!r} -> face {anim}", got is not None and got.name == "face" and got.arg == anim, str(got))
    for text in ("what is privacy", "I turned off the mic yesterday", "the camera on my phone", "look at this",
                 "stop listening to music", "what's your status on the report", "turn off the lights", "look it up",
                 "be careful", "", None, "privacy mode " * 20, "privacy mode" + ", please" * 12):
        check(f"not a command: {str(text)[:30]!r}", parse(text) is None, str(parse(text)))


class Sensor:
    def __init__(self, name, on=False):
        from src.senses.base import State
        self.S = State
        self.name, self.on = name, on
        self.state = State.ACTIVE if on else State.DISABLED

    def enable(self):
        self.on, self.state = True, self.S.ACTIVE

    def disable(self):
        self.on, self.state = False, self.S.DISABLED

    def due(self, now):
        return False


class Player:
    def __init__(self):
        self.library = {n: object() for n in ("happy", "sad", "wink", "idle", "boot", "surprised", "very_happy")}
        self.played = []
        self.idle = "idle"

    def play(self, name, force=False):
        self.played.append((name, force))
        return True

    def set_idle(self, name):
        self.idle = name


def rig(reply="Sure."):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    per = PeripheryRegistry()
    for name in ("microphone", "camera", "screenshot"):
        per.sensors[name] = Sensor(name, on=(name == "microphone"))

    class Ash(tw.Ash):
        def think(self, query, image=None, context=""):
            self.calls.append(query)
            r = tw.Resp(reply)
            r.pathway = type("P", (), {"value": "slow"})()
            return r
    ash = Ash()
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9),
                        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    from src.face.bridge import FaceBridge
    rt.face = FaceBridge(Player())

    def say(text, addressed=True, source="microphone", meta=None):
        clock[0] += 3
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6, meta=meta or {}))
        rt._tick()
    rt.say, rt.out, rt.ash_, rt.t = say, out, ash, clock
    return rt


def v2_doing():
    print("\n=== V2. Doing them ===")
    rt = rig()
    rt.say("Go to privacy mode ash")
    check("privacy mode on", rt.privacy.paused and out_has(rt, "Privacy mode on"), str(rt.out[-1:]))
    check("no model asked", rt.ash_.calls == [])
    rt.say("privacy off", source="console")
    check("and off", not rt.privacy.paused and rt.out[-1].startswith("Privacy mode off"))
    rt.say("turn off the mic")
    check("mic off", not rt.periphery.sensors["microphone"].on and rt.out[-1].startswith("Microphone off."))
    rt.say("turn on the mic", source="console")
    check("mic on (typed)", rt.periphery.sensors["microphone"].on and rt.out[-1] == "Microphone on.")
    rt.say("activate camera")
    check("camera on", rt.periphery.sensors["camera"].on and rt.out[-1] == "Camera on.")
    rt.say("Turn on the screen capture")
    check("screen capture on", rt.periphery.sensors["screenshot"].on and rt.out[-1] == "Screen capture on.")
    rt.say("turn off screen capture")
    check("and off", not rt.periphery.sensors["screenshot"].on)
    rt.say("Go to sleep ash")
    check("quiet mode", rt.gate.muted and "wake up" in rt.out[-1])
    rt.say("wake up")
    check("and back", not rt.gate.muted and rt.out[-1] == "I'm back.")
    rt.say("Smile for me, Ash.")
    check("a smile, really", rt.face.player.played[-1] == ("happy", True) and rt.out[-1] == "Like this.")
    rt.say("what faces can you make?")
    check("the faces listed, not the plumbing", rt.out[-1].startswith("I can look happy, sad, surprised, very happy and wink")
          and "boot" not in rt.out[-1] and "idle" not in rt.out[-1], rt.out[-1])
    rt.say("What's your CBU status as?")
    s = rt.out[-1]
    check("the real status", s.startswith("CPU at ") and "memory at" in s and "using microphone, camera" in s, s)
    check("and still no model", rt.ash_.calls == [])
    rt.face = None
    rt.say("smile")
    check("no face running: said so", rt.out[-1] == "My face isn't running right now.")
    rt2 = rig()
    del rt2.periphery.sensors["camera"]
    rt2.say("turn on the camera")
    check("no camera: said so", rt2.out[-1] == "There's no camera set up on this machine.")
    check("status counted", rt.stats["voice_commands"] >= 10)
    check("status line when paused and quiet", voicectl.status_line(None, [], True, True)
          == "Privacy mode is on; I'm in quiet mode.")


def out_has(rt, s):
    return any(s in o for o in rt.out)


def v3_who():
    print("\n=== V3. Who may ===")
    rt = rig()
    rt.say("privacy mode", addressed=False)
    check("overheard 'privacy mode': still done (it only turns things off)", rt.privacy.paused)
    rt.privacy.resume()
    rt.say("turn off the mic", addressed=False)
    check("overheard 'turn off the mic': done", not rt.periphery.sensors["microphone"].on)
    rt.say("activate camera", addressed=False)
    check("overheard 'activate camera': not done", not rt.periphery.sensors["camera"].on)
    rt.say("activate camera", addressed=False, meta={"address": {"likely": True}})
    check("probably said to ASH: done", rt.periphery.sensors["camera"].on)
    rt.say("turn off the camera")
    rt.say("privacy mode")
    rt.say("turn on the camera")
    check("nothing on while privacy mode is on", not rt.periphery.sensors["camera"].on
          and "Privacy mode is on" in rt.out[-1], rt.out[-1])


def v4_shutdown():
    print("\n=== V4. Shutting down ===")
    rt = rig()
    asked = []
    rt.request_shutdown = lambda: asked.append(1)
    rt.say("Ash turn off.")
    check("asked first", rt.out[-1] == "Shut ASH down completely? Say yes to confirm." and asked == [])
    rt.say("yes")
    check("a yes does it", asked == [1] and rt.out[-1] == "Shutting down. Bye.")
    rt.say("shut down")
    rt.say("what time is it")
    rt.say("yes")
    check("anything else in between: the question is dropped", asked == [1])
    rt.say("shut down")
    rt.t[0] += 60
    rt.say("yes")
    check("a yes too late: not shut down", asked == [1])
    rt.say("shut down")
    rt.say("yes", addressed=False)
    check("an overheard yes: not shut down", asked == [1])
    rt2 = rig()
    rt2.say("shut down")
    rt2.say("yes")
    check("can't from here: said so", "can't switch myself off" in rt2.out[-1])


def v5_claims():
    print("\n=== V5. Claimed actions ===")
    for t in ("I'm turning off now. Good night.", "As I go to sleep, I'll continue running in the background.",
              "Go to privacy mode.", "I have turned on the camera for you.", "I'm switching off the microphone."):
        check(f"a claim: {t[:40]!r}", claimed_action(t))
    for t in ("I didn't go to privacy mode at all because there was no trigger.", "You can turn off the camera from "
              "the console.", "The sun is turning off.", "I'll smile more often.", "Alas I go to sleepwalking", ""):
        check(f"not a claim: {t[:40]!r}", not claimed_action(t))
    rt = rig(reply="I'm turning off now. Good night.")
    rt.say("ash are you tired")
    check("a model answer claiming it is replaced by the truth", rt.out[-1] == voicectl.HONEST, rt.out[-1])
    rt.response_sink("I'm turning off the camera now.", {"addressed": True, "pathway": "wonder"})
    check("a line from a rule (not the model) is left alone", rt.out[-1] == "I'm turning off the camera now.")
    rt = rig(reply="It is 5 pm.")
    rt.say("ash what time is it")
    check("an ordinary answer is left alone", rt.out[-1] == "It is 5 pm.")


def v6_face_quiet():
    print("\n=== V6. Quiet mode looks sleepy ===")
    from src.face.bridge import attach_to_daemon
    rt = rig()
    p = Player()
    attach_to_daemon(rt, p)
    rt.gate.mute(True)
    rt._social_step(rt.t[0])
    check("muted: sleepy", p.idle == "sleepy")
    rt.gate.mute(False)
    rt._social_step(rt.t[0])
    check("unmuted: awake", p.idle == "idle")


if __name__ == "__main__":
    for fn in (v1_parse, v2_doing, v3_who, v4_shutdown, v5_claims, v6_face_quiet):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
