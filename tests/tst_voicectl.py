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
  V8  A model answer that states whether the mic or camera is on is checked against the real state
  V7  Several devices in one line ("Mic off, camera off.", "turn off the mic and the camera", "cam off"), each with its own direction

Run:  PYTHONIOENCODING=utf-8 python tests/tst_voicectl.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def v2b_timed():
    print("\n=== V2b. For a while ===")
    P = voicectl.parse
    check("'mute microphone for 10 minutes' is a timed mic off", P("Mute microphone for 10 minutes.") == voicectl.Command("mic_off", seconds=600.0))
    check("'privacy mode until 10 minutes' (as it was heard)", P("Ash, privacy mode until 10 minutes.").seconds == 600.0)
    check("'mic off for ten minutes' (a number word)", P("mic off for ten minutes").seconds == 600.0)
    check("'camera off for an hour'", P("turn off the camera for an hour").seconds == 3600.0)
    check("turning something ON for a while is not a thing", P("turn on the camera for ten minutes") is None)
    check("without a time it is as before", P("mute the mic").seconds == 0.0)
    rt = rig()
    rt.say("Mute microphone for 10 minutes")
    check("the mic is off, and it says when it is back", not rt.periphery.sensors["microphone"].on and "turn it back on in 10 minutes" in rt.out[-1], rt.out[-1])
    check("no model was asked", rt.ash_.calls == [])
    rt._timed_restore(rt.t[0] + 300)
    check("five minutes later it is still off", not rt.periphery.sensors["microphone"].on)
    rt._timed_restore(rt.t[0] + 601)
    check("after ten it is back on by itself", rt.periphery.sensors["microphone"].on)
    rt.say("Ash, privacy mode for 10 minutes")
    check("privacy mode for a while", rt.privacy.paused and "turn it back on in 10 minutes" in rt.out[-1], rt.out[-1])
    rt._timed_restore(rt.t[0] + 700)
    check("and it ends", not rt.privacy.paused)


def v2c_faces():
    print("\n=== V2c. Faces said the way people say them ===")
    P = voicectl.parse
    for t, anim in [("Can you do a crying face?", "sad"), ("do a laughing face", "very_happy"), ("make a furious face", "angry"),
                    ("show me a puzzled face", "confused"), ("do a sad face", "sad")]:
        check(f"{t!r} -> {anim}", P(t) == voicectl.Command("face", anim), str(P(t)))
    rt = rig()
    rt.say("Can you do a crying face?")
    check("a face", rt.face.player.played[-1] == ("sad", True), str(rt.face.player.played[-1:]))
    rt.say("Again.")
    check("'Again.' does the same face again, by rule", rt.face.player.played[-1] == ("sad", True) and len(rt.face.player.played) == 2 and rt.ash_.calls == [],
          str(rt.face.player.played) + str(rt.ash_.calls))
    rt.t[0] += 200
    rt.say("Again.")
    check("a few minutes later 'Again.' is not about a face", len(rt.face.player.played) == 2)
    # 2026-11-20: these went to the model, which said "I can't move my face"
    for t, anim in [("Wink using your face.", "wink"), ("Link using your face.", "wink"), ("Cry for me, Ash.", "sad"), ("Cry.", "sad"),
                    ("laugh for me", "very_happy"), ("can you cry", "sad"), ("frown at me", "angry")]:
        check(f"{t!r} -> {anim}", P(t) == voicectl.Command("face", anim), str(P(t)))
    for t in ("I want to cry", "the baby cried all night", "link the file to the folder", "why do people cry", "crying is fine"):
        check(f"{t!r} is not a face request", P(t) is None, str(P(t)))
    n = len(rt.ash_.calls)
    rt.say("Cry for me, Ash.")
    check("in the runtime it is a face, with no model", rt.face.player.played[-1] == ("sad", True) and len(rt.ash_.calls) == n, str(rt.face.player.played[-1:]))


def out_has(rt, s):
    return any(s in o for o in rt.out)


def v3_who():
    print("\n=== V3. Who may ===")
    rt = rig()
    n_out = len(rt.out)
    rt.say("Ash, resume")
    check("'resume' with privacy off is not 'privacy off' (it is the music's)", not any("Privacy mode off" in o for o in rt.out[n_out:]),
          str(rt.out[n_out:]))
    rt.say("privacy mode")
    rt.say("resume")
    check("...but with privacy on it still ends it", not rt.privacy.paused)
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
    import threading
    rt = rig()
    rt.request_shutdown = lambda: None
    rt.say("Ash turn off.")
    rt._thinking, rt._respond_thread = True, threading.current_thread()          # a slow answer is still being worked out
    n = len(rt.out)
    rt.say("yes")
    rt.response_sink("You've shut down all the applications.", {"addressed": True, "pathway": "slow"})
    check("an answer still being worked out is not said after 'Bye.'", rt.out[n:] == ["Shutting down. Bye."], str(rt.out[n:]))
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
    rt.t[0] += 25
    n_out = len(rt.out)
    rt._tick()
    check("unanswered for 20 s: ONE reminder", any("Still want me to shut down" in o for o in rt.out[n_out:]), str(rt.out[n_out:]))
    n_out = len(rt.out)
    rt._tick()
    check("...not on every tick", not any("Still want me" in o for o in rt.out[n_out:]))
    rt.t[0] += 17
    rt.say("ye'")
    check("a 'ye' 42 seconds later (with a stray apostrophe) still confirms: the window is 90 s", asked == [1, 1], str(asked))
    asked.clear()
    asked.append(1)
    rt.say("shut down")
    rt.t[0] += 120
    rt.say("yes")
    check("a yes too late: not shut down", asked == [1], str(asked))
    rt.say("shut down")
    rt.say("yes", addressed=False)
    check("an overheard yes: not shut down", asked == [1])
    rt.say("yes")
    asked.clear()
    rt.say("shut down")
    rt.say("did you see the match last night", addressed=False)
    rt.say("yes")
    check("a line from the room between the question and the yes does not cancel it", asked == [1], str(asked))
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


def v7_many():
    print("\n=== V7. Several devices in one line ===")
    from src.daemon.voicectl import parse_many
    names = lambda q: [c.name for c in parse_many(q)]          # noqa: E731
    for said, want in [("Mic off, camera off.", ["mic_off", "camera_off"]), ("Switch the camera off and the mic off.", ["camera_off", "mic_off"]),
                       ("turn off the mic and the camera", ["mic_off", "camera_off"]), ("mic and camera off", ["mic_off", "camera_off"]),
                       ("cam off", ["camera_off"]), ("Mic and camera of ASH.", ["mic_off", "camera_off"]), ("turn off mic and turn on camera", ["mic_off", "camera_on"]),
                       ("turn the mic off and the camera on", ["mic_off", "camera_on"]), ("please turn the camera and the mic off", ["camera_off", "mic_off"]),
                       ("turn on the camera", ["camera_on"]), ("mic off mic off", ["mic_off"])]:
        check(f"{said!r}", names(said) == want, str(names(said)))
    for said in ("what does the camera do on mondays", "is the camera on?", "tell me about the camera", "mic", "the camera is broken", "I like my mic",
                 "turn off the lights", "camera", "my camera is on top of the monitor", "how do I turn the mic off", "what can you do then", "Come it off."):
        check(f"{said!r} is not a device command", names(said) == [], str(names(said)))
    rt = rig()
    rt.periphery.sensors["camera"].on = True
    rt.say("Mic off, camera off.")
    check("both go off in one answer, with no model", not rt.periphery.sensors["microphone"].on and not rt.periphery.sensors["camera"].on
          and rt.out[-1].startswith("Microphone off. Camera off.") and rt.ash_.calls == [], str(rt.out[-1:]))
    rt.say("turn on the camera and the mic", source="console")
    check("both back on (typed)", rt.periphery.sensors["microphone"].on and rt.periphery.sensors["camera"].on and "Camera on." in rt.out[-1] and "Microphone on." in rt.out[-1], rt.out[-1])
    rt.say("cam off")
    check("'cam off' is understood", not rt.periphery.sensors["camera"].on and rt.out[-1] == "Camera off.", rt.out[-1])
    rt.say("turn on the mic and the camera", addressed=False, source="microphone")
    check("turning on from overheard speech is refused (it only ever makes ASH more private)", not rt.periphery.sensors["camera"].on)
    rt.say("mic off and camera off", addressed=False, source="microphone")
    check("but off is allowed from any line the microphone hears", not rt.periphery.sensors["microphone"].on)
    n = len(rt.ash_.calls)
    rt.say("what should I cook tonight")
    check("an ordinary sentence still goes to the model", len(rt.ash_.calls) > n)


def v8_state_claims():
    print("\n=== V8. What the model says about the devices ===")
    from src.daemon.voicectl import device_claims, device_truth
    for said, want in [("Mic and camera are both already off. You're good to go without them.", {"microphone": False, "camera": False}),
                       ("The microphone is on.", {"microphone": True}), ("Your camera is currently off.", {"camera": False}),
                       ("The mic is already off.", {"microphone": False}), ("The camera's active right now.", {"camera": True})]:
        check(f"claims in {said[:44]!r}", device_claims(said) == want, str(device_claims(said)))
    for said in ("You can turn off the camera from the console.", "If the mic is on, I can hear you.", "Want me to turn the mic off?", "Is the camera on?",
                 "To switch the microphone off, say mic off.", "I like my camera.", "The weather is on the way.", "Say turn on the mic and I will.", ""):
        check(f"not a claim: {said[:44]!r}", device_claims(said) == {}, str(device_claims(said)))
    check("a true claim is left alone", device_truth({"microphone": False}, {"microphone": False}) is None)
    check("a false one is corrected, naming only what it talked about",
          device_truth({"microphone": False, "camera": False}, {"microphone": True, "camera": False}) == "Let me correct that: the microphone is on and the camera is off.")
    rt = rig(reply="Mic and camera are both already off. You're good to go without them.")
    rt.say("ash what are you doing about the sensors")
    check("while the mic is on the model's 'both already off' is replaced by the truth", rt.out[-1] == "Let me correct that: the microphone is on and the camera is off.", rt.out[-1])
    rt.periphery.sensors["microphone"].disable()
    rt.say("ash what are you doing about the sensors")
    check("when it is true it is left alone", rt.out[-1].startswith("Mic and camera are both already off"), rt.out[-1])
    rt2 = rig(reply="You can turn off the camera from the console.")
    rt2.say("ash how do i stop the camera")
    check("advice about a device is left alone", rt2.out[-1] == "You can turn off the camera from the console.", rt2.out[-1])


if __name__ == "__main__":
    for fn in (v1_parse, v2_doing, v2b_timed, v2c_faces, v3_who, v4_shutdown, v5_claims, v6_face_quiet, v7_many, v8_state_claims):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
