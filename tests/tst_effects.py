"""
tst_effects.py -- a command must DO the thing, not just change a flag or the GUI.

Every other test of voice/console commands used fake sensors that flip a
boolean, so they could not tell "the microphone is off" from "ASH says the
microphone is off". These use the real classes: MicrophoneSensor (its capture
thread), CameraSensor (the device handle), ScreenshotSensor, PrivacyPolicy,
the face player and the speaker.

  E1  Mic off stops the capture thread and drops what it had heard; on restarts it
  E2  Privacy mode really switches the mic, camera and screen off -- and
      only those that were on come back when it ends
  E3  While privacy mode is on nothing sensitive can be switched on (voice or console),
      and one that gets on anyway is switched off at the next tick
  E4  A sensor that cannot work says why, instead of "Camera on."
  E5  Camera off lets go of the device
  E6  Screen capture off forgets what it read; a read finishing late is dropped
  E7  A face request really changes the face player
  E8  Stop really stops the speaker; sleep really silences; shutdown really asks to shut down
  E9  The console's enable/disable goes the same way and reports failure

Run:  PYTHONIOENCODING=utf-8 python tests/tst_effects.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
T0 = tw.T0


def real_mic(available=True):
    from src.senses.mic import MicrophoneSensor

    class Mic(MicrophoneSensor):
        """The real sensor and its real start/stop; only the hardware read is replaced."""
        def probe(self):
            return (True, "") if available else (False, "no audio backend (test)")

        def _loop(self):
            while not self._stop.wait(0.01):
                pass
    return Mic(enabled=False)


class FakeCap:
    def __init__(self):
        self.released = 0

    def isOpened(self):
        return self.released == 0

    def read(self):
        return False, None

    def release(self):
        self.released += 1


def real_camera(available=True):
    from src.senses.devices import CameraSensor

    class Cam(CameraSensor):
        requires_ok = True

        def probe(self):
            return (True, "") if available else (False, "no camera at index 0")
    c = Cam(enabled=False)
    if c.state.value == "unavailable" and available:
        # opencv isn't installed here: the lifecycle under test doesn't need it
        from src.senses.base import State
        c.state, c.unavailable_reason = State.DISABLED, None
    return c


def real_screen():
    from src.senses.screen import ScreenshotSensor
    s = ScreenshotSensor(enabled=False)
    if s.state.value == "unavailable":
        from src.senses.base import State
        s.state, s.unavailable_reason = State.DISABLED, None
    return s


def rig(with_face=False, with_speaker=False, **cfg):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    per = PeripheryRegistry()
    per.sensors["microphone"] = real_mic()
    per.sensors["camera"] = real_camera()
    per.sensors["screenshot"] = real_screen()
    sp = None
    if with_speaker:
        from src.senses.actuators import SpeakActuator

        class Tts:
            engine = True

            def stream_audio(self, text, voice=None):
                yield b"RIFF"
        sp = SpeakActuator(tts_engine=Tts(), player=lambda w, stop: stop.wait(10), dry_run=False)
        sp.enable()
        per.actuators["speak"] = sp

    class Ash(tw.Ash):
        def think(self, query, image=None, context=""):
            self.calls.append(query)
            r = tw.Resp("Sure.")
            r.pathway = type("P", (), {"value": "slow"})()
            return r
    ash = Ash()
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, **cfg),
                        response_sink=lambda t, m: out.append(t))
    rt.clock = lambda: clock[0]
    if with_face:
        from src.face import build_face
        from src.face.bridge import FaceBridge
        rt.face = FaceBridge(build_face(driver="null", animations_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "animations")))

    def say(text, addressed=True, source="microphone"):
        clock[0] += 3
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed,
                              salience=0.9, urgency=0.6))
        rt._tick()
    rt.say, rt.out, rt.t, rt.sp = say, out, clock, sp
    rt.mic, rt.cam, rt.scr = per.sensors["microphone"], per.sensors["camera"], per.sensors["screenshot"]
    return rt


def alive(mic):
    return bool(mic._thread and mic._thread.is_alive())


def stop_all(rt):
    rt.mic.close()


def e1_mic():
    print("\n=== E1. Microphone ===")
    from src.senses.base import SensorEvent, Modality
    rt = rig()
    try:
        rt.say("turn on the mic", source="console")
        check("'on' starts the real capture thread", alive(rt.mic) and rt.mic.state.value == "active"
              and rt.out[-1] == "Microphone on.", str(rt.out[-1:]))
        rt.say("turn off the mic")
        rt.set_sensor("microphone", True)
        rt.mic._out.append(SensorEvent(source="microphone", modality=Modality.TEXT, text="heard before off"))
        rt.set_sensor("microphone", False)
        check("'off' stops the thread", not alive(rt.mic) and rt.mic.state.value == "disabled", str(rt.mic._thread))
        check("...and what it had heard is dropped, not answered later", len(rt.mic._out) == 0)
        check("...and it said so", rt.out[-1].startswith("Microphone off."))
        rt.say("turn on the mic", source="console")
        check("on again restarts it", alive(rt.mic))
    finally:
        stop_all(rt)


def e2_privacy():
    print("\n=== E2. Privacy mode ===")
    rt = rig()
    try:
        rt.cam._cap = FakeCap()
        cap = rt.cam._cap
        rt.set_sensor("microphone", True)
        rt.set_sensor("camera", True)                      # the screen stays off
        check("(mic and camera on, screen off)", alive(rt.mic) and rt.cam.state.value == "active"
              and rt.scr.state.value == "disabled")
        rt.say("go to privacy mode")
        check("the microphone really stopped", not alive(rt.mic) and rt.mic.state.value == "disabled")
        check("the camera really let go", cap.released >= 1 and rt.cam.state.value == "disabled")
        check("privacy mode is on", rt.privacy.paused)
        rt.say("privacy off", source="console")
        check("the mic is listening again", alive(rt.mic) and rt.mic.state.value == "active")
        check("the camera is back", rt.cam.state.value == "active")
        check("the screen -- which was off -- stays off", rt.scr.state.value == "disabled")
        # console route into privacy mode
        rt.privacy.pause("console")
        check("the console's pause does the same", not alive(rt.mic) and rt.cam.state.value == "disabled")
        rt.privacy.resume()
        check("and its resume", alive(rt.mic) and rt.cam.state.value == "active")
        # off during privacy mode stays off after it
        rt.privacy.pause("console")
        rt.privacy.resume()
        rt.say("turn off the mic")
        rt.privacy.pause("console")
        rt.privacy.resume()
        check("a mic you turned off before privacy mode is not turned on by it", not alive(rt.mic))
    finally:
        stop_all(rt)


def e3_refuse():
    print("\n=== E3. No back door ===")
    rt = rig()
    try:
        rt.privacy.pause("console")
        rt.say("turn on the mic")
        check("voice: refused, and said why", not alive(rt.mic) and rt.mic.state.value == "disabled"
              and "Privacy mode is on" in rt.out[-1], str(rt.out[-1:]))
        ok, why = rt.set_sensor("camera", True)
        check("console: refused too", not ok and "privacy" in why and rt.cam.state.value == "disabled", why)
        rt.mic.enable()                                    # something switches it on anyway
        check("(it is running)", alive(rt.mic))
        rt._tick()
        check("the next tick switches it off again", not alive(rt.mic) and rt.mic.state.value == "disabled")
        rt.privacy.resume()
        check("(it was held, so ending privacy mode brings it back)", alive(rt.mic))
    finally:
        stop_all(rt)


def e4_cannot():
    print("\n=== E4. A sensor that cannot work ===")
    rt = rig()
    try:
        from src.senses.base import State
        rt.cam.state, rt.cam.unavailable_reason = State.UNAVAILABLE, "no camera at index 0"
        rt.say("turn on the camera")
        check("it does not say 'Camera on.'", "Camera on." not in rt.out[-1], rt.out[-1])
        check("it says it couldn't, and why", "couldn't" in rt.out[-1] and "no camera at index 0" in rt.out[-1],
              rt.out[-1])
        bad = real_mic(available=False)
        bad.enable()
        check("an unavailable microphone does not start a capture thread", not alive(bad)
              and bad.state.value == "unavailable")
        rt.periphery.sensors["microphone"] = bad
        rt.say("turn on the mic")
        check("and says so", "couldn't" in rt.out[-1] and "no audio backend" in rt.out[-1], rt.out[-1])
        rt.say("turn on the screen capture")
        check("a working one still works", rt.scr.state.value == "active" and rt.out[-1] == "Screen capture on.")
        rt.periphery.sensors.pop("camera")
        rt.say("turn on the camera")
        check("no such sensor: said", "no camera" in rt.out[-1].lower(), rt.out[-1])
    finally:
        stop_all(rt)


def e5_camera():
    print("\n=== E5. Camera device ===")
    rt = rig()
    cap = FakeCap()
    rt.cam._cap = cap
    rt.cam._bg, rt.cam._frames_seen = object(), 9
    rt.set_sensor("camera", True)
    rt.say("turn off the camera")
    check("the device was released and forgotten", cap.released == 1 and rt.cam._cap is None)
    check("what it saw is forgotten", rt.cam._bg is None and rt.cam._frames_seen == 0)
    check("and it said so", rt.out[-1] == "Camera off.")


def e6_screen():
    print("\n=== E6. Screen capture ===")
    from src.senses.base import State
    rt = rig()
    rt.set_sensor("screenshot", True)
    rt.scr._ocr_pending = {"text": "secret page", "title": "t"}
    rt.scr._last_view = {"title": "t", "scene": "", "text": "secret page", "at": time.time()}
    rt.say("turn off screen capture")
    check("a reading not yet delivered is dropped", rt.scr._ocr_pending is None)
    check("and it will not describe the screen", rt.scr._last_view is None)
    check("off, said", rt.scr.state == State.DISABLED and rt.out[-1] == "Screen capture off.")


def e7_face():
    print("\n=== E7. Face ===")
    rt = rig(with_face=True)
    player = rt.face.player
    rt.say("smile for me")
    check("a smile is the animation playing", player.current is not None and player.current.name == "happy",
          str(getattr(player.current, "name", None)))
    check("and it said so", rt.out[-1] == "Like this.", rt.out[-1])
    rt.say("look surprised")
    check("surprised is playing", player.current.name == "surprised", str(player.current.name))
    poses = []
    player.on_frame = lambda p, f: poses.append(p)
    player.pose = player.pose
    before = player.current_t
    rt.say("make a sad face")
    check("sad replaces it", player.current.name == "sad")
    rt.say("what faces can you make?")
    check("the list names real animations", "happy" in rt.out[-1] and "sad" in rt.out[-1], rt.out[-1])
    rt.face = None
    rt.say("smile")
    check("no face running: says so", "face isn't running" in rt.out[-1], rt.out[-1])


def e8_speech_sleep_shutdown():
    print("\n=== E8. Stop, sleep, shutdown ===")
    rt = rig(with_speaker=True)
    sp = rt.sp
    calls = []
    rt.request_shutdown = lambda: calls.append(1)
    sp.act("My current status is warm and upbeat and here is a long thing to say")
    time.sleep(0.1)
    check("(speaking)", sp.speaking)
    rt.say("stop")
    time.sleep(0.1)
    check("'stop' really stops the speaker", not sp.speaking)
    rt.say("go to sleep")
    check("sleep really mutes the gate", rt.gate.muted)
    rt.say("wake up")
    check("wake really unmutes it", not rt.gate.muted)
    rt.say("shut down")
    check("shutdown asks first and does nothing yet", not calls and "Say yes" in rt.out[-1], rt.out[-1])
    rt.say("yes")
    check("a yes really requests the shutdown, once", calls == [1], str(calls))


def e9_console():
    print("\n=== E9. Console route ===")
    from src.ui import UIBus, attach_ui
    rt = rig()
    try:
        _, ctl, _state = attach_ui(rt, UIBus(), opener=None, status_every_s=10 ** 9,
                                   config_path=os.path.join(tempfile.mkdtemp(), "c.json"))
        r = ctl("enable", "microphone")
        check("console enable starts the mic", r.get("ok") and alive(rt.mic), str(r))
        r = ctl("disable", "microphone")
        check("console disable stops it", r.get("ok") and not alive(rt.mic), str(r))
        ctl("pause", "")
        r = ctl("enable", "camera")
        check("in privacy mode console enable fails and says why", not r.get("ok") and "privacy" in r.get("error", ""),
              str(r))
        r = ctl("enable", "nothing_here")
        check("an unknown name fails", not r.get("ok"), str(r))
        from src.senses.base import State
        ctl("resume", "")
        rt.cam.state, rt.cam.unavailable_reason = State.UNAVAILABLE, "no camera at index 0"
        r = ctl("enable", "camera")
        check("an unavailable sensor is reported, not 'Turned on'", not r.get("ok") and "no camera" in r.get("error", ""),
              str(r))
    finally:
        stop_all(rt)


if __name__ == "__main__":
    for fn in (e1_mic, e2_privacy, e3_refuse, e4_cannot, e5_camera, e6_screen, e7_face, e8_speech_sleep_shutdown,
               e9_console):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
