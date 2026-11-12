"""
tst_lane.py -- commands that don't wait behind a model answer.

On 2026-11-07 "screen capture on", typed at 22:35:09, ran at 22:35:16: right
after a 32-second model answer ended. The loop is one thread and a model
answer takes 20-70 s, so "privacy mode" or "stop" could wait that long, and
the microphone was not even listened to meanwhile.

  L1  While a model answer is in progress, a control command runs at once;
      the answer still finishes, and the command runs only once
  L2  The microphone is listened to meanwhile
  L3  Not busy: the lane does nothing (the loop handles everything in order)
  L4  Only commands: a question is left for the loop
  L5  "Stop" cuts speech while the model is still thinking
  L6  A turn-on said to the room (not to ASH) is not run by the lane, and is
      handed back to the loop
  L7  The runtime starts the lane

Run:  PYTHONIOENCODING=utf-8 python tests/tst_lane.py
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


class SlowAsh(tw.Ash):
    """A model that takes as long as the test lets it."""

    def __init__(self):
        super().__init__()
        self.go = threading.Event()
        self.started = threading.Event()

    def think(self, query, image=None, context=""):
        self.calls.append(query)
        self.started.set()
        self.go.wait(15)
        r = tw.Resp("A long answer.")
        r.pathway = type("P", (), {"value": "slow"})()
        return r


class Mic:
    name = "microphone"

    def __init__(self):
        from src.senses.base import Capability, State
        self.state = State.ACTIVE
        self.pending = []
        self.capability = Capability(name="microphone", privacy_sensitive=True)

    def due(self, now):
        return True

    def poll(self):
        out, self.pending = self.pending, []
        return out

    def enable(self):
        from src.senses.base import State
        self.state = State.ACTIVE

    def close(self):
        pass

    def disable(self):
        from src.senses.base import State
        self.state = State.DISABLED


def rig(with_speaker=False):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    per = PeripheryRegistry()
    per.sensors["microphone"] = Mic()
    out = []
    ash = SlowAsh()
    if with_speaker:
        from src.senses.actuators import SpeakActuator

        class Tts:
            engine = True

            def stream_audio(self, text, voice=None):
                yield b"RIFF"
        sp = SpeakActuator(tts_engine=Tts(), player=lambda w, stop: stop.wait(10), dry_run=False)
        sp.enable()
        per.actuators["speak"] = sp
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9),
                        response_sink=lambda t, m: out.append(t))

    def ev(text, source="console", addressed=True):
        return SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed, salience=0.9,
                           urgency=0.6, ts=time.time())
    rt.ev, rt.out, rt.ash_, rt.mic = ev, out, ash, per.sensors["microphone"]
    return rt


def busy(rt, question="tell me something interesting about the sea"):
    """Put the loop inside a model answer, on its own thread."""
    rt.submit(rt.ev(question))
    th = threading.Thread(target=rt._tick, daemon=True)
    th.start()
    ok = rt.ash_.started.wait(10)
    return th, ok


def l1_runs_at_once():
    print("\n=== L1. A command does not wait ===")
    rt = rig()
    th, ok = busy(rt)
    check("(the model is busy)", ok and rt._thinking == 1)
    rt.submit(rt.ev("privacy mode"))
    t0 = time.time()
    rt._lane_step()
    check("privacy mode ran at once, with the model still thinking", rt.privacy.paused and th.is_alive()
          and time.time() - t0 < 2.0, f"paused={rt.privacy.paused}")
    check("and said what it did", any("Privacy mode on" in o for o in rt.out), str(rt.out))
    rt.ash_.go.set()
    th.join(10)
    check("the answer still finished", not th.is_alive() and "A long answer." in rt.out, str(rt.out))
    check("the command ran once", sum("Privacy mode on" in o for o in rt.out) == 1 and rt.stats.get("lane") == 1,
          f"{rt.out} lane={rt.stats.get('lane')}")
    rt._tick()
    check("and the loop did not run it again", sum("Privacy mode on" in o for o in rt.out) == 1)


def l2_listens():
    print("\n=== L2. Listening meanwhile ===")
    rt = rig()
    th, ok = busy(rt)
    rt.mic.pending.append(rt.ev("turn off the mic", source="microphone"))
    rt._lane_step()
    from src.senses.base import State
    check("a command heard while it was thinking was run", rt.mic.state == State.DISABLED
          and any(o.startswith("Microphone off") for o in rt.out), str(rt.out))
    rt.ash_.go.set()
    th.join(10)


def l3_idle():
    print("\n=== L3. Not busy: nothing ===")
    rt = rig()
    rt.submit(rt.ev("privacy mode"))
    rt.mic.pending.append(rt.ev("hello there", source="microphone"))
    rt._lane_step()
    check("the command is left for the loop", not rt.privacy.paused and len(rt._queue) == 1)
    check("the mic is left alone", len(rt.mic.pending) == 1)
    rt._tick()
    check("and the loop does it", rt.privacy.paused)


def l4_only_commands():
    print("\n=== L4. Only commands ===")
    rt = rig()
    th, ok = busy(rt)
    rt.submit(rt.ev("what is the tallest mountain on earth"))
    rt._lane_step()
    check("a question is left in the queue", len(rt._queue) == 1 and rt.ash_.calls == [
        "tell me something interesting about the sea"], str(rt.ash_.calls))
    check("...untouched: not counted, not marked, not journaled by the lane", not rt.stats.get("lane")
          and not rt._lane_skip)
    rt.submit(rt.ev("privacy mode", source="screen", addressed=False))
    rt._lane_step()
    check("text read off the screen is never a command", not rt.privacy.paused and len(rt._queue) == 2
          and not rt._lane_skip)
    rt.ash_.go.set()
    th.join(10)


def l5_stop():
    print("\n=== L5. Stop, while thinking ===")
    rt = rig(with_speaker=True)
    sp = rt.periphery.actuators["speak"]
    sp.act("My current status is warm and upbeat and here is a long thing to say")
    time.sleep(0.1)
    check("(speaking)", sp.speaking)
    th, ok = busy(rt)
    rt.submit(rt.ev("stop", source="microphone"))
    rt._lane_step()
    time.sleep(0.1)
    check("'stop' cut the speech while the model was thinking", not sp.speaking and th.is_alive())
    check("and nothing was said back", not any(o.lower().startswith("stop") for o in rt.out), str(rt.out))
    rt.ash_.go.set()
    th.join(10)


def l6_not_for_ash():
    print("\n=== L6. Said to the room ===")
    from src.senses.base import State
    rt = rig()
    rt.periphery.sensors["camera"] = Mic()
    rt.periphery.sensors["camera"].name = "camera"
    rt.periphery.sensors["camera"].disable()
    th, ok = busy(rt)
    rt.submit(rt.ev("activate camera", source="microphone", addressed=False))
    rt._lane_step()
    check("a turn-on not said to ASH is not run", rt.periphery.sensors["camera"].state == State.DISABLED)
    check("it is handed back to the loop", len(rt._queue) == 1)
    rt._lane_step()
    check("and the lane does not take it again", len(rt._queue) == 1 and not rt.stats.get("lane"))
    rt.ash_.go.set()
    th.join(10)


def l8_journal_across_midnight():
    print("\n=== L8. 'What just happened' across midnight ===")
    from datetime import datetime
    from src.daemon import Journal
    from src.senses.base import Modality, SensorEvent
    j = Journal(tempfile.mkdtemp())
    now = time.time()
    midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    for text, ts in (("an hour before midnight", midnight - 3600), ("ten minutes before midnight", midnight - 600),
                     ("just after midnight", midnight + 120), ("now-ish", now - 5)):
        j.append(SensorEvent(source="console", modality=Modality.TEXT, text=text, ts=ts, salience=0.5))
    j.flush()
    got = [r["text"] for r in j.recent(seconds=now - (midnight - 1800), limit=50)]
    check("the last window includes yesterday's side of midnight", "ten minutes before midnight" in got
          and "just after midnight" in got and "now-ish" in got and "an hour before midnight" not in got, str(got))
    check("a short window is still just now", [r["text"] for r in j.recent(seconds=60)] == ["now-ish"])
    check("the prompt block shows them in order", j.context_block(seconds=now - (midnight - 1800)).index("ten minutes")
          < j.context_block(seconds=now - (midnight - 1800)).index("now-ish"))
    j2 = Journal(tempfile.mkdtemp())
    for text, days_ago in (("five days ago", 5), ("two days ago", 2)):
        j2.append(SensorEvent(source="console", modality=Modality.TEXT, text=text, ts=now - days_ago * 86400,
                              salience=0.5))
    j2.flush()
    got2 = [r["text"] for r in j2.recent(seconds=86400 * 30, limit=50)]
    check("a long window reads back at most three days", got2 == ["two days ago"], str(got2))


def l9_never_mind():
    print("\n=== L9. 'Never mind' drops the answer being worked out ===")
    from src.ui import settings
    rt = rig()
    th, ok = busy(rt)
    w = settings.waiting(rt)
    check("the console is told ASH is thinking, and can offer 'Never mind'", w is not None and w["kind"] == "thinking"
          and "sea" in w["text"] and w["replies"] == ["Never mind"], str(w))
    rt.submit(rt.ev("never mind"))
    rt._lane_step()
    check("it was heard at once", rt._cancelled and any("dropped it" in o for o in rt.out), str(rt.out))
    rt.ash_.go.set()
    th.join(10)
    check("the answer, when it arrived, was not said", "A long answer." not in rt.out, str(rt.out))
    check("...and it is counted", rt.stats.get("cancelled") == 1)
    check("nothing is waiting any more", not rt._cancelled and settings.waiting(rt) is None)
    # the next question is answered normally
    rt.ash_.go.set()
    rt.submit(rt.ev("tell me about the moon"))
    rt._tick()
    check("the next answer is said", "A long answer." in rt.out, str(rt.out))
    # not thinking: "never mind" is left to whatever asked (a question, a plan...)
    rt2 = rig()
    rt2.submit(rt2.ev("never mind"))
    rt2._lane_step()
    check("when ASH isn't thinking the lane leaves it alone", not rt2._cancelled and len(rt2._queue) == 1)
    # a longer sentence that merely starts with those words is not a cancel
    rt3 = rig()
    th3, ok3 = busy(rt3)
    rt3.submit(rt3.ev("never mind the weather, what is the capital of france"))
    rt3._lane_step()
    check("'never mind the weather, what is...' is a question, not a cancel", not rt3._cancelled
          and len(rt3._queue) == 1)
    rt3.ash_.go.set()
    th3.join(10)


def l7_started():
    print("\n=== L7. The runtime starts the lane ===")
    rt = rig()
    rt.start()
    try:
        check("a lane thread is running", rt._lane is not None and rt._lane.is_alive())
    finally:
        rt.stop(timeout=3)
        rt.ash_.go.set()


if __name__ == "__main__":
    for fn in (l1_runs_at_once, l2_listens, l3_idle, l4_only_commands, l5_stop, l6_not_for_ash, l7_started, l8_journal_across_midnight, l9_never_mind):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
