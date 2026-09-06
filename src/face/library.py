"""
src/face/library.py

The premade animations.

Each one is a handful of keyframes. They are written in Python rather than
loaded from JSON because they are code the runtime depends on -- a missing
`blink.json` should not be able to break the face -- but every one of them
exports to the editor's format via `Animation.to_json()`, so the workflow is:
load a builtin in the editor, adjust it, save it into `animations/`, and the
loader prefers the file over the builtin.

Timing notes, since these numbers are the whole craft here:

  * A human blink is 100-150ms of closing and 150-200ms of opening -- the
    reopen is slower than the close. Symmetric blinks read as mechanical, and
    this one asymmetry does more for believability than any other single
    value in this file.
  * Saccades are 30-80ms and overshoot slightly, hence `back` easing on
    look-arounds.
  * Expressions hold. A `happy` that decays after 400ms reads as a twitch;
    the hold is what makes it a mood rather than an event.
  * Nothing returns exactly to rest. Real faces settle near neutral, not on
    it, so the idle pose carries a slight asymmetry.
"""

from __future__ import annotations

from typing import Dict, List

from .model import Animation, Keyframe

# Priority bands. Anything above 80 preempts conversation; anything below 20
# only plays when nothing else wants the screen.
P_IDLE = 10
P_AMBIENT = 30
P_EXPRESSION = 50
P_REACTION = 65
P_SPEECH = 70
P_ALERT = 90
P_CRITICAL = 100


def _a(name: str, kfs: List[Keyframe], **kw) -> Animation:
    return Animation(name=name, keyframes=kfs, **kw)


# ----------------------------------------------------------------------
# Reflexes
# ----------------------------------------------------------------------
BLINK = _a("blink", [
    Keyframe(0.00, both={"open": 1.0}, ease="in"),
    Keyframe(0.09, both={"open": 0.06}, ease="in"),      # fast close
    Keyframe(0.13, both={"open": 0.06}, ease="linear"),
    Keyframe(0.30, both={"open": 1.0}, ease="out"),      # slower reopen
], priority=P_IDLE + 5, interruptible=False, hold_last=False, tags=["reflex"])

DOUBLE_BLINK = _a("double_blink", [
    Keyframe(0.00, both={"open": 1.0}, ease="in"),
    Keyframe(0.08, both={"open": 0.06}),
    Keyframe(0.20, both={"open": 1.0}, ease="out"),
    Keyframe(0.30, both={"open": 0.06}, ease="in"),
    Keyframe(0.46, both={"open": 1.0}, ease="out"),
], priority=P_IDLE + 5, interruptible=False, hold_last=False, tags=["reflex"])

WINK = _a("wink", [
    Keyframe(0.00, left={"open": 1.0}),
    Keyframe(0.10, left={"open": 0.05}, right={"lid_bottom": 0.18}),
    Keyframe(0.34, left={"open": 0.05}),
    Keyframe(0.50, left={"open": 1.0}, right={"lid_bottom": 0.0}, ease="out"),
], priority=P_REACTION, tags=["playful"])


# ----------------------------------------------------------------------
# Idle
# ----------------------------------------------------------------------
IDLE = _a("idle", [
    # Never perfectly still. The drift is small enough to be subliminal and
    # is the single cheapest thing that stops a face looking switched-off.
    Keyframe(0.0, face={"look_x": 0.0, "look_y": 0.0}, both={"w": 36, "h": 36}),
    Keyframe(2.2, face={"look_x": 1.2, "look_y": -0.8}),
    Keyframe(4.6, face={"look_x": -1.0, "look_y": 0.6}),
    Keyframe(6.8, face={"look_x": 0.4, "look_y": 0.9}),
    Keyframe(9.0, face={"look_x": 0.0, "look_y": 0.0}),
], loop=True, priority=P_IDLE, tags=["idle"])

LOOK_AROUND = _a("look_around", [
    Keyframe(0.00, face={"look_x": 0}),
    Keyframe(0.18, face={"look_x": -9}, ease="back"),     # saccade + overshoot
    Keyframe(0.90, face={"look_x": -9}, ease="linear"),
    Keyframe(1.10, face={"look_x": 9}, ease="back"),
    Keyframe(1.85, face={"look_x": 9}, ease="linear"),
    Keyframe(2.10, face={"look_x": 0}, ease="out"),
], priority=P_AMBIENT, tags=["idle", "curious"])

SLEEPY = _a("sleepy", [
    Keyframe(0.0, both={"lid_top": 0.15, "h": 34}),
    Keyframe(1.6, both={"lid_top": 0.45, "h": 30}, face={"look_y": 2}),
    Keyframe(3.0, both={"lid_top": 0.22, "h": 34}, face={"look_y": 0}),
    Keyframe(4.4, both={"lid_top": 0.52, "h": 28}, face={"look_y": 3}),
    Keyframe(6.0, both={"lid_top": 0.28, "h": 33}, face={"look_y": 1}),
], loop=True, priority=P_IDLE + 2, tags=["idle", "tired"])

SLEEPING = _a("sleeping", [
    Keyframe(0.0, both={"open": 0.05, "w": 34}, face={"brightness": 0.35}),
    Keyframe(2.0, both={"w": 30}, face={"brightness": 0.22}),   # slow "breath"
    Keyframe(4.0, both={"w": 34}, face={"brightness": 0.35}),
], loop=True, priority=P_IDLE + 3, blink_suppress=True, tags=["sleep"])


# ----------------------------------------------------------------------
# Expressions
# ----------------------------------------------------------------------
HAPPY = _a("happy", [
    # The bottom lid rising is what makes a smile without a mouth. Eyes
    # narrow from below when people genuinely smile; from above when they
    # squint. Getting this backwards makes a face look suspicious, not glad.
    Keyframe(0.00, both={"lid_bottom": 0.0, "h": 36, "radius": 10}),
    Keyframe(0.22, both={"lid_bottom": 0.34, "h": 38, "radius": 16}, ease="out"),
    Keyframe(1.60, both={"lid_bottom": 0.34}, ease="linear"),
], priority=P_EXPRESSION, tags=["positive"])

VERY_HAPPY = _a("very_happy", [
    Keyframe(0.00, both={"lid_bottom": 0.0, "h": 36}),
    Keyframe(0.18, both={"lid_bottom": 0.45, "h": 40, "radius": 18}, ease="back"),
    Keyframe(0.34, both={"y": -2}, ease="out"),
    Keyframe(0.50, both={"y": 0}, ease="bounce"),
    Keyframe(1.80, both={"lid_bottom": 0.42}),
], priority=P_EXPRESSION + 5, tags=["positive"])

SAD = _a("sad", [
    # Outer corners down: lid_angle is mirrored per eye by the model, so a
    # single positive value produces the correct opposing slants.
    Keyframe(0.00, both={"lid_top": 0.0, "lid_angle": 0}),
    Keyframe(0.45, both={"lid_top": 0.30, "lid_angle": -16, "h": 32},
             face={"look_y": 3}, ease="out"),
    Keyframe(2.20, both={"lid_top": 0.30}),
], priority=P_EXPRESSION, tags=["negative"])

ANGRY = _a("angry", [
    Keyframe(0.00, both={"lid_top": 0.0, "lid_angle": 0}),
    Keyframe(0.16, both={"lid_top": 0.34, "lid_angle": 22, "radius": 5, "h": 34},
             ease="in"),
    Keyframe(1.40, both={"lid_top": 0.34, "lid_angle": 22}),
], priority=P_EXPRESSION + 5, tags=["negative"])

ANNOYED = _a("annoyed", [
    Keyframe(0.00, both={"lid_top": 0.0}),
    Keyframe(0.25, both={"lid_top": 0.30, "lid_angle": 10}, face={"look_x": 6},
             ease="out"),
    Keyframe(0.75, face={"look_x": 6}),
    Keyframe(1.05, face={"look_x": 0}, ease="in_out"),     # the side-eye
    Keyframe(2.00, both={"lid_top": 0.24}),
], priority=P_EXPRESSION, tags=["negative", "dry"])

SURPRISED = _a("surprised", [
    Keyframe(0.00, both={"w": 36, "h": 36}),
    Keyframe(0.11, both={"w": 44, "h": 46, "radius": 20, "lid_top": 0}, ease="back"),
    Keyframe(0.70, both={"w": 42, "h": 44}),
    Keyframe(1.30, both={"w": 38, "h": 38}, ease="out"),
], priority=P_REACTION, blink_suppress=True, tags=["reaction"])

CONFUSED = _a("confused", [
    # Asymmetry is the entire signal. One eye narrows and the head tilts.
    Keyframe(0.00, left={"h": 36}, right={"h": 36}, face={"tilt": 0}),
    Keyframe(0.35, left={"h": 28, "lid_top": 0.22}, right={"h": 40},
             face={"tilt": -7}, ease="out"),
    Keyframe(1.60, face={"tilt": -7}),
], priority=P_EXPRESSION, tags=["uncertain"])

SKEPTICAL = _a("skeptical", [
    Keyframe(0.00, left={"lid_top": 0.0}, right={"lid_top": 0.0}),
    Keyframe(0.30, left={"lid_top": 0.38, "lid_angle": 12},
             right={"lid_top": 0.08}, face={"look_x": -4}, ease="out"),
    Keyframe(1.80, left={"lid_top": 0.38}),
], priority=P_EXPRESSION, tags=["uncertain", "dry"])

CURIOUS = _a("curious", [
    Keyframe(0.00, both={"w": 36, "h": 36}, face={"tilt": 0}),
    Keyframe(0.30, both={"w": 39, "h": 41}, face={"tilt": 6, "look_y": -2}, ease="out"),
    Keyframe(1.10, face={"tilt": 6}),
    Keyframe(1.60, face={"tilt": 4}),
], priority=P_EXPRESSION, tags=["positive", "curious"])

LOVE = _a("love", [
    Keyframe(0.00, both={"h": 36, "lid_bottom": 0.0}),
    Keyframe(0.30, both={"h": 40, "lid_bottom": 0.40, "radius": 20}, ease="out"),
    Keyframe(0.60, both={"w": 40}, ease="bounce"),
    Keyframe(2.20, both={"w": 37}),
], priority=P_EXPRESSION, tags=["positive"])


# ----------------------------------------------------------------------
# Cognitive states -- these are the ones wired to the brain
# ----------------------------------------------------------------------
THINKING = _a("thinking", [
    # Gaze up and off to one side, the universal "working on it". Blink
    # suppressed, because concentration suppresses blinking in people too.
    Keyframe(0.0, face={"look_x": 0, "look_y": 0}),
    Keyframe(0.4, face={"look_x": 7, "look_y": -5}, both={"lid_top": 0.12}, ease="out"),
    Keyframe(1.5, face={"look_x": 9, "look_y": -6}),
    Keyframe(2.4, face={"look_x": 5, "look_y": -4}),
    Keyframe(3.4, face={"look_x": 8, "look_y": -6}),
], loop=True, priority=P_SPEECH, blink_suppress=True, tags=["cognitive"])

LISTENING = _a("listening", [
    # Wide, still, locked forward. Stillness is the signal -- an animated
    # "listening" face looks like it is thinking about something else.
    Keyframe(0.00, both={"w": 36, "h": 36}),
    Keyframe(0.20, both={"w": 38, "h": 40, "lid_top": 0}, face={"look_x": 0, "look_y": 0},
             ease="out"),
    Keyframe(2.00, both={"h": 39}),
], loop=True, priority=P_SPEECH, tags=["cognitive"])

SPEAKING = _a("speaking", [
    Keyframe(0.00, both={"h": 36}),
    Keyframe(0.16, both={"h": 33, "lid_bottom": 0.12}),
    Keyframe(0.32, both={"h": 37, "lid_bottom": 0.04}),
    Keyframe(0.48, both={"h": 34, "lid_bottom": 0.10}),
], loop=True, priority=P_SPEECH, tags=["cognitive"])

PROCESSING = _a("processing", [
    # Eyes sweep like a progress indicator. Reads as "busy" without a spinner.
    Keyframe(0.0, face={"look_x": -8}, both={"w": 30, "h": 30}),
    Keyframe(0.5, face={"look_x": 8}, ease="in_out"),
    Keyframe(1.0, face={"look_x": -8}, ease="in_out"),
], loop=True, priority=P_SPEECH, blink_suppress=True, tags=["cognitive"])

ACKNOWLEDGE = _a("acknowledge", [
    # A nod, done with vertical gaze since there is no neck.
    Keyframe(0.00, face={"look_y": 0}),
    Keyframe(0.14, face={"look_y": 5}, both={"lid_top": 0.2}, ease="in"),
    Keyframe(0.32, face={"look_y": 0}, both={"lid_top": 0.0}, ease="out"),
], priority=P_REACTION, hold_last=False, tags=["reaction"])

REFUSE = _a("refuse", [
    # A head-shake. Used when the Executive vetoes an action -- the face says
    # no before the sentence does.
    Keyframe(0.00, face={"look_x": 0}, both={"lid_top": 0.25, "lid_angle": 10}),
    Keyframe(0.13, face={"look_x": -7}, ease="in_out"),
    Keyframe(0.28, face={"look_x": 7}, ease="in_out"),
    Keyframe(0.43, face={"look_x": -5}, ease="in_out"),
    Keyframe(0.58, face={"look_x": 0}, ease="out"),
], priority=P_REACTION + 5, interruptible=False, tags=["reaction", "negative"])


# ----------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------
ALERT = _a("alert", [
    Keyframe(0.00, both={"w": 36, "h": 36}, face={"brightness": 1.0}),
    Keyframe(0.10, both={"w": 42, "h": 46}, face={"brightness": 1.0}, ease="back"),
    Keyframe(0.25, face={"brightness": 0.35}),
    Keyframe(0.40, face={"brightness": 1.0}),
    Keyframe(0.55, face={"brightness": 0.35}),
    Keyframe(0.70, face={"brightness": 1.0}),
], priority=P_ALERT, interruptible=False, blink_suppress=True, tags=["alert"])

ERROR = _a("error", [
    Keyframe(0.00, both={"lid_top": 0.0, "radius": 10}),
    Keyframe(0.12, both={"lid_top": 0.35, "lid_angle": 25, "radius": 3, "w": 34},
             ease="in"),
    Keyframe(0.24, both={"x": -3}),
    Keyframe(0.34, both={"x": 3}),
    Keyframe(0.44, both={"x": -2}),
    Keyframe(0.54, both={"x": 0}),
    Keyframe(1.40, both={"lid_top": 0.30}),
], priority=P_ALERT, interruptible=False, tags=["alert", "negative"])

BOOT = _a("boot", [
    # Power-on. A thin line that opens, like a CRT waking up.
    Keyframe(0.00, both={"open": 0.0, "w": 4}, face={"brightness": 0.0}),
    Keyframe(0.25, both={"open": 0.03, "w": 44}, face={"brightness": 1.0}, ease="out"),
    Keyframe(0.55, both={"open": 0.03, "w": 44}),
    Keyframe(0.95, both={"open": 1.0, "w": 36}, ease="out"),
    Keyframe(1.30, both={"open": 1.0}),
], priority=P_CRITICAL, interruptible=False, blink_suppress=True, tags=["system"])

SHUTDOWN = _a("shutdown", [
    Keyframe(0.00, both={"open": 1.0, "w": 36}, face={"brightness": 1.0}),
    Keyframe(0.40, both={"open": 0.04, "w": 42}, ease="in"),
    Keyframe(0.75, both={"w": 3}, face={"brightness": 0.6}, ease="in"),
    Keyframe(1.00, both={"w": 0}, face={"brightness": 0.0}),
], priority=P_CRITICAL, interruptible=False, blink_suppress=True, tags=["system"])


# ----------------------------------------------------------------------
BUILTINS: Dict[str, Animation] = {
    a.name: a for a in [
        BLINK, DOUBLE_BLINK, WINK,
        IDLE, LOOK_AROUND, SLEEPY, SLEEPING,
        HAPPY, VERY_HAPPY, SAD, ANGRY, ANNOYED, SURPRISED, CONFUSED,
        SKEPTICAL, CURIOUS, LOVE,
        THINKING, LISTENING, SPEAKING, PROCESSING, ACKNOWLEDGE, REFUSE,
        ALERT, ERROR, BOOT, SHUTDOWN,
    ]
}


def load_library(directory: str = "animations") -> Dict[str, Animation]:
    """Builtins, overridden by any JSON in `directory`.

    Files win over builtins deliberately: that is what makes the editor
    useful. Tweak `happy.json`, drop it in, restart -- no code change.
    """
    import json
    import logging
    import os

    log = logging.getLogger("ash.face.library")
    lib = dict(BUILTINS)
    if not os.path.isdir(directory):
        return lib

    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(directory, fn)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                anim = Animation.from_dict(json.load(fh))
            if anim.name in lib:
                log.info("Animation '%s' overridden by %s", anim.name, fn)
            lib[anim.name] = anim
        except Exception:
            log.exception("Skipping bad animation file %s", path)
    return lib


def export_builtins(directory: str = "animations/builtin"):
    """Dump every builtin as JSON so they can be opened in the editor."""
    import os

    os.makedirs(directory, exist_ok=True)
    for name, anim in BUILTINS.items():
        with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as fh:
            fh.write(anim.to_json())
    return len(BUILTINS)
