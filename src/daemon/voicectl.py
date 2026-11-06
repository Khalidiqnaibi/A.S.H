"""
src/daemon/voicectl.py

Controlling ASH by voice -- what the console's buttons do, said out loud.

    privacy mode / go private              everything sensitive off (mic,
                                           camera, screen); typed "privacy
                                           off" in the console turns it back
    privacy off / resume
    turn off the mic / stop listening      just the microphone
    turn on / off the camera               (on only when said to ASH)
    turn on / off screen capture           (on only when said to ASH)
    go to sleep / quiet mode               nothing volunteered, sleepy face;
                                           "wake up" ends it
    turn off / shut down                   asks first, then stops the daemon
    smile / look surprised / wink          a face, from the face animations
    what faces can you make?               lists them
    what's your status / CPU / memory?     the real numbers, not a guess

Rule-based, and each one says what it actually did. Before this, all of
these went to the language model, which answered "Go to privacy mode." or
"I'm turning off now. Good night." and did nothing.

Turning something off is allowed from any line the microphone hears (it can
only make ASH more private); turning something on needs a line meant for
ASH.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_A = r"^\s*(?:(?:hey|hi|ok(?:ay)?)\s+)?(?:(?:ash|ass)[,\s]+)?(?:(?:please|can\s+you|could\s+you|would\s+you)[,\s]+)*"
_Z = r"(?:[,\s]+(?:ash|please|now|for\s+me|right\s+now|for\s+a\s+while))*\s*[.!?]*\s*$"

_RULES = [
    ("privacy_on", r"(?:go\s+(?:in)?to\s+|enter\s+|turn\s+on\s+|switch\s+(?:to|on)\s+|activate\s+|enable\s+|start\s+)?"
                   r"privacy(?:\s+mode)?(?:\s+on)?|go\s+private|pause\s+(?:everything|all\s+(?:the\s+)?sensors|your\s+sensors)"
                   r"|stop\s+(?:watching\s+and\s+listening|listening\s+and\s+watching|watching\s+me)"),
    ("privacy_off", r"(?:turn\s+off|exit|leave|end|disable|stop)\s+privacy(?:\s+mode)?|privacy(?:\s+mode)?\s+off"
                    r"|resume(?:\s+(?:everything|sensors|listening))?|un-?pause"),
    ("mic_off", r"(?:turn|switch|shut)\s+off\s+(?:the\s+|your\s+)?(?:mic|microphone)|(?:turn|switch|shut)\s+(?:the\s+|your\s+)?"
                r"(?:mic|microphone)\s+off|(?:mic|microphone)\s+off|stop\s+listening|mute\s+(?:the\s+|your\s+)?(?:mic|microphone)"),
    ("mic_on", r"(?:turn|switch)\s+on\s+(?:the\s+|your\s+)?(?:mic|microphone)|(?:turn|switch)\s+(?:the\s+|your\s+)?"
               r"(?:mic|microphone)\s+on|(?:mic|microphone)\s+on|start\s+listening|un-?mute\s+(?:the\s+|your\s+)?(?:mic|microphone)"),
    ("camera_on", r"(?:turn|switch)\s+on\s+(?:the\s+|your\s+)?camera|(?:turn|switch)\s+(?:the\s+|your\s+)?camera\s+on"
                  r"|(?:activate|enable|start|open)\s+(?:the\s+|your\s+)?camera|camera\s+on"),
    ("camera_off", r"(?:turn|switch|shut)\s+off\s+(?:the\s+|your\s+)?camera|(?:turn|switch)\s+(?:the\s+|your\s+)?camera\s+off"
                   r"|(?:deactivate|disable|stop|close)\s+(?:the\s+|your\s+)?camera|camera\s+off"),
    ("screen_on", r"(?:turn|switch)\s+on\s+(?:the\s+)?screen\s*(?:capture|shots?|reading)|(?:enable|activate|start)\s+(?:the\s+)?"
                  r"screen\s*(?:capture|shots?|reading)|screen\s*(?:capture|shots?)\s+on"),
    ("screen_off", r"(?:turn|switch)\s+off\s+(?:the\s+)?screen\s*(?:capture|shots?|reading)|(?:disable|stop)\s+(?:the\s+)?"
                   r"screen\s*(?:capture|shots?|reading)|screen\s*(?:capture|shots?)\s+off"),
    ("sleep", r"go\s+to\s+sleep|(?:time\s+to\s+)?sleep\s+now|take\s+a\s+nap|quiet\s+mode|do\s+not\s+disturb|"
              r"be\s+quiet\s+for\s+a\s+(?:while|bit)|stop\s+talking\s+for\s+a\s+while"),
    ("wake", r"wake\s+up|you\s+can\s+talk\s+again|un-?mute(?:\s+yourself)?|end\s+quiet\s+mode"),
    ("shutdown", r"(?:turn|shut|power)\s+(?:yourself\s+)?(?:off|down)|shut\s*down|go\s+offline|switch\s+(?:yourself\s+)?off"),
    ("faces", r"(?:what|which)\s+(?:faces|expressions|animations|emotes?|emotions)\s+(?:do\s+you\s+have|can\s+you\s+(?:do|make|show))"
              r"|show\s+me\s+(?:your|the|all\s+(?:your|the))\s+(?:faces|expressions|animations|emotes?)"
              r"|(?:list|show)\s+(?:your\s+)?(?:faces|expressions|animations|emotes?)"),
    ("status", r"(?:what'?s|what\s+is|how'?s|how\s+is|give\s+me)\s+(?:your|the)\s+(?:system\s+)?(?:status|cpu|cbu|memory|ram|load)"
               r"(?:\s+(?:status|usage|like|at|as))*|status\s+report|system\s+status|(?:cpu|cbu|memory|ram)\s+(?:status|usage)"
               r"|how\s+(?:is|'s)\s+the\s+(?:system|computer|pc|laptop)\s+doing"),
]
_COMPILED = [(name, re.compile(_A + r"(?:" + pat + r")" + _Z, re.I)) for name, pat in _RULES]

FACES: Dict[str, str] = {
    "smile": "happy", "happy": "happy", "happy face": "happy", "big smile": "very_happy", "very happy": "very_happy",
    "excited": "very_happy", "sad": "sad", "angry": "angry", "mad": "angry", "surprised": "surprised",
    "shocked": "surprised", "sleepy": "sleepy", "tired": "sleepy", "confused": "confused", "curious": "curious",
    "love": "love", "in love": "love", "skeptical": "skeptical", "doubtful": "skeptical", "annoyed": "annoyed",
    "wink": "wink", "thinking": "thinking", "thoughtful": "thinking", "nod": "acknowledge", "look around": "look_around",
}
_FACE = re.compile(_A + r"(?:(?P<smile>smile|wink|nod)(?:\s+at\s+me)?"
                   r"|(?:make|show\s+me|give\s+me|do)\s+(?:a|an|your|the)\s+(?P<a>[a-z ]{3,15}?)\s*(?:face|look|expression)"
                   r"|(?:look|be|act)\s+(?P<b>[a-z ]{3,15}?)"
                   r"|(?:show\s+me|make|give\s+me|do)\s+(?:a\s+)?(?P<c>big\s+smile|smile|wink))" + _Z, re.I)


@dataclass
class Command:
    name: str                  # privacy_on, mic_off, face, ...
    arg: str = ""              # the animation, for a face

    @property
    def turns_on(self) -> bool:
        """Turning something on (or showing a face) needs a line meant for ASH."""
        return self.name in ("privacy_off", "mic_on", "camera_on", "screen_on", "wake", "shutdown", "face", "faces",
                             "status")


def parse(text: str) -> Optional[Command]:
    t = (text or "").strip()
    if not t or len(t) > 90:
        return None
    for name, rx in _COMPILED:
        if rx.match(t):
            return Command(name)
    m = _FACE.match(t)
    if m:
        word = (m.group("smile") or m.group("a") or m.group("b") or m.group("c") or "").lower().strip()
        anim = FACES.get(word)
        if anim:
            return Command("face", anim)
    return None


# A model answer that says it did one of these -- which it cannot do.
_CLAIM = re.compile(r"\b(?:i'?m|i\s+am|i'?ve|i\s+have|i'?ll|i\s+will|now)\s+(?:now\s+)?(?:going\s+to\s+)?"
                    r"(?:turning\s+(?:myself\s+)?(?:off|on)|turned\s+(?:myself\s+)?(?:off|on)|shutting\s+down|shut\s+down|"
                    r"going\s+(?:in)?to\s+(?:sleep|privacy\s+mode)|gone\s+(?:in)?to\s+(?:sleep|privacy\s+mode)|"
                    r"(?:turn|turning|turned|switch|switching|switched)\s+(?:off|on)\s+(?:the\s+|my\s+)?(?:mic|microphone|camera|screen)|"
                    r"activat(?:e|ing|ed)\s+(?:the\s+|my\s+)?(?:camera|privacy\s+mode|screen))\b"
                    r"|\bas\s+i\s+go\s+to\s+sleep\b"
                    r"|^\s*(?:go(?:ing)?\s+to\s+privacy\s+mode|privacy\s+mode\s+(?:on|activated))\.?\s*$", re.I)


def claimed_action(reply: str) -> bool:
    """Does a model answer claim to have done (or be doing) something that
    only a voice command can do?"""
    return bool(_CLAIM.search(reply or ""))


HONEST = ("I can't do that from a chat answer. Say it as a command -- \u201cprivacy mode\u201d, "
          "\u201cturn off the mic\u201d, \u201cturn on the camera\u201d, \u201cgo to sleep\u201d or "
          "\u201cshut down\u201d -- and I'll do it.")


def status_line(stats: Optional[Dict], sensors: List[str], muted: bool, paused: bool) -> str:
    """The real numbers, said plainly."""
    parts = []
    if stats:
        parts.append(f"CPU at {stats['cpu']:.0f}%, memory at {stats['ram']:.0f}%"
                     + (f" ({stats['ram_used_gb']:.1f} of {stats['ram_total_gb']:.0f} GB)" if stats.get("ram_total_gb") else ""))
        if stats.get("disk_free_gb") is not None:
            parts.append(f"{stats['disk_free_gb']:.0f} GB free on the disk")
        if stats.get("battery") is not None:
            parts.append(f"battery {stats['battery']:.0f}%" + (" and charging" if stats.get("plugged") else ""))
    if paused:
        parts.append("privacy mode is on")
    else:
        parts.append("using " + (", ".join(sensors) if sensors else "no sensors"))
    if muted:
        parts.append("I'm in quiet mode")
    s = "; ".join(parts) + "."
    return s[0].upper() + s[1:]


def read_stats() -> Optional[Dict]:
    try:
        import os
        import shutil

        import psutil
        mem = psutil.virtual_memory()
        out = {"cpu": psutil.cpu_percent(interval=0.2), "ram": mem.percent,
               "ram_used_gb": mem.used / 2 ** 30, "ram_total_gb": mem.total / 2 ** 30}
        try:
            out["disk_free_gb"] = shutil.disk_usage(os.path.abspath(os.sep)).free / 2 ** 30
        except Exception:
            out["disk_free_gb"] = None
        try:
            b = psutil.sensors_battery()
            out["battery"], out["plugged"] = (b.percent, b.power_plugged) if b else (None, None)
        except Exception:
            out["battery"] = None
        return out
    except Exception:
        return None
