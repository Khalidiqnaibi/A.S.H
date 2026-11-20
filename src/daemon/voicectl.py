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
                  r"screen\s*(?:capture|shots?|reading)|(?:turn|switch)\s+(?:the\s+)?screen\s*(?:capture|shots?|reading)\s+on"
                  r"|screen\s*(?:capture|shots?)\s+on"),
    ("screen_off", r"(?:turn|switch)\s+off\s+(?:the\s+)?screen\s*(?:capture|shots?|reading)|(?:disable|stop)\s+(?:the\s+)?"
                   r"screen\s*(?:capture|shots?|reading)|(?:turn|switch)\s+(?:the\s+)?screen\s*(?:capture|shots?|reading)\s+off"
                   r"|screen\s*(?:capture|shots?)\s+off"),
    ("sleep", r"mute(?:\s+(?:yourself|ash))?|go\s+to\s+sleep|(?:time\s+to\s+)?sleep\s+now|take\s+a\s+nap|quiet\s+mode|do\s+not\s+disturb|"
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

# "Mic off, camera off." / "turn off the mic and the camera" / "switch the camera off and the mic off" / "cam off": several devices in one line,
# each with its own direction. Today five tries at this went to the model (and three answers were cancelled by the next try), because the
# rules above only know one device and one fixed phrasing.
_DEVICE = {"mic": "mic", "mics": "mic", "mike": "mic", "microphone": "mic", "microphones": "mic",
           "camera": "camera", "cameras": "camera", "cam": "camera", "webcam": "camera"}
_DIR_OFF = frozenset({"off", "disable", "disabled", "kill", "stop", "mute", "deactivate", "close", "of"})     # "of": whisper's "off"
_DIR_ON = frozenset({"on", "enable", "enabled", "start", "unmute", "activate", "open"})
_GLUE = frozenset("turn switch shut put set the your my please and then also ash can you could would now both for me it them to a up both together all".split())
_QUESTION = frozenset("what whats what's why how who when where which is are does did do am".split())


def parse_many(text: str) -> List[Command]:
    """Every device command in one line, in order: [mic_off, camera_off]. A direction applies to the devices before it ("mic and camera off")
    or, said first, to the ones after it ("turn off the mic and the camera"). Anything else in the line, or a question, gives []."""
    t = re.sub(r"[^a-z\s']", " ", (text or "").lower().replace("\u2019", "'"))
    words = t.split()
    if not words or len(words) > 12 or re.search(r"\?", text or ""):
        return []
    if words[0] in _QUESTION:
        return []
    pending: List[str] = []
    lead: Optional[str] = None
    out: List[Command] = []
    seen_dev = False
    for i, w in enumerate(words):
        if w in _DEVICE:
            seen_dev = True
            d = _DEVICE[w]
            if lead is not None:
                out.append(Command(f"{d}_{lead}"))
            else:
                pending.append(d)
        elif w in _DIR_OFF or w in _DIR_ON:
            if w == "of" and any(x not in _GLUE for x in words[i + 1:]):
                return []                                    # "of" is only whisper's "off" at the end of the line ("camera of ASH")
            direction = "off" if w in _DIR_OFF else "on"
            if pending:
                out.extend(Command(f"{d}_{direction}") for d in pending)
                pending = []
            else:
                lead = direction
        elif w in _GLUE:
            continue
        else:
            return []                                        # something else is being said: not for this
    if pending or not seen_dev:
        return []                                            # a device with no direction, or no device at all
    names = []
    for c in out:
        if c.name not in names:
            names.append(c.name)
    return [Command(n) for n in names]


FACES: Dict[str, str] = {
    "smile": "happy", "happy": "happy", "happy face": "happy", "big smile": "very_happy", "very happy": "very_happy",
    "excited": "very_happy", "sad": "sad", "angry": "angry", "mad": "angry", "surprised": "surprised",
    "shocked": "surprised", "sleepy": "sleepy", "tired": "sleepy", "confused": "confused", "curious": "curious",
    "love": "love", "in love": "love", "skeptical": "skeptical", "doubtful": "skeptical", "annoyed": "annoyed",
    "wink": "wink", "thinking": "thinking", "thoughtful": "thinking", "nod": "acknowledge", "look around": "look_around",
    # said the way people say them ("can you do a crying face?"): the nearest face it has
    "crying": "sad", "cry": "sad", "sobbing": "sad", "tearful": "sad", "upset": "sad", "unhappy": "sad", "heartbroken": "sad",
    "laughing": "very_happy", "laugh": "very_happy", "grinning": "very_happy", "joyful": "very_happy", "delighted": "very_happy",
    "furious": "angry", "grumpy": "angry", "cross": "annoyed", "irritated": "annoyed", "bored": "sleepy", "exhausted": "sleepy",
    "puzzled": "confused", "baffled": "confused", "shocked face": "surprised", "amazed": "surprised", "astonished": "surprised",
    "suspicious": "skeptical", "unsure": "skeptical", "interested": "curious", "winking": "wink", "smiling": "happy", "cheerful": "happy",
    "frown": "angry", "grin": "very_happy", "pout": "sad", "sulk": "sad", "yawn": "sleepy", "blush": "love", "smirk": "skeptical",
    "giggle": "very_happy",
}
AGAIN = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:do\s+(?:it|that)\s+again|again|once\s+more|one\s+more\s+time|another\s+one)\s*[.!?]*\s*$", re.I)
# "Wink using your face", "Cry for me, Ash", "Cry.", "laugh for me" went to the model, which said "I can't move my face". A face said
# as a verb, with "for me" / "at me" / "using your face" after it; and whisper's "Link using your face" is the wink it sounds like.
_FACE_FOR = r"(?:\s+(?:for\s+me|at\s+me|(?:using|with)\s+(?:your\s+|the\s+)?face|now|again))*"
_FACE = re.compile(_A + r"(?:(?P<smile>smile|wink|nod)" + _FACE_FOR +
                   r"|(?P<v>cry|laugh|frown|grin|pout|sulk|yawn|blush|smirk|giggle)" + _FACE_FOR +
                   r"|(?P<misheard>link|wing|ink)\s+(?:using|with)\s+(?:your\s+|the\s+)?face"
                   r"|(?:make|show\s+me|give\s+me|do)\s+(?:a|an|your|the)\s+(?P<a>[a-z ]{3,15}?)\s*(?:face|look|expression)"
                   r"|(?:look|be|act)\s+(?P<b>[a-z ]{3,15}?)"
                   r"|(?:show\s+me|make|give\s+me|do)\s+(?:a\s+)?(?P<c>big\s+smile|smile|wink))" + _Z, re.I)


@dataclass
class Command:
    name: str                  # privacy_on, mic_off, face, ...
    arg: str = ""              # the animation, for a face
    seconds: float = 0.0       # "for 10 minutes": how long, for the ones that come back by themselves

    @property
    def turns_on(self) -> bool:
        """Turning something on (or showing a face) needs a line meant for ASH."""
        return self.name in ("privacy_off", "mic_on", "camera_on", "screen_on", "wake", "shutdown", "face", "faces",
                             "status")


# "mute microphone for 10 minutes", "privacy mode until 10 minutes" (as it was heard), "mic off for ten minutes": switched off, and
# switched back on by itself. Only the ones that make ASH more private take a time.
_DUR = re.compile(r"[,\s]+(?:for|until|till|in)\s+(?:the\s+next\s+)?(?:(?P<a>an?|one)|(?P<n>\d+(?:\.\d+)?))\s*"
                  r"(?P<u>hours?|hrs?|minutes?|mins?|seconds?|secs?)\s*[.!?]*\s*$", re.I)
TIMED = ("mic_off", "camera_off", "screen_off", "privacy_on")


def parse(text: str) -> Optional[Command]:
    t = (text or "").strip()
    if not t or len(t) > 90:
        return None
    from .numwords import digits
    t = digits(t)
    seconds = 0.0
    m = _DUR.search(t)
    if m:
        n = 1.0 if m.group("a") else float(m.group("n"))
        unit = m.group("u").lower()
        seconds = n * (3600.0 if unit.startswith("h") else 60.0 if unit.startswith("m") else 1.0)
        base = t[:m.start()]
        for name, rx in _COMPILED:
            if rx.match(base):
                if name in TIMED and 10.0 <= seconds <= 24 * 3600.0:
                    return Command(name, seconds=seconds)
                return None                       # "turn on the camera for ten minutes" is not something it does
    for name, rx in _COMPILED:
        if rx.match(t):
            return Command(name)
    m = _FACE.match(t)
    if m:
        word = (m.group("smile") or m.group("v") or m.group("a") or m.group("b") or m.group("c") or "").lower().strip()
        if m.group("misheard"):
            word = "wink"
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


# A model answer that STATES whether a device is on: "Mic and camera are both already off." (said at 22:22 while the microphone was on -- it was
# switched off at 22:23). The model cannot see the sensors, so such a sentence is checked against the real state.
_DEV_WORDS = {"mic": "microphone", "mics": "microphone", "microphone": "microphone", "mike": "microphone",
              "camera": "camera", "cam": "camera", "webcam": "camera"}
_STATE_OFF = ("off", "disabled", "muted", "inactive", "switched off", "turned off", "not running", "not on")
_STATE_ON = ("on", "enabled", "active", "running", "switched on", "turned on", "live", "listening")
_STATES = re.compile(r"\b(?:already|currently|right now|both|still)?\s*(" + "|".join(sorted(map(re.escape, _STATE_OFF + _STATE_ON), key=len, reverse=True)) + r")\b", re.I)
_BE = re.compile(r"\b(?:is|are|'s|was|were|remains?|stays?)\b|\balready\b|\bboth\b", re.I)
_INSTRUCTION = re.compile(r"\b(?:turn|switch|shut|put|set|to|can|could|should|would|if|when|whenever|unless|say|type|press|click)\b[^.!?]{0,30}$", re.I)


def device_claims(reply: str) -> Dict[str, bool]:
    """{"microphone": False, "camera": False} for the devices a reply says are off (False) or on (True); {} when it says nothing of the kind.
    A suggestion ("you can turn off the camera"), a condition ("if the mic is on") or a question is not a claim."""
    out: Dict[str, bool] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", str(reply or "")):
        low = sentence.lower()
        if "?" in sentence or not _BE.search(low):
            continue
        devs = [d for w, d in _DEV_WORDS.items() if re.search(r"\b" + w + r"\b", low)]
        if not devs:
            continue
        first = min(m.start() for w in _DEV_WORDS for m in re.finditer(r"\b" + w + r"\b", low))
        if re.match(r"\s*(?:if|when|whenever|unless|once)\b", low) or _INSTRUCTION.search(low[:first]):
            continue
        m = _STATES.search(low, first)
        if not m:
            continue
        on = m.group(1).lower() in _STATE_ON
        for d in devs:
            out[d] = on
    return out


def device_truth(claims: Dict[str, bool], actual: Dict[str, bool]) -> Optional[str]:
    """None when every claim is true; otherwise the real state of the devices it talked about, said plainly."""
    wrong = [d for d, on in claims.items() if d in actual and actual[d] != on]
    if not wrong:
        return None
    bits = [f"the {d} is {'on' if actual[d] else 'off'}" for d in claims if d in actual]
    return "Let me correct that: " + " and ".join(bits) + "."


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
