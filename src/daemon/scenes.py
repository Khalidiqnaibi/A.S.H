"""
src/daemon/scenes.py

"Work mode", "wind down", "break time": one phrase that sets the whole mood.

A scene can do four things, and the reply says only the ones that really
happened:

    quiet     mute what ASH volunteers for N minutes (it still answers you),
              back on its own after -- the same quiet as "quiet till 9"
    face      the face shows a feeling (an existing animation)
    apps      open some apps (only the ones listed, by the names the desktop
              module knows: "code", "chrome" ... or any you added under desktop.apps)
    say       a line to speak

Three come built in; daemon.json's `scenes` adds or changes them:

    "scenes": {
        "work":  {"quiet_min": 120, "apps": ["code", "chrome"], "say": "Work mode. Two hours, no interruptions."},
        "movie": {"quiet_min": 150, "face": "sleepy", "say": "Enjoy the movie."}
    }

"End work mode", "I'm back", "scene off" ends the current one. "What scenes do
you have?" lists them. Rules and the clock only: no language model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_A = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:please\s+)?"
_Z = r"\s*[.!]*\s*$"


@dataclass
class Scene:
    name: str
    say: str = ""
    quiet_min: float = 0.0
    face: str = ""
    apps: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


BUILTIN: Dict[str, Scene] = {
    "work": Scene("work", say="Work mode. I'll stay out of your way.", quiet_min=90, face="curious",
                  aliases=["deep work", "work time", "get to work"]),
    "wind down": Scene("wind down", say="Winding down. I'll keep things calm.", quiet_min=120, face="sleepy",
                       aliases=["evening", "relax", "chill", "unwind", "bedtime"]),
    "break": Scene("break", say="Take a proper break. I'll be back in a bit.", quiet_min=15, face="happy",
                   aliases=["coffee break", "lunch", "lunch break", "stretch"]),
}

_END = re.compile(_A + r"(?:(?:end|stop|exit|leave|finish|cancel)\s+(?:the\s+)?(?:(?P<n>[a-z ]{2,20}?)\s+(?:mode|scene|time)|(?:scene|mode))"
                  r"|scene\s+off|i'?m\s+back|back\s+to\s+normal)" + _Z, re.I)
_LIST = re.compile(_A + r"(?:what|which)\s+(?:scenes|modes)\s+(?:do\s+you\s+have|are\s+there|can\s+you\s+do)|list\s+(?:the\s+|your\s+)?(?:scenes|modes)"
                   + _Z, re.I)
_START = re.compile(_A + r"(?P<v>(?:start|begin|enter|switch\s+to|go\s+(?:in)?to|activate|turn\s+on|time\s+for|it'?s)\s+)?(?:(?:the|my)\s+)?"
                    r"(?P<n>[a-z][a-z ]{1,20}?)(?P<suf>\s+(?:mode|scene|time))?" + _Z, re.I)


class Scenes:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.scenes: Dict[str, Scene] = {k: Scene(v.name, v.say, v.quiet_min, v.face, list(v.apps), list(v.aliases))
                                         for k, v in BUILTIN.items()}
        for name, d in (cfg or {}).items():
            if not isinstance(d, dict):
                continue
            key = str(name).strip().lower()
            base = self.scenes.get(key) or Scene(key)
            try:
                base.quiet_min = float(d.get("quiet_min", base.quiet_min) or 0)
            except (TypeError, ValueError):
                pass
            base.say = str(d.get("say", base.say) or "")
            base.face = str(d.get("face", base.face) or "")
            apps = d.get("apps", base.apps)
            base.apps = [str(a) for a in apps] if isinstance(apps, (list, tuple)) else base.apps
            base.aliases = [str(a).lower() for a in d.get("aliases", base.aliases)]
            self.scenes[key] = base
        self.current: Optional[str] = None

    def find(self, phrase: str) -> Optional[Scene]:
        p = re.sub(r"\s+", " ", (phrase or "").strip().lower())
        for s in self.scenes.values():
            if p == s.name or p in s.aliases:
                return s
        return None

    def parse(self, text: str) -> Optional[Tuple[str, Optional[Scene]]]:
        """("start", scene) | ("end", scene-or-None) | ("list", None) | None."""
        t = (text or "").strip()
        if not t or len(t) > 70:
            return None
        if _LIST.match(t):
            return "list", None
        m = _END.match(t)
        if m:
            n = (m.group("n") or "").strip()
            return "end", self.find(n) if n else None
        m = _START.match(t)
        if m:
            s = self.find(m.group("n"))
            # A bare "evening" or "lunch" is just a word; only a scene's own name
            # starts it alone. An alias needs "start ..." or "... mode".
            if s is not None and (m.group("v") or m.group("suf") or m.group("n").strip().lower() == s.name and " " in s.name):
                return "start", s
        return None

    def listing(self) -> str:
        names = ", ".join(s.name for s in self.scenes.values())
        return f"I have these scenes: {names}. Say a name, like “work mode”, to start one."
