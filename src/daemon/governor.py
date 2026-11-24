"""
src/daemon/governor.py

Getting out of the machine's way when the machine is struggling -- not only when a game is up.

On the real machine, memory sat at 95% and a single answer cost 60-73 seconds (43 s of the model phrasing it, 21 s more nominating a tool),
while ASH itself held a language model, a speech model, an OCR engine and a sentence encoder in RAM. `gamemode.py` already knows how to step
aside -- lower its own priority, pause screen reading, let go of the vision model, tell Ollama to unload -- but only for a game. This watches
the machine's own numbers and asks the same machinery to do it when memory is nearly gone or the CPU is saturated, and to give it all back
once things have calmed down.

Hysteresis, so it does not flap: it takes memory high for a couple of minutes to step aside, and memory clearly low for a few to come back.
The microphone is left on (the person is still talking to it); only what is heavy and optional is paused. CPU is a weaker signal -- ASH's
own model is a large share of it -- so it only counts when it is pinned for a long while.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("ash.daemon.governor")


@dataclass
class GovernorConfig:
    enabled: bool = True
    memory_high: float = 88.0         # percent of RAM in use that counts as pressure...
    memory_low: float = 75.0          # ...and the level it has to fall to before ASH comes back
    cpu_high: float = 97.0            # pinned CPU (a weaker signal: ASH's own model is a big part of it)
    cpu_low: float = 75.0
    memory_for_s: float = 120.0       # memory must stay high this long
    cpu_for_s: float = 240.0          # CPU must stay pinned this long
    calm_for_s: float = 180.0         # both below the low marks this long before everything is given back
    sample_s: float = 15.0            # how often to look

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GovernorConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k):
                try:
                    setattr(c, k, type(getattr(c, k))(v))
                except (TypeError, ValueError):
                    pass
        return c


def read_machine() -> Tuple[Optional[float], Optional[float]]:
    """(memory %, cpu %) now; None for what cannot be read."""
    try:
        import psutil
        return float(psutil.virtual_memory().percent), float(psutil.cpu_percent(interval=None))
    except Exception:
        return None, None


class Governor:
    def __init__(self, game, cfg: Optional[GovernorConfig] = None, clock: Callable[[], float] = time.time,
                 probe: Callable[[], Tuple[Optional[float], Optional[float]]] = read_machine):
        self.game = game                  # a GameMode (gamemode.py): the machinery that steps aside
        self.cfg = cfg or GovernorConfig()
        self.clock = clock
        self.probe = probe
        self._last = -1e18
        self._mem_hot = 0.0               # since when memory has been above the high mark (0 = not)
        self._cpu_hot = 0.0
        self._calm = 0.0                  # since when both have been below the low marks
        self.mem: Optional[float] = None
        self.cpu: Optional[float] = None
        self.stats = {"squeezed": 0, "relaxed": 0}

    @property
    def squeezed(self) -> bool:
        return bool(getattr(self.game, "squeezed", False))

    def tick(self, now: Optional[float] = None) -> Optional[str]:
        """Look at the machine (every sample_s). Returns "squeezed" / "relaxed" when it changed something, else None."""
        if not self.cfg.enabled or self.game is None:
            return None
        now = self.clock() if now is None else now
        if now - self._last < self.cfg.sample_s:
            return None
        self._last = now
        try:
            mem, cpu = self.probe()
        except Exception:
            logger.debug("machine not read", exc_info=True)
            return None
        self.mem, self.cpu = mem, cpu
        c = self.cfg
        if mem is not None and mem >= c.memory_high:
            self._mem_hot = self._mem_hot or now
        else:
            self._mem_hot = 0.0
        if cpu is not None and cpu >= c.cpu_high:
            self._cpu_hot = self._cpu_hot or now
        else:
            self._cpu_hot = 0.0
        calm = (mem is None or mem <= c.memory_low) and (cpu is None or cpu <= c.cpu_low)
        self._calm = (self._calm or now) if calm else 0.0

        if not self.squeezed:
            why = ""
            if self._mem_hot and now - self._mem_hot >= c.memory_for_s:
                why = f"memory at {mem:.0f}%"
            elif self._cpu_hot and now - self._cpu_hot >= c.cpu_for_s:
                why = f"the CPU pinned at {cpu:.0f}%"
            if why and self.game.squeeze(why, now):
                self.stats["squeezed"] += 1
                logger.warning("Resource pressure (%s): stepping aside", why)
                return "squeezed"
        elif self._calm and now - self._calm >= c.calm_for_s:
            if self.game.relax(now):
                self.stats["relaxed"] += 1
                logger.warning("Machine calm again: back to normal")
                return "relaxed"
        return None

    def status(self) -> Dict[str, Any]:
        return {"enabled": self.cfg.enabled, "squeezed": self.squeezed, "memory": self.mem, "cpu": self.cpu,
                "why": getattr(self.game, "squeeze_why", "") if self.squeezed else "",
                "squeezes": self.stats["squeezed"], "relaxes": self.stats["relaxed"]}

    def report(self) -> str:
        if not self.cfg.enabled:
            return "The resource governor is off (governor.enabled in daemon.json)."
        look = ""
        if self.mem is not None:
            look = f"Memory {self.mem:.0f}%" + (f", CPU {self.cpu:.0f}%" if self.cpu is not None else "") + ". "
        if self.squeezed:
            return (look + f"I'm keeping out of the way ({self.game.squeeze_why}): lower priority, screen reading paused, the language "
                    f"model unloaded. I come back when it has been calm for a few minutes.")
        return look + "The machine is fine, so I'm running normally."
