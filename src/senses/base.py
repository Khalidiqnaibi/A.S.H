"""
src/senses/base.py

The periphery: everything that touches the world outside the process.

Two kinds of thing live here, and they are deliberately asymmetric:

    Sensor    -- produces SensorEvents. Cheap, safe, always allowed to run.
    Actuator  -- changes the world. Gated, permissioned, veto-able by the
                 VLA channel, dry-run by default.

Adding hardware should never require editing the brain. A new sensor is one
class with a `read()`; a new actuator is one class with an `act()` plus a
declared `ActionClass`. Both self-register, both declare their capabilities,
and both report honestly when their backing library isn't installed rather
than crashing the daemon at 3am.

Availability is a first-class concept
-------------------------------------
An always-on system runs on machines where half the optional dependencies are
missing, the webcam is unplugged, and the user revoked screen-recording
permission last Tuesday. Every peripheral therefore has three states:

    UNAVAILABLE  -- backing library or device missing. Never retried hot.
    DISABLED     -- available but switched off by config or the user.
    ACTIVE       -- running.

The daemon reports all three at startup so you can see at a glance what your
ASH can actually perceive, instead of discovering silently-dead sensors weeks
later.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("ash.senses")


class State(str, Enum):
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    ACTIVE = "active"
    ERROR = "error"


class Modality(str, Enum):
    """What kind of percept a sensor produces. Maps onto the brain's
    sensory extractors -- `vision` goes to the ViT path, everything else is
    described in text and goes to the language path."""

    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    TELEMETRY = "telemetry"


@dataclass
class SensorEvent:
    """One observation. This is what flows into the ambient loop.

    `urgency` is the sensor's own claim that this needs a human told about it
    right now (battery at 3%, a build failed, motion in an empty house). The
    attention gate treats it as evidence, not as a command -- a sensor cannot
    force ASH to speak.
    """

    source: str                       # sensor name
    modality: Modality = Modality.TELEMETRY
    text: str = ""                    # human/LLM-readable description
    data: Any = None                  # raw payload (frame, samples, dict)
    salience: float = 0.3             # 0..1 how much this stands out
    urgency: float = 0.0              # 0..1 sensor's claim on attention
    addressed: bool = False           # user directly spoke to ASH
    ts: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)
    # If False, the raw payload is never written to disk (privacy).
    persistable: bool = True

    def summary(self) -> str:
        return f"[{self.source}] {self.text}"[:400]

    def as_record(self) -> Dict[str, Any]:
        """Journal-safe form: no binary payloads."""
        return {
            "ts": round(self.ts, 3),
            "source": self.source,
            "modality": self.modality.value,
            "text": self.text[:2000],
            "salience": round(self.salience, 3),
            "urgency": round(self.urgency, 3),
            "addressed": self.addressed,
            "meta": {k: v for k, v in self.meta.items()
                     if isinstance(v, (str, int, float, bool, type(None)))},
        }


@dataclass
class Capability:
    """What a peripheral needs and what it exposes."""

    name: str
    description: str = ""
    requires: List[str] = field(default_factory=list)   # python module names
    platforms: List[str] = field(default_factory=list)  # empty = any
    privacy_sensitive: bool = False   # captures the user or their surroundings


# ----------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------
class Sensor(ABC):
    """Base class for everything that perceives.

    Subclasses implement `read()` (polling) or override `stream()` for
    push-driven sources. Polling is the default because it is trivially
    supervisable: a sensor that hangs gets noticed by the loop's watchdog,
    whereas a wedged callback thread just silently stops producing.
    """

    capability: Capability = Capability(name="sensor")
    default_interval: float = 5.0     # seconds between reads
    modality: Modality = Modality.TELEMETRY

    def __init__(self, interval: Optional[float] = None, enabled: bool = True):
        self.interval = interval if interval is not None else self.default_interval
        self.state = State.DISABLED
        self.last_read: float = 0.0
        self.error_count = 0
        self.event_count = 0
        self._enabled = enabled
        self._probe()

    # -- lifecycle ------------------------------------------------------
    def _probe(self):
        """Decide UNAVAILABLE vs DISABLED vs ACTIVE at construction time."""
        import importlib
        import sys as _sys

        for mod in self.capability.requires:
            try:
                importlib.import_module(mod)
            except Exception:
                self.state = State.UNAVAILABLE
                self.unavailable_reason = f"missing module: {mod}"
                logger.info("Sensor %s unavailable: %s", self.name, self.unavailable_reason)
                return

        if self.capability.platforms:
            plat = _sys.platform
            if not any(plat.startswith(p) for p in self.capability.platforms):
                self.state = State.UNAVAILABLE
                self.unavailable_reason = f"platform {plat} not in {self.capability.platforms}"
                return

        try:
            ok, reason = self.probe()
        except Exception as e:
            ok, reason = False, str(e)
        if not ok:
            self.state = State.UNAVAILABLE
            self.unavailable_reason = reason or "probe failed"
            logger.info("Sensor %s unavailable: %s", self.name, self.unavailable_reason)
            return

        self.state = State.ACTIVE if self._enabled else State.DISABLED

    def probe(self) -> tuple:
        """Device-level availability check. Return (ok, reason)."""
        return True, ""

    @property
    def name(self) -> str:
        return self.capability.name

    def enable(self):
        if self.state == State.DISABLED:
            self.state = State.ACTIVE

    def disable(self):
        if self.state in (State.ACTIVE, State.ERROR):
            self.state = State.DISABLED

    def due(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        return self.state == State.ACTIVE and (now - self.last_read) >= self.interval

    # -- the actual work -------------------------------------------------
    @abstractmethod
    def read(self) -> List[SensorEvent]:
        """Return zero or more events. Returning [] is normal and expected --
        most polls of most sensors observe nothing worth reporting."""

    def poll(self) -> List[SensorEvent]:
        """Read with error containment. A sensor that throws gets backed off,
        then disabled after repeated failure, rather than taking down a
        process that is supposed to run for weeks."""
        self.last_read = time.time()
        try:
            events = self.read() or []
            self.error_count = 0
            self.event_count += len(events)
            return events
        except Exception:
            self.error_count += 1
            logger.exception("Sensor %s read failed (%d consecutive)", self.name, self.error_count)
            # Exponential backoff, then give up.
            self.interval = min(self.interval * 2, 300.0)
            if self.error_count >= 5:
                self.state = State.ERROR
                logger.error("Sensor %s disabled after %d failures", self.name, self.error_count)
            return []

    def close(self):
        """Release devices. Always called on shutdown."""

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "interval": round(self.interval, 2),
            "events": self.event_count,
            "errors": self.error_count,
            "privacy_sensitive": self.capability.privacy_sensitive,
            "reason": getattr(self, "unavailable_reason", None),
        }


# ----------------------------------------------------------------------
# Actuators
# ----------------------------------------------------------------------
class Actuator(ABC):
    """Base class for everything that acts on the world.

    Every actuator declares an `action_class` string that the brain maps onto
    `ActionClass` -- which is what puts it inside the VLA channel's
    jurisdiction and therefore under its veto. An actuator cannot opt out of
    that: the class is read at registration, not asked for at call time.

    `dry_run` defaults to True. An always-on process with keyboard control is
    a liability until you have watched it not do anything stupid for a week,
    so the safe default is to log the intended action and perform nothing.
    """

    capability: Capability = Capability(name="actuator")
    action_class: str = "mutation"     # -> brain.ActionClass
    reversible: bool = False
    intents: List[str] = field(default_factory=list)  # overridden per subclass

    def __init__(self, dry_run: bool = True, enabled: bool = False):
        self.dry_run = dry_run
        self.state = State.DISABLED
        self.call_count = 0
        self._enabled = enabled
        self._probe()

    def _probe(self):
        import importlib
        import sys as _sys

        for mod in self.capability.requires:
            try:
                importlib.import_module(mod)
            except Exception:
                self.state = State.UNAVAILABLE
                self.unavailable_reason = f"missing module: {mod}"
                return
        if self.capability.platforms:
            plat = _sys.platform
            if not any(plat.startswith(p) for p in self.capability.platforms):
                self.state = State.UNAVAILABLE
                self.unavailable_reason = f"platform {plat} unsupported"
                return
        self.state = State.ACTIVE if self._enabled else State.DISABLED

    @property
    def name(self) -> str:
        return self.capability.name

    def enable(self, dry_run: Optional[bool] = None):
        if dry_run is not None:
            self.dry_run = dry_run
        if self.state == State.DISABLED:
            self.state = State.ACTIVE

    def disable(self):
        self.state = State.DISABLED

    @abstractmethod
    def act(self, args: Any) -> Dict[str, Any]:
        """Perform the action. Return {"ok": bool, ...}."""

    def __call__(self, args: Any) -> Dict[str, Any]:
        if self.state != State.ACTIVE:
            return {"ok": False, "error": f"actuator '{self.name}' is {self.state.value}",
                    "reason": getattr(self, "unavailable_reason", None)}
        self.call_count += 1
        if self.dry_run:
            logger.warning("[DRY RUN] %s would execute: %r", self.name, str(args)[:200])
            return {"ok": True, "dry_run": True,
                    "result": f"(dry run) {self.name} would have run with: {str(args)[:200]}"}
        try:
            return self.act(args)
        except Exception as e:
            logger.exception("Actuator %s failed", self.name)
            return {"ok": False, "error": str(e)}

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "dry_run": self.dry_run,
            "action_class": self.action_class,
            "reversible": self.reversible,
            "calls": self.call_count,
            "reason": getattr(self, "unavailable_reason", None),
        }


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
class PeripheryRegistry:
    """Process-wide registry of sensors and actuators."""

    def __init__(self):
        self.sensors: Dict[str, Sensor] = {}
        self.actuators: Dict[str, Actuator] = {}
        self._lock = threading.Lock()

    def add_sensor(self, sensor: Sensor):
        with self._lock:
            self.sensors[sensor.name] = sensor
        logger.info("Registered sensor %s [%s]", sensor.name, sensor.state.value)

    def add_actuator(self, actuator: Actuator):
        with self._lock:
            self.actuators[actuator.name] = actuator
        logger.info("Registered actuator %s [%s, dry_run=%s]",
                    actuator.name, actuator.state.value, actuator.dry_run)

    def active_sensors(self) -> List[Sensor]:
        return [s for s in self.sensors.values() if s.state == State.ACTIVE]

    def bind_to_vla(self, vla, action_class_enum):
        """Register every actuator with the brain's VLA channel so the
        Executive's jurisdiction and veto rules apply to hardware exactly as
        they do to software tools. Nothing reaches a motor without a board
        vote."""
        n = 0
        for a in self.actuators.values():
            if a.state == State.UNAVAILABLE:
                continue
            for intent in (a.intents or [a.name]):
                vla.register_actuator(intent, a, reversible=a.reversible)
                n += 1
        logger.info("Bound %d actuator intent(s) to the VLA channel", n)
        return n

    def bind_to_registry(self, tool_registry, ToolEntry):
        """Also expose actuators as ordinary tools so the classifier can
        route to them. They remain gated: the registry is only how the intent
        finds the function, not permission to run it."""
        n = 0
        for a in self.actuators.values():
            if a.state == State.UNAVAILABLE:
                continue
            intents = a.intents or [a.name]
            tool_registry.register(ToolEntry(
                name=a.name, intents=intents, func=a,
                description=a.capability.description,
                kind="actuator",
            ), overwrite=True)
            n += 1
        return n

    def close_all(self):
        for s in self.sensors.values():
            try:
                s.close()
            except Exception:
                logger.exception("Error closing sensor %s", s.name)

    def status(self) -> Dict[str, Any]:
        return {
            "sensors": [s.status() for s in self.sensors.values()],
            "actuators": [a.status() for a in self.actuators.values()],
        }

    def report(self) -> str:
        """Human-readable startup banner. Worth printing every boot: it is
        the difference between 'ASH is running' and 'ASH is running and can
        actually see three of the five things you think it can'."""
        lines = ["Periphery:"]
        for s in self.sensors.values():
            mark = {"active": "+", "disabled": "-", "unavailable": "x", "error": "!"}[s.state.value]
            extra = f"  ({s.unavailable_reason})" if s.state == State.UNAVAILABLE else ""
            lines.append(f"  [{mark}] sensor   {s.name:<18} every {s.interval:>5.1f}s{extra}")
        for a in self.actuators.values():
            mark = {"active": "+", "disabled": "-", "unavailable": "x", "error": "!"}[a.state.value]
            dr = " DRY-RUN" if a.dry_run else " LIVE"
            extra = f"  ({a.unavailable_reason})" if a.state == State.UNAVAILABLE else dr
            lines.append(f"  [{mark}] actuator {a.name:<18} {a.action_class:<12}{extra}")
        return "\n".join(lines)


PERIPHERY = PeripheryRegistry()
