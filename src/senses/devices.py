"""
src/senses/devices.py

Camera, machine telemetry, and a generic serial sensor for everything else.

The camera follows the same rule as the screenshot sensor: motion-gated, not
interval-gated. A webcam pointed at an empty chair for eight hours should
produce zero events, not 2,880 identical frames. Motion is measured as the
mean absolute difference against a slowly-updated background estimate, which
is cheap enough to run every second on a CPU and robust to gradual lighting
change.

`SerialSensor` is the extension point for physical sensors -- temperature,
distance, IMU, whatever is on the other end of a USB or UART link. It expects
newline-delimited JSON or `key=value` pairs, which covers essentially every
Arduino/ESP32 sketch anyone writes, and hands the parsed dict straight through
as telemetry.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import Capability, Modality, Sensor, SensorEvent

logger = logging.getLogger("ash.senses.devices")


class CameraSensor(Sensor):
    """Motion-gated webcam capture."""

    capability = Capability(
        name="camera",
        description="Webcam frames, emitted only when the scene changes.",
        requires=["cv2"],
        privacy_sensitive=True,
    )
    default_interval = 2.0
    modality = Modality.VISION

    def __init__(self, interval: Optional[float] = None, enabled: bool = False,
                 index: int = 0, motion_threshold: float = 0.045,
                 warmup_frames: int = 5):
        self.index = index
        self.motion_threshold = motion_threshold
        self.warmup_frames = warmup_frames
        self._cap = None
        self._bg = None
        self._frames_seen = 0
        self._motion_events = 0
        super().__init__(interval=interval, enabled=enabled)

    def probe(self) -> Tuple[bool, str]:
        try:
            import cv2
        except Exception as e:
            return False, f"opencv missing ({e})"
        cap = cv2.VideoCapture(self.index)
        ok = cap.isOpened()
        cap.release()
        return (True, "") if ok else (False, f"no camera at index {self.index}")

    def _open(self):
        import cv2
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self._cap

    def read(self) -> List[SensorEvent]:
        import cv2
        import numpy as np

        cap = self._open()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")

        self._frames_seen += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._bg is None:
            self._bg = gray
            return []

        diff = float(np.mean(np.abs(gray - self._bg)))
        # Slow background update: adapts to daylight, not to a person sitting
        # still for two minutes.
        self._bg = 0.92 * self._bg + 0.08 * gray

        if self._frames_seen <= self.warmup_frames or diff < self.motion_threshold:
            return []

        self._motion_events += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            from PIL import Image
            img = Image.fromarray(rgb)
        except Exception:
            img = rgb

        return [SensorEvent(
            source=self.name, modality=Modality.VISION,
            text=f"motion in camera view (delta {diff:.3f})",
            data=img,
            salience=float(min(1.0, 0.35 + diff * 6.0)),
            urgency=0.0,
            meta={"motion": round(diff, 4)},
        )]

    def close(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def status(self) -> Dict[str, Any]:
        d = super().status()
        d.update({"frames": self._frames_seen, "motion_events": self._motion_events})
        return d


class SystemSensor(Sensor):
    """Machine state: CPU, memory, disk, battery, network.

    Emits only on *threshold crossings*, not every poll. The point is not to
    log a time series -- that's what monitoring tools are for -- but to notice
    the handful of moments the user would want mentioned: battery about to
    die, disk nearly full, something pinning the CPU.
    """

    capability = Capability(
        name="system", description="CPU, RAM, disk and battery threshold events.",
        requires=["psutil"],
    )
    default_interval = 30.0
    modality = Modality.TELEMETRY

    def __init__(self, interval: Optional[float] = None, enabled: bool = True,
                 cpu_high: float = 90.0, mem_high: float = 90.0,
                 disk_high: float = 92.0, battery_low: float = 20.0,
                 battery_critical: float = 8.0):
        self.thresholds = {
            "cpu": cpu_high, "mem": mem_high, "disk": disk_high,
            "battery_low": battery_low, "battery_critical": battery_critical,
        }
        self._fired: Dict[str, bool] = {}
        super().__init__(interval=interval, enabled=enabled)

    def _edge(self, key: str, condition: bool) -> bool:
        """Fire once on the rising edge; re-arm when it clears. Prevents a
        low battery from generating an event every 30 seconds for an hour."""
        was = self._fired.get(key, False)
        self._fired[key] = condition
        return condition and not was

    def read(self) -> List[SensorEvent]:
        import psutil

        out: List[SensorEvent] = []
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage(os.path.abspath(os.sep)).percent

        if self._edge("cpu", cpu >= self.thresholds["cpu"]):
            top = self._top_process()
            out.append(SensorEvent(
                source=self.name, text=f"CPU at {cpu:.0f}%" + (f", mostly {top}" if top else ""),
                salience=0.5, urgency=0.25, meta={"cpu": cpu, "top": top}))

        if self._edge("mem", mem >= self.thresholds["mem"]):
            out.append(SensorEvent(
                source=self.name, text=f"memory at {mem:.0f}%",
                salience=0.55, urgency=0.35, meta={"mem": mem}))

        if self._edge("disk", disk >= self.thresholds["disk"]):
            out.append(SensorEvent(
                source=self.name, text=f"disk {disk:.0f}% full",
                salience=0.6, urgency=0.5, meta={"disk": disk}))

        bat = getattr(psutil, "sensors_battery", lambda: None)()
        if bat is not None:
            pct, plugged = bat.percent, bat.power_plugged
            if self._edge("bat_crit", pct <= self.thresholds["battery_critical"] and not plugged):
                out.append(SensorEvent(
                    source=self.name, text=f"battery critical at {pct:.0f}%, not charging",
                    salience=0.9, urgency=0.95, meta={"battery": pct}))
            elif self._edge("bat_low", pct <= self.thresholds["battery_low"] and not plugged):
                out.append(SensorEvent(
                    source=self.name, text=f"battery at {pct:.0f}% and unplugged",
                    salience=0.6, urgency=0.55, meta={"battery": pct}))
            # Charging state is what the maintenance phase actually wants.
            if self._edge("plugged", bool(plugged)):
                out.append(SensorEvent(
                    source=self.name, text="power connected", salience=0.2,
                    meta={"plugged": True}))
        return out

    @staticmethod
    def _top_process() -> str:
        try:
            import psutil
            procs = sorted(psutil.process_iter(["name", "cpu_percent"]),
                           key=lambda p: p.info.get("cpu_percent") or 0, reverse=True)
            return procs[0].info.get("name", "") if procs else ""
        except Exception:
            return ""

    def snapshot(self) -> Dict[str, Any]:
        """On-demand full reading, for when ASH is actually asked."""
        try:
            import psutil
            bat = getattr(psutil, "sensors_battery", lambda: None)()
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage(os.path.abspath(os.sep)).percent,
                "battery_percent": bat.percent if bat else None,
                "power_plugged": bat.power_plugged if bat else None,
                "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
            }
        except Exception as e:
            return {"error": str(e)}


class SerialSensor(Sensor):
    """Generic line-oriented serial sensor.

    Accepts newline-delimited JSON (`{"temp":21.5,"rh":40}`) or key=value
    pairs (`temp=21.5 rh=40`). Any field listed in `alert_on` with a value
    outside its (min, max) band raises the event's urgency, which is how a
    physical sensor earns the right to interrupt.
    """

    capability = Capability(
        name="serial", description="Line-based sensor data over a serial port.",
        requires=["serial"],
    )
    default_interval = 2.0
    modality = Modality.TELEMETRY

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200,
                 interval: Optional[float] = None, enabled: bool = False,
                 name_suffix: str = "", alert_on: Optional[Dict[str, Tuple[float, float]]] = None):
        self.port, self.baud = port, baud
        self.alert_on = alert_on or {}
        self._ser = None
        self._last: Dict[str, Any] = {}
        if name_suffix:
            self.capability = Capability(
                name=f"serial_{name_suffix}",
                description=self.capability.description,
                requires=["serial"],
            )
        super().__init__(interval=interval, enabled=enabled)

    def probe(self) -> Tuple[bool, str]:
        try:
            import serial  # noqa: F401
        except Exception as e:
            return False, f"pyserial missing ({e})"
        if not os.path.exists(self.port) and not self.port.upper().startswith("COM"):
            return False, f"port {self.port} not present"
        return True, ""

    def _open(self):
        import serial
        if self._ser is None or not self._ser.is_open:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        return self._ser

    @staticmethod
    def _parse(line: str) -> Optional[Dict[str, Any]]:
        line = line.strip()
        if not line:
            return None
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                return None
        pairs = re.findall(r"(\w+)\s*[=:]\s*(-?[\d.]+)", line)
        return {k: float(v) for k, v in pairs} or None

    def read(self) -> List[SensorEvent]:
        ser = self._open()
        latest = None
        # Drain the buffer; only the newest complete reading matters.
        while ser.in_waiting:
            raw = ser.readline().decode("utf-8", errors="replace")
            parsed = self._parse(raw)
            if parsed:
                latest = parsed
        if not latest:
            return []

        alerts = []
        urgency = 0.0
        for field, (lo, hi) in self.alert_on.items():
            v = latest.get(field)
            if v is None:
                continue
            if v < lo or v > hi:
                alerts.append(f"{field}={v} outside [{lo}, {hi}]")
                urgency = max(urgency, 0.7)

        # Suppress unchanged readings; a sensor reporting 21.5C forever is
        # not news.
        if latest == self._last and not alerts:
            return []
        self._last = latest

        text = ", ".join(f"{k}={v}" for k, v in latest.items())
        if alerts:
            text = "ALERT: " + "; ".join(alerts) + " | " + text

        return [SensorEvent(
            source=self.name, modality=Modality.TELEMETRY, text=text,
            data=latest, salience=0.7 if alerts else 0.25, urgency=urgency,
            meta={k: v for k, v in latest.items()},
        )]

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
