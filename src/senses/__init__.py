"""
ASH periphery: sensors and actuators.

Sensors produce SensorEvents. Actuators change the world and inherit the
brain's safety model by declaring an `action_class`.

`build_periphery(config, ...)` constructs everything from a config dict and
returns the registry, wiring cross-references (screenshot needs the window
sensor for its blocklist, the input actuator needs it for the same reason).
"""

from .base import (
    PERIPHERY, PeripheryRegistry, Sensor, Actuator, SensorEvent,
    Capability, Modality, State,
)
from .screen import ActiveWindowSensor, ScreenshotSensor, IdleSensor
from .devices import CameraSensor, SystemSensor, SerialSensor
from .actuators import (
    NotifyActuator, SpeakActuator, LaunchActuator, InputActuator,
    ClipboardActuator, MotorActuator, ShellActuator,
)

__all__ = [
    "PERIPHERY", "PeripheryRegistry", "Sensor", "Actuator", "SensorEvent",
    "Capability", "Modality", "State",
    "ActiveWindowSensor", "ScreenshotSensor", "IdleSensor",
    "CameraSensor", "SystemSensor", "SerialSensor",
    "NotifyActuator", "SpeakActuator", "LaunchActuator", "InputActuator",
    "ClipboardActuator", "MotorActuator", "ShellActuator",
    "build_periphery",
]


def build_periphery(config: dict, tts_engine=None, registry=None) -> PeripheryRegistry:
    """Construct the full periphery from a daemon config dict.

    Order matters: the active-window sensor is built first because both the
    screenshot sensor and the input actuator need it to enforce the privacy
    blocklist, and a blocklist that isn't wired in is worse than none at all
    because it looks like protection.
    """
    reg = registry or PERIPHERY
    s = config.get("sensors", {})
    a = config.get("actuators", {})

    win_cfg = s.get("active_window", {})
    window = ActiveWindowSensor(
        interval=win_cfg.get("interval", 3.0),
        enabled=win_cfg.get("enabled", True),
        blocklist=(config.get("privacy", {}).get("blocked_apps") or None),
    )
    reg.add_sensor(window)

    idle_cfg = s.get("idle", {})
    reg.add_sensor(IdleSensor(interval=idle_cfg.get("interval", 10.0),
                              enabled=idle_cfg.get("enabled", True)))

    sys_cfg = s.get("system", {})
    reg.add_sensor(SystemSensor(interval=sys_cfg.get("interval", 30.0),
                                enabled=sys_cfg.get("enabled", True)))

    shot_cfg = s.get("screenshot", {})
    reg.add_sensor(ScreenshotSensor(
        interval=shot_cfg.get("interval", 20.0),
        enabled=shot_cfg.get("enabled", False),
        window_sensor=window,
        save_frames=shot_cfg.get("save_frames", False),
        max_frames=shot_cfg.get("max_frames", 500),
    ))

    mic_cfg = s.get("microphone", {})
    from .mic import MicrophoneSensor
    mic = MicrophoneSensor(
        enabled=mic_cfg.get("enabled", False),
        device=mic_cfg.get("device"),
        keep_audio=mic_cfg.get("keep_audio", False),
    )
    reg.add_sensor(mic)

    cam_cfg = s.get("camera", {})
    reg.add_sensor(CameraSensor(
        interval=cam_cfg.get("interval", 2.0),
        enabled=cam_cfg.get("enabled", False),
        index=cam_cfg.get("index", 0),
        motion_threshold=cam_cfg.get("motion_threshold", 0.045),
    ))

    ser_cfg = s.get("serial", {})
    if ser_cfg.get("enabled"):
        reg.add_sensor(SerialSensor(
            port=ser_cfg.get("port", "/dev/ttyUSB0"),
            baud=ser_cfg.get("baud", 115200),
            interval=ser_cfg.get("interval", 2.0),
            enabled=True,
            alert_on={k: tuple(v) for k, v in (ser_cfg.get("alert_on") or {}).items()},
        ))

    # -- actuators ------------------------------------------------------
    def _cfg(name):
        c = a.get(name, {})
        return {"enabled": c.get("enabled", False), "dry_run": c.get("dry_run", True)}, c

    base, _ = _cfg("notify")
    reg.add_actuator(NotifyActuator(**base))

    base, _ = _cfg("speak")
    reg.add_actuator(SpeakActuator(tts_engine=tts_engine, **base))

    base, _ = _cfg("clipboard")
    reg.add_actuator(ClipboardActuator(**base))

    base, c = _cfg("launch")
    reg.add_actuator(LaunchActuator(allowlist=c.get("allowlist", []),
                                    allow_urls=c.get("allow_urls", True), **base))

    base, _ = _cfg("input")
    reg.add_actuator(InputActuator(window_sensor=window, **base))

    base, c = _cfg("shell")
    reg.add_actuator(ShellActuator(allowlist=c.get("allowlist", []), **base))

    base, c = _cfg("motor")
    reg.add_actuator(MotorActuator(port=c.get("port", "/dev/ttyUSB0"),
                                   baud=c.get("baud", 115200),
                                   limits=c.get("limits", {}), **base))
    return reg
