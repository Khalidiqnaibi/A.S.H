"""
ASH periphery: sensors and actuators.

Sensors produce SensorEvents. Actuators change the world and inherit the
brain's safety model by declaring an `action_class`.

`build_periphery(config, ...)` constructs everything from a config dict and
returns the registry, wiring cross-references (screenshot needs the window
sensor for its blocklist, the input actuator needs it for the same reason).
"""

import time
import os

from .base import (
    PERIPHERY, PeripheryRegistry, Sensor, Actuator, SensorEvent,
    Capability, Modality, State,
)
from .screen import ActiveWindowSensor, ScreenshotSensor, IdleSensor
from .nowplaying import NowPlayingSensor
from .devices import CameraSensor, SystemSensor, SerialSensor
from .actuators import (
    NotifyActuator, SpeakActuator, LaunchActuator, InputActuator,
    ClipboardActuator, MotorActuator, ShellActuator,
)

__all__ = [
    "PERIPHERY", "PeripheryRegistry", "Sensor", "Actuator", "SensorEvent",
    "Capability", "Modality", "State",
    "ActiveWindowSensor", "ScreenshotSensor", "IdleSensor", "NowPlayingSensor",
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

    priv = config.get("privacy", {})
    from .screen import set_private_content_protection
    set_private_content_protection(priv.get("protect_private_content", False))

    win_cfg = s.get("active_window", {})
    window = ActiveWindowSensor(
        interval=win_cfg.get("interval", 3.0),
        enabled=win_cfg.get("enabled", True),
        blocklist=(priv.get("blocked_apps") or None),
        use_default_blocklist=priv.get("use_default_blocklist", True),
    )
    reg.add_sensor(window)

    np_cfg = s.get("now_playing", {})
    reg.add_sensor(NowPlayingSensor(
        interval=np_cfg.get("interval", 5.0),
        enabled=np_cfg.get("enabled", True),
        window_sensor=window,
        skip_below_s=np_cfg.get("skip_below_s", 30.0),
    ))

    idle_cfg = s.get("idle", {})
    idle = IdleSensor(interval=idle_cfg.get("interval", 10.0), enabled=idle_cfg.get("enabled", True))
    reg.add_sensor(idle)

    # Browser tabs, from the ash-tabs extension. Off until switched on: it is a record of what you browse.
    bt = s.get("browser_tabs", {})
    if bt.get("enabled", False):
        from src.daemon.ledger import Ledger
        from src.daemon.pagekind import Classifier
        from .browsertabs import BrowserTabSensor, load_or_create_token, write_extension_config
        state = os.environ.get("ASH_STATE_DIR", os.path.join(os.getcwd(), "state"))
        token = load_or_create_token(os.path.join(state, "browser_token.txt"))
        tabs = BrowserTabSensor(
            interval=bt.get("interval", 2.0), enabled=True,
            classifier=Classifier(os.path.join(state, "pagekind.json")),
            ledger=Ledger(os.path.join(state, "ledger.sqlite3"), retention_days=bt.get("retention_days", 30),
                          keep_titles=bt.get("keep_titles", False)),
            port=int(bt.get("port", 47831)), token=token, listen=True,
            min_s=bt.get("min_s", 3.0), grace_s=bt.get("grace_s", 15.0),
            idle_s=(idle.seconds_idle if idle.state == State.ACTIVE else None),
        )
        reg.add_sensor(tabs)
        ext = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "extension", "ash-tabs")
        write_extension_config(ext, tabs.port, token)

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
        read_text=shot_cfg.get("read_text", False),
        describe=shot_cfg.get("describe", True),
        read_text_every_s=shot_cfg.get("read_text_every_s", 120.0),
        ocr_threads=shot_cfg.get("ocr_threads", 2),
    ))

    mic_cfg = s.get("microphone", {})
    from .mic import MicrophoneSensor
    from .livevoice import BargeIn, WakeGate, openwakeword_detector
    wake = None
    if mic_cfg.get("wake_word_model"):
        det = openwakeword_detector(str(mic_cfg["wake_word_model"]))
        if det is not None:
            from . import address as _address
            wake = WakeGate(det, threshold=float(mic_cfg.get("wake_threshold", 0.5)),
                            conversation_open=lambda: _address.conversation.openness(time.time()) > 0.02)
    mic = MicrophoneSensor(
        enabled=mic_cfg.get("enabled", False),
        device=mic_cfg.get("device"),
        keep_audio=mic_cfg.get("keep_audio", False),
        speculative=bool(mic_cfg.get("speculative", True)),
        barge=(BargeIn() if mic_cfg.get("barge_in", True) else None),
        wake=wake,
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
    speaker = SpeakActuator(tts_engine=tts_engine, **base)
    reg.add_actuator(speaker)
    # Talking over ASH stops it (livevoice.BargeIn), heard by the microphone while the speaker is busy.
    mic.speaking_fn = lambda: bool(speaker.speaking)
    mic.on_barge_in = speaker.stop

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
