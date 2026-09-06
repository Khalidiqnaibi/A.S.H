"""
src/senses/actuators.py

Everything that changes the world.

Each actuator declares an `action_class` that maps onto the brain's
`ActionClass` enum, and that is what determines who gets to veto it:

    notify / speak      -> mutation     (VLA has standing, low stakes)
    clipboard / focus   -> mutation
    launch / keystroke  -> irreversible (typing cannot be un-typed)
    motor / gpio        -> physical     (VLA veto + grounding requirement)

`dry_run=True` is the default on every one of them. An always-on process with
synthetic keyboard control is exactly the kind of thing that should have to be
switched on deliberately, once you've watched the logs and seen what it *would*
have done.

The keystroke and click actuators additionally refuse to operate on a window
whose title is on the blocklist -- the same list the screenshot sensor uses.
Automating a click inside a password manager is not a thing that should be
one classifier misfire away.
"""

from __future__ import annotations

import logging
import os
import platform
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional

from .base import Actuator, Capability, State

logger = logging.getLogger("ash.senses.actuators")


class NotifyActuator(Actuator):
    """Desktop notification. The safest way for an always-on ASH to say
    something without hijacking focus or talking over a meeting."""

    capability = Capability(
        name="notify", description="Show a desktop notification.",
    )
    action_class = "mutation"
    reversible = True
    intents = ["notify", "notification", "remind_me", "alert_user"]

    def probe_backend(self) -> Optional[str]:
        s = platform.system()
        if s == "Windows":
            return "win"
        if s == "Darwin":
            return "osascript"
        from shutil import which
        return "notify-send" if which("notify-send") else None

    def act(self, args: Any) -> Dict[str, Any]:
        text = args if isinstance(args, str) else str((args or {}).get("text", args))
        title = "A.S.H"
        backend = self.probe_backend()

        if backend == "win":
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, text[:500], title, 0x40 | 0x1000)
                return {"ok": True, "result": "notification shown"}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        if backend == "osascript":
            subprocess.run(["osascript", "-e",
                            f'display notification {shlex.quote(text[:400])} with title "{title}"'],
                           timeout=5)
            return {"ok": True, "result": "notification shown"}
        if backend == "notify-send":
            subprocess.run(["notify-send", title, text[:400]], timeout=5)
            return {"ok": True, "result": "notification shown"}
        return {"ok": False, "error": "no notification backend"}


class SpeakActuator(Actuator):
    """Text to speech via the TTS engine ASH already ships with."""

    capability = Capability(name="speak", description="Say something out loud.")
    action_class = "mutation"
    reversible = False      # you cannot un-say a thing in a quiet room
    intents = ["speak", "say", "talk"]

    def __init__(self, tts_engine=None, **kw):
        self.tts = tts_engine
        super().__init__(**kw)

    def act(self, args: Any) -> Dict[str, Any]:
        text = args if isinstance(args, str) else str(args)
        if self.tts is None:
            return {"ok": False, "error": "no TTS engine attached"}
        try:
            chunks = 0
            for _ in self.tts.stream_audio(text):
                chunks += 1
            return {"ok": True, "result": f"spoke {len(text)} chars in {chunks} chunks"}
        except Exception as e:
            return {"ok": False, "error": str(e)}


class LaunchActuator(Actuator):
    """Start an application or open a file/URL.

    Allowlisted. An open-ended "run whatever string the classifier produced"
    is a remote code execution primitive with extra steps, so the allowlist is
    not optional and is empty by default.
    """

    capability = Capability(name="launch", description="Open an allowlisted app, file, or URL.")
    action_class = "irreversible"
    reversible = False
    intents = ["launch_app", "open_app", "open_url"]

    def __init__(self, allowlist: Optional[List[str]] = None, allow_urls: bool = True, **kw):
        self.allowlist = [a.lower() for a in (allowlist or [])]
        self.allow_urls = allow_urls
        super().__init__(**kw)

    def act(self, args: Any) -> Dict[str, Any]:
        target = (args if isinstance(args, str) else str((args or {}).get("target", ""))).strip()
        if not target:
            return {"ok": False, "error": "no target"}

        is_url = target.startswith(("http://", "https://"))
        if is_url and not self.allow_urls:
            return {"ok": False, "error": "URL launching disabled"}
        if not is_url and not any(a in target.lower() for a in self.allowlist):
            return {"ok": False,
                    "error": f"'{target}' is not on the launch allowlist ({len(self.allowlist)} entries)"}

        s = platform.system()
        try:
            if s == "Windows":
                os.startfile(target)  # type: ignore[attr-defined]
            elif s == "Darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
            return {"ok": True, "result": f"launched {target}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}


class InputActuator(Actuator):
    """Synthetic keyboard and mouse.

    The most dangerous thing in this file. Three guards:
      * dry_run default
      * window-title blocklist checked immediately before every action
      * a rate limit, because a stuck loop typing into a terminal is a very
        bad afternoon
    """

    capability = Capability(
        name="input", description="Type text, press keys, click.",
        requires=["pyautogui"],
    )
    action_class = "irreversible"
    reversible = False
    intents = ["type_text", "press_key", "click_mouse"]

    def __init__(self, window_sensor=None, max_per_minute: int = 30, **kw):
        self.window_sensor = window_sensor
        self.max_per_minute = max_per_minute
        self._recent: List[float] = []
        super().__init__(**kw)

    def _rate_ok(self) -> bool:
        now = time.time()
        self._recent = [t for t in self._recent if now - t < 60]
        if len(self._recent) >= self.max_per_minute:
            return False
        self._recent.append(now)
        return True

    def act(self, args: Any) -> Dict[str, Any]:
        import pyautogui

        if not self._rate_ok():
            return {"ok": False, "error": f"rate limit: >{self.max_per_minute} inputs/min"}

        if self.window_sensor is not None:
            title, _ = self.window_sensor.current()
            if self.window_sensor.is_blocked(title):
                return {"ok": False, "error": "target window is on the privacy blocklist"}

        spec = args if isinstance(args, dict) else {"type": str(args)}
        pyautogui.FAILSAFE = True

        if "type" in spec:
            pyautogui.typewrite(str(spec["type"])[:2000], interval=0.01)
            return {"ok": True, "result": f"typed {len(str(spec['type']))} chars"}
        if "key" in spec:
            keys = spec["key"] if isinstance(spec["key"], list) else [spec["key"]]
            pyautogui.hotkey(*keys)
            return {"ok": True, "result": f"pressed {'+'.join(keys)}"}
        if "click" in spec:
            x, y = spec["click"]
            pyautogui.click(x, y)
            return {"ok": True, "result": f"clicked ({x}, {y})"}
        return {"ok": False, "error": "expected one of: type, key, click"}


class ClipboardActuator(Actuator):
    """Read/write the clipboard. Reversible enough to be low-stakes, but it
    is still a mutation the user can observe, so it goes to the board."""

    capability = Capability(name="clipboard", description="Read or set the clipboard.")
    action_class = "mutation"
    reversible = True
    intents = ["clipboard", "copy_text", "paste_text"]

    def act(self, args: Any) -> Dict[str, Any]:
        try:
            import pyperclip
        except Exception:
            return {"ok": False, "error": "pyperclip not installed"}
        if args in (None, "", "read"):
            return {"ok": True, "result": pyperclip.paste()[:4000]}
        pyperclip.copy(str(args))
        return {"ok": True, "result": f"copied {len(str(args))} chars"}


class MotorActuator(Actuator):
    """Serial/GPIO motor control.

    `action_class = "physical"` is what puts this under the VLA's grounding
    requirement: without a fresh vision percept, the VLA vetoes and the motor
    does not move. That rule lives in the brain, not here, which is the point
    -- a new actuator inherits the safety model by declaring its class.

    Commands are sent as newline-terminated strings, which matches the
    firmware convention `SerialSensor` reads.
    """

    capability = Capability(name="motor", description="Drive a motor over serial.",
                            requires=["serial"])
    action_class = "physical"
    reversible = False
    intents = ["move_motor", "actuate", "drive_motor"]

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200,
                 limits: Optional[Dict[str, Any]] = None, **kw):
        self.port, self.baud = port, baud
        self.limits = limits or {}
        self._ser = None
        super().__init__(**kw)

    def _open(self):
        import serial
        if self._ser is None or not self._ser.is_open:
            self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
            time.sleep(1.8)   # most boards reset on port open
        return self._ser

    def act(self, args: Any) -> Dict[str, Any]:
        spec = args if isinstance(args, dict) else {"cmd": str(args)}

        # Clamp before sending. Firmware should validate too, but a bounds
        # check on this side means a hallucinated angle never reaches a servo.
        for k, (lo, hi) in self.limits.items():
            if k in spec:
                spec[k] = max(lo, min(hi, float(spec[k])))

        cmd = spec.get("cmd") or " ".join(f"{k}={v}" for k, v in spec.items())
        ser = self._open()
        ser.write((cmd + "\n").encode())
        ser.flush()
        reply = ser.readline().decode("utf-8", errors="replace").strip()
        return {"ok": True, "result": f"sent {cmd!r}", "reply": reply}

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass


class ShellActuator(Actuator):
    """Run an allowlisted shell command.

    Present because "PC actions" realistically means this eventually, and it
    is far better to have one audited, allowlisted, non-shell-interpolating
    implementation than five ad-hoc `os.system` calls scattered around. The
    allowlist matches the *first token only* and there is no shell=True.
    """

    capability = Capability(name="shell", description="Run an allowlisted command.")
    action_class = "irreversible"
    reversible = False
    intents = ["run_command", "shell_command"]

    def __init__(self, allowlist: Optional[List[str]] = None, timeout: float = 20.0, **kw):
        self.allowlist = set(a.lower() for a in (allowlist or []))
        self.timeout = timeout
        super().__init__(**kw)

    def act(self, args: Any) -> Dict[str, Any]:
        cmd = args if isinstance(args, str) else str((args or {}).get("cmd", ""))
        parts = shlex.split(cmd)
        if not parts:
            return {"ok": False, "error": "empty command"}
        if parts[0].lower() not in self.allowlist:
            return {"ok": False,
                    "error": f"'{parts[0]}' not in shell allowlist {sorted(self.allowlist)}"}
        try:
            out = subprocess.run(parts, capture_output=True, text=True, timeout=self.timeout)
            return {"ok": out.returncode == 0, "result": out.stdout[-4000:],
                    "stderr": out.stderr[-1000:], "code": out.returncode}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {self.timeout}s"}
