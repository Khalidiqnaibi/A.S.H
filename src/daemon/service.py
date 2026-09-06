"""
src/daemon/service.py

Process supervision for a thing that is supposed to run for months.

Responsibilities that have nothing to do with cognition but everything to do
with whether the cognition is still running on day 40:

  * SINGLE INSTANCE. A PID-file lock with liveness checking, so a second
    launch refuses instead of running two microphones and two journals over
    the same files.
  * SIGNALS. SIGINT/SIGTERM shut down cleanly -- flush the journal, release
    the camera, close serial ports. A killed daemon that leaves a locked
    /dev/ttyUSB0 behind is a daemon you stop trusting.
  * CONTROL SOCKET. A loopback-only line protocol on 127.0.0.1 for `status`,
    `pause`, `mute`, `say`, `sleep`. This is how you talk to a headless
    always-on process without wiring a whole UI, and how `ashctl` works.
  * WATCHDOG. If the ambient loop stalls past its threshold, restart it in
    place rather than sitting silently dead.
  * CONFIG. One JSON file, hot-reloadable, with every peripheral off by
    default except the cheap non-invasive ones.

The control socket binds to 127.0.0.1 only and is not authenticated, which is
appropriate for a single-user desktop and *not* appropriate on a shared box --
set `control.enabled: false` there and use the CLI directly.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ash.daemon.service")

STATE_DIR = os.environ.get("ASH_STATE_DIR", os.path.join(os.getcwd(), "state"))
PID_FILE = os.path.join(STATE_DIR, "ashd.pid")
CONFIG_FILE = os.environ.get("ASH_DAEMON_CONFIG", os.path.join(os.getcwd(), "daemon.json"))

DEFAULT_CONFIG: Dict[str, Any] = {
    "sensors": {
        # Cheap and non-invasive: on.
        "active_window": {"enabled": True, "interval": 3.0},
        "idle": {"enabled": True, "interval": 10.0},
        "system": {"enabled": True, "interval": 30.0},
        # Invasive: off until the user says otherwise.
        "screenshot": {"enabled": False, "interval": 20.0, "save_frames": False},
        "microphone": {"enabled": False, "keep_audio": False},
        "camera": {"enabled": False, "interval": 2.0, "index": 0},
        "serial": {"enabled": False, "port": "/dev/ttyUSB0", "baud": 115200},
    },
    "actuators": {
        "notify": {"enabled": True, "dry_run": False},
        "speak": {"enabled": False, "dry_run": False},
        "clipboard": {"enabled": False, "dry_run": True},
        "launch": {"enabled": False, "dry_run": True, "allowlist": []},
        "input": {"enabled": False, "dry_run": True},
        "shell": {"enabled": False, "dry_run": True, "allowlist": []},
        "motor": {"enabled": False, "dry_run": True, "port": "/dev/ttyUSB0"},
    },
    "attention": {
        "base_threshold": 0.62,
        "refractory_seconds": 180,
        "max_unsolicited_per_hour": 6,
        "quiet_hours": [23, 8],
    },
    "privacy": {
        "redact": True,
        "raw_journal_days": 7,
        "frame_days": 2,
        "audio_days": 0,
        "require_local_asr": True,
        "blocked_apps": [],
    },
    "runtime": {
        "tick_seconds": 1.0,
        "sleep_after_idle_s": 1500,
    },
    "face": {
        "enabled": True,
        "driver": None,          # null | terminal | window | oled | tft; None = auto
        "width": 128, "height": 64, "fps": 30,
        "animations_dir": "animations",
        "source": "drives",      # drives | emotions
    },
    "control": {"enabled": True, "host": "127.0.0.1", "port": 8787},
    "log": {"level": "INFO", "file": "logs/ashd.log", "max_mb": 20, "backups": 5},
}


def load_config(path: str = CONFIG_FILE) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))   # deep copy
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                user = json.load(fh)
            _deep_merge(cfg, user)
            logger.info("Loaded daemon config from %s", path)
        except Exception:
            logger.exception("Bad config at %s -- using defaults", path)
    else:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            logger.info("Wrote default daemon config to %s", path)
        except Exception:
            logger.exception("Could not write default config")
    return cfg


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ----------------------------------------------------------------------
class SingleInstance:
    """PID-file lock that tolerates a stale file from a crashed run."""

    def __init__(self, path: str = PID_FILE):
        self.path = path
        self.acquired = False

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            try:
                with open(self.path) as fh:
                    pid = int(fh.read().strip())
                if _pid_alive(pid):
                    logger.error("ASH daemon already running (pid %d)", pid)
                    return False
                logger.warning("Removing stale pid file (pid %d is gone)", pid)
            except Exception:
                logger.warning("Unreadable pid file; replacing")
            try:
                os.remove(self.path)
            except Exception:
                pass
        with open(self.path, "w") as fh:
            fh.write(str(os.getpid()))
        self.acquired = True
        return True

    def release(self):
        if self.acquired and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except Exception:
                pass


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
class ControlServer:
    """Line protocol on loopback. One command per line, JSON reply."""

    def __init__(self, runtime, host: str = "127.0.0.1", port: int = 8787):
        self.runtime = runtime
        self.host, self.port = host, port
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.host, self.port))
            self._sock.listen(4)
            self._sock.settimeout(1.0)
        except Exception:
            logger.exception("Control server could not bind %s:%s", self.host, self.port)
            return
        self._thread = threading.Thread(target=self._serve, daemon=True, name="ash-control")
        self._thread.start()
        logger.info("Control server on %s:%s", self.host, self.port)

    def stop(self):
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket):
        try:
            conn.settimeout(10)
            data = conn.recv(65536).decode("utf-8", errors="replace").strip()
            reply = self.dispatch(data)
            conn.sendall((json.dumps(reply, default=str) + "\n").encode())
        except Exception as e:
            try:
                conn.sendall((json.dumps({"ok": False, "error": str(e)}) + "\n").encode())
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def dispatch(self, line: str) -> Dict[str, Any]:
        parts = line.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        rt = self.runtime

        if cmd == "status":
            return {"ok": True, "status": rt.status()}
        if cmd == "pause":
            rt.privacy.pause(arg or "control")
            return {"ok": True, "paused": True}
        if cmd == "resume":
            rt.privacy.resume()
            return {"ok": True, "paused": False}
        if cmd == "mute":
            rt.gate.mute(True)
            return {"ok": True, "muted": True}
        if cmd == "unmute":
            rt.gate.mute(False)
            return {"ok": True, "muted": False}
        if cmd == "say":
            resp = rt.ash.think(arg)
            return {"ok": True, "text": resp.text, "pathway": resp.pathway.value}
        if cmd == "why":
            return {"ok": True, "trace": rt.ash.explain_last()}
        if cmd == "sleep":
            return {"ok": True, "report": rt._sleep_phase()}
        if cmd == "rollup":
            return {"ok": True, "report": rt.journal.rollup_pending(
                episodic=getattr(rt.ash, "episodic_memory", None))}
        if cmd == "face":
            face = getattr(rt, "face", None)
            if face is None:
                return {"ok": False, "error": "face not enabled"}
            if arg:
                played = face.player.play(arg, force=True)
                return {"ok": played, "playing": face.player.status()["current"]}
            return {"ok": True, "face": face.status()}
        if cmd == "sensors":
            return {"ok": True, "report": rt.periphery.report()}
        if cmd in ("enable", "disable"):
            target = rt.periphery.sensors.get(arg) or rt.periphery.actuators.get(arg)
            if target is None:
                return {"ok": False, "error": f"no peripheral named {arg!r}"}
            target.enable() if cmd == "enable" else target.disable()
            return {"ok": True, "peripheral": target.status()}
        if cmd == "shutdown":
            threading.Thread(target=rt.stop, daemon=True).start()
            return {"ok": True, "stopping": True}
        return {"ok": False, "error": f"unknown command {cmd!r}",
                "commands": ["status", "pause", "resume", "mute", "unmute", "say",
                             "why", "sleep", "rollup", "sensors", "face",
                             "enable", "disable", "shutdown"]}


def control_client(cmd: str, host: str = "127.0.0.1", port: int = 8787,
                   timeout: float = 30.0) -> Dict[str, Any]:
    """Used by `ashctl`."""
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall((cmd + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8", errors="replace") or "{}")


# ----------------------------------------------------------------------
def setup_logging(cfg: Dict[str, Any]):
    import logging.handlers

    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-26s %(message)s")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    path = cfg.get("file")
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            path, maxBytes=int(cfg.get("max_mb", 20)) * 1_000_000,
            backupCount=int(cfg.get("backups", 5)), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # These are chatty and never interesting at INFO in a 24h log.
    for noisy in ("urllib3", "PIL", "matplotlib", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Watchdog:
    """Restarts the ambient loop if it stalls. Does not restart the process --
    that is the OS supervisor's job (systemd, Task Scheduler, launchd)."""

    def __init__(self, runtime, interval: float = 30.0):
        self.runtime = runtime
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.restarts = 0

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ash-watchdog")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                if not self.runtime.healthy():
                    self.restarts += 1
                    logger.error("Watchdog: ambient loop unhealthy -- restarting (#%d)",
                                 self.restarts)
                    self.runtime.stop(timeout=3)
                    self.runtime.start()
            except Exception:
                logger.exception("Watchdog itself failed")


def install_signal_handlers(on_shutdown: Callable[[], None]):
    def _handler(signum, _frame):
        logger.warning("Received signal %s -- shutting down", signum)
        on_shutdown()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass
    # Windows console close
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, _handler)
        except Exception:
            pass
