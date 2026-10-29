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
    `pause`, `mute`, `say`, `sleep`, `simulate`. This is how you talk to a
    headless always-on process without wiring a whole UI, and how `ashctl`
    works.
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

# How long `ashctl` waits for a reply. Most commands answer out of in-memory
# state and are instant, so a short ceiling on those is a useful "the daemon
# is wedged" signal rather than a hang.
#
# `say` is a different animal: it runs a full cognitive cycle, and on a
# CPU-bound local model that is minutes, not seconds. Against mistral here
# the first call took 122s -- ashctl printed "Control error: timed out" at
# 30s for an answer the daemon then went on to produce perfectly well, which
# makes the daemon look broken when it is only thinking. The slow commands
# now wait as long as the brain is itself allowed to take, plus a margin for
# the rest of the cycle around the model call.
#
# A daemon that is not listening still fails instantly either way: that is
# connection refused, not a timeout.
CTL_TIMEOUT = float(os.environ.get("ASH_CTL_TIMEOUT", "30"))
CTL_SLOW_TIMEOUT = float(os.environ.get("ASH_LLM_TIMEOUT", "300")) + 30.0
# Commands that wait on the model rather than on memory.
SLOW_COMMANDS = ("say", "sleep", "rollup")


def ctl_timeout(cmd: str) -> float:
    """How long to wait for this command's reply."""
    verb = (cmd or "").strip().split(" ", 1)[0].lower()
    return CTL_SLOW_TIMEOUT if verb in SLOW_COMMANDS else CTL_TIMEOUT

DEFAULT_CONFIG: Dict[str, Any] = {
    "sensors": {
        # Cheap and non-invasive: on.
        "active_window": {"enabled": True, "interval": 3.0},
        "idle": {"enabled": True, "interval": 10.0},
        "system": {"enabled": True, "interval": 30.0},
        # Invasive: off until the user says otherwise.
        # `describe` is free and tells text from video from an empty screen.
        # `read_text` is OCR: seconds of CPU per frame and the contents of
        # the screen turned into journaled text, so it is off with capture.
        "screenshot": {"enabled": False, "interval": 20.0, "save_frames": False,
                       "describe": True, "read_text": False,
                       "read_text_every_s": 120.0},
        "microphone": {"enabled": False, "keep_audio": False},
        "camera": {"enabled": False, "interval": 2.0, "index": 0},
        "serial": {"enabled": False, "port": "/dev/ttyUSB0", "baud": 115200},
    },
    "actuators": {
        "notify": {"enabled": True, "dry_run": False},
        "speak": {"enabled": True, "dry_run": False},
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
        # Your own entries, added to the built-in ones below.
        "blocked_apps": [],
        # The general protections, on because ASH is meant to be run by people
        # other than its author. Everything here is local, so someone who does
        # not mind ASH reading their own screen can turn either off:
        #   use_default_blocklist     passwords, banking, calls... : no capture,
        #                             and the title is redacted everywhere
        #   protect_private_content   chat and mail apps: no capture, no OCR,
        #                             and no file/subject taken from the title
        "use_default_blocklist": True,
        "protect_private_content": True,
    },
    "runtime": {
        "tick_seconds": 1.0,
        "sleep_after_idle_s": 1500,
        # Say the unprompted things out loud, not just on screen. This was an
        # undocumented key defaulting to off, which meant the free-moment
        # question and the offer of help -- the two features whose whole point
        # is speaking up unasked -- arrived in silence, and nothing in the
        # config file hinted that there was a switch. Set false for the quiet
        # channel; the attention gate's budgets still ration what gets here.
        "speak_unprompted": True,
        # Write the free-moment check-in with the language model instead of
        # composing it from the facts. Off: it froze the whole ambient loop
        # for a minute at a time and produced lines like "Just finished a bit
        # of work in Opera, if you needed me."
        "check_in_with_model": False,
        # Minutes of continuous activity (in seconds) before a pause after it is
        # worth a remark. 120 made "about 2 minutes of work" a four-times-an-hour
        # thing to say aloud.
        "flow_min_burst_s": 600.0,
    },
    "face": {
        "enabled": True,
        "driver": None,          # null | terminal | window | oled | tft; None = auto
        "width": 128, "height": 64, "fps": 30,
        "animations_dir": "animations",
        "source": "drives",      # drives | emotions
    },
    "control": {"enabled": True, "host": "127.0.0.1", "port": 8787},
    "ui": {
        "enabled": True,
        "port": 8788,            # loopback only
        "theme": "nebula",       # nebula | blossom | paper | forest
        "open_on_start": True,   # open the console window when the daemon starts
        "open_on_message": True, # reopen it when ASH has something to say and it is closed
    },
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
            from src.senses import address
            rt.privacy.pause(arg or "control")
            # Whatever was being said, it is over. Without this the follow-up
            # window outlives the pause and the first thing heard on resume
            # is read as a reply to a conversation from before.
            address.conversation.close()
            return {"ok": True, "paused": True}
        if cmd == "resume":
            rt.privacy.resume()
            return {"ok": True, "paused": False}
        if cmd == "mute":
            from src.senses import address
            rt.gate.mute(True)
            address.conversation.close()
            return {"ok": True, "muted": True}
        if cmd == "unmute":
            rt.gate.mute(False)
            return {"ok": True, "muted": False}
        if cmd == "say":
            resp = rt.ash.think(arg)
            return {"ok": True, "text": resp.text, "pathway": resp.pathway.value}
        if cmd == "simulate":
            return self._simulate(arg)
        if cmd == "why":
            return {"ok": True, "trace": rt.ash.explain_last()}
        if cmd == "history":
            try:
                n = int(arg) if arg.strip() else 50
            except ValueError:
                n = 50
            return {"ok": True, "history": rt.ash.explain_history(n)}
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
                             "why", "history", "sleep", "rollup", "sensors", "face",
                             "enable", "disable", "simulate", "shutdown"]}

    # ------------------------------------------------------------------
    def _simulate(self, arg: str) -> Dict[str, Any]:
        """Inject a synthetic SensorEvent as if a real sensor produced it,
        through the exact same AttentionGate/journal/response path real
        events take -- rt.submit() is the one entry point every real sensor
        also uses. This exercises routing (addressed detection, salience,
        urgency, observe-vs-respond) without needing real hardware or a
        quiet room. Presets below mirror each real sensor's own numbers
        (see src/senses/mic.py, devices.py) so a simulated event scores the
        same way a real one from that sensor would.

        Syntax: simulate <type> <text...>
          mic <text>             -- scored by the same src/senses/address.py
                                     model, and built by the same
                                     mic.build_event(), real speech is
          active_window <text>   -- ambient text observation
          idle <seconds>         -- idle-duration telemetry
          battery_low / battery_critical / cpu_high -- canned system alerts,
                                     for testing urgency-driven escalation
                                     without draining a real battery
          <anything else> <text> -- generic ambient default

        Result is submitted async (queued, processed on the next tick, same
        as a real sensor) -- follow up with `why` or `status` a moment
        later to see the attention decision and (if it escalated) the
        response.
        """
        from src.senses.base import Modality, SensorEvent

        parts = arg.split(" ", 1)
        kind = (parts[0] if parts else "").strip().lower()
        text = parts[1] if len(parts) > 1 else ""
        if not kind:
            return {"ok": False, "error": "usage: simulate <type> <text...>"}

        if kind == "mic":
            from src.senses import address
            from src.senses.mic import build_event
            # Built by the same function the real sensor uses, so the
            # simulated sentence carries exactly the salience, urgency and
            # address verdict a spoken one would.
            ev = build_event("sim:mic", text, address.score(text), simulated=True)
        elif kind == "active_window":
            ev = SensorEvent(
                source="sim:active_window", modality=Modality.TEXT, text=text,
                salience=0.35, urgency=0.0, meta={"simulated": True},
            )
        elif kind == "idle":
            try:
                seconds = float(text or "0")
            except ValueError:
                seconds = 0.0
            ev = SensorEvent(
                source="sim:idle", modality=Modality.TELEMETRY,
                text=f"idle for {seconds:.0f}s",
                salience=0.2, urgency=0.0,
                meta={"simulated": True, "seconds": seconds},
            )
        elif kind in ("battery_low", "battery_critical", "cpu_high"):
            presets = {
                # (salience, urgency, text) -- matches devices.py's own
                # SystemSensor numbers for the same conditions.
                "battery_low": (0.6, 0.55, text or "battery low"),
                "battery_critical": (0.9, 0.95, text or "battery critical"),
                "cpu_high": (0.5, 0.25, text or "high cpu usage"),
            }
            sal, urg, msg = presets[kind]
            ev = SensorEvent(
                source="sim:system", modality=Modality.TELEMETRY, text=msg,
                salience=sal, urgency=urg, meta={"simulated": True, "kind": kind},
            )
        else:
            ev = SensorEvent(
                source=f"sim:{kind}", modality=Modality.TEXT, text=text,
                salience=0.3, urgency=0.0, meta={"simulated": True},
            )

        self.runtime.submit(ev)
        return {
            "ok": True,
            "submitted": {
                "source": ev.source, "text": ev.text, "addressed": ev.addressed,
                "salience": ev.salience, "urgency": ev.urgency,
            },
            "note": "queued -- run `why` or `status` in a moment to see the outcome",
        }


def control_client(cmd: str, host: str = "127.0.0.1", port: int = 8787,
                   timeout: Optional[float] = None) -> Dict[str, Any]:
    """Used by `ashctl`. With no timeout given, one is chosen from the
    command: see CTL_TIMEOUT and SLOW_COMMANDS above."""
    with socket.create_connection(
            (host, port), timeout=ctl_timeout(cmd) if timeout is None else timeout) as s:
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
