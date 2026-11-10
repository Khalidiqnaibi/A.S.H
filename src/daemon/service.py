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
        # What song or video is playing, from the titles of player windows
        # (Spotify, a YouTube tab, VLC...). Only known players are looked at.
        # taste.from_playing decides whether it teaches ASH what you like.
        "now_playing": {"enabled": True, "interval": 5.0, "skip_below_s": 30.0},
        "idle": {"enabled": True, "interval": 10.0},
        "system": {"enabled": True, "interval": 30.0},
        # Invasive: off until the user says otherwise.
        # `describe` is free and tells text from video from an empty screen.
        # `read_text` is OCR: seconds of CPU per frame and the contents of
        # the screen turned into journaled text, so it is off with capture.
        "screenshot": {"enabled": False, "interval": 20.0, "save_frames": False,
                       "describe": True, "read_text": False,
                       "read_text_every_s": 120.0, "ocr_threads": 2},
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
        "sounds": False,         # a small chime with some expressions, on lines ASH volunteers
    },
    # Wondering: ASH notices what it does not know (who "my brother" is, what
    # an acronym means) and asks. Urgent gaps may be asked at an ordinary
    # pause; the rest are saved for a free moment. Master switch: enabled.
    "wondering": {
        "enabled": True,
        "max_per_hour": 2,
        "max_per_day": 8,
        "min_gap_s": 600,
        "subject_cooldown_s": 21600,
        "expire_s": 259200,
        "urgent_at": 0.75,
        "keep_evidence": True,
        "sources": ["console", "microphone", "mic"],   # add "screen" to wonder about on-screen text
        "ignore_names": [],
        "kinds": {"person": True, "role": True, "between": True, "term": True,
                  "fact": True, "ambiguous": True, "intent": True,
                  "conflict": True, "stale": True},
        # A relation that can change (boss, roommate, partner...) is checked
        # when the person comes up after this long: "Is Dana still your boss?"
        "stale_after_s": 15552000,
        # Before asking what a term means, look it up and ask to confirm the
        # guess. memory: past sentences; files: text files under roots (add
        # your project folders); web: a Wikipedia summary, only for acronyms
        # and quoted terms, only the term is sent. Set web false air-gapped.
        "lookup": {
            "enabled": True,
            "memory": True,
            "files": True,
            "roots": ["~/Documents", "~/Desktop"],
            "web": True,
        },
    },
    # A free moment (a pause after real work) can hold several things: closing
    # a gap (see wondering), a light joke, asking how the user is, a note on how
    # ASH is running. They compete on value x fit; this is their shared budget.
    # What the user does, kept like a conversation: a sentence per real stretch
    # of work in episodic memory, the parts that connect to something already
    # known saved as entities, and the ones that stay unexplained asked about
    # at a free moment (see activity.py). Reads window title fields and a few
    # keywords from the screen reader, never the text; withheld windows give
    # it nothing to learn from.
    "activity": {
        "enabled": True,
        "remember": True,        # the sentence per stretch
        "ask": True,             # wonder about connected, unexplained topics
        "keep_terms": True,      # keep keywords read off the screen
        "min_stretch_s": 180.0,  # shorter than this is a glance
        "episode_min_s": 300.0,  # a stretch this long earns a sentence
        "max_episodes_per_hour": 6,
    },
    # Taste: what the user, and the people they talk about, like and dislike
    # ("I love Interstellar", "Ali likes jazz when he's sad"), learned from
    # what is said and kept with what ASH knows about people (taste.py). It
    # answers "what do I like?" itself and gives the model the list only when
    # the question is about choosing something. Never learned from a guest's
    # voice.
    "taste": {
        "enabled": True,
        "learn_from_others": True,   # "Ali likes jazz", on the user's word
        "mirror": True,              # a short likes/dislikes on the person's entity
        "in_answers": True,          # background for "what should I watch?"
        # What is played, from the now_playing sensor: a song heard to the end
        # counts a little for it, one skipped counts against it, a favourite
        # builds up over plays. Weaker than being told, and never overrides it.
        "from_playing": True,
        "listen_s": 90.0,            # a song this long counts as listened to
        "watch_s": 600.0,            # a video or an episode this long counts as watched
    },
    # Follow-ups: "my interview is on Thursday" is remembered, and once it is
    # over ASH asks how it went, at a free moment, once. Meetings and calls are
    # followed up too unless routine is off; they are worth less than an exam.
    "follow_ups": {
        "enabled": True,
        "routine": True,
        "window_s": 259200,      # how long after the event the question is still worth asking
        "max_open": 25,
    },
    # Being corrected ("no, I meant ...") right after a reply that used a tool:
    # the reading that was wrong is remembered and nudged down for the same
    # wording next time, the meant one up. Kept in state/corrections.json.
    "corrections": {"enabled": True},
    # Watches: "tell me when the download finishes / chrome closes / the
    # battery is charged / the CPU calms down / report.docx changes", checked
    # every few seconds and let go after a day; plus a warning when the disk
    # will be full within two days at the rate it is filling.
    # Desktop: "open Outlook", "switch to Chrome", "close Word" (asked first,
    # and a polite close the app can still interrupt to save), "run my
    # morning check". Only known app names; add yours under apps, e.g.
    # "apps": {"jira": "https://jira.example.com", "blender": "C:/Program Files/Blender/blender.exe"}.
    "desktop": {"enabled": True, "confirm_close": True, "apps": {}},
    # Research: "look into X" -- looked up in the background (Wikipedia, your
    # files under wondering.lookup.roots, what you said before) and told at a
    # free moment or in the evening briefing.
    "research": {"enabled": True},
    # Calendar, read-only, from iCalendar: exported .ics files, or a private
    # feed URL (Google: "secret address in iCal format"; Outlook: "publish
    # calendar"). Used in the briefings and for "what's on my calendar today".
    "calendar": {"enabled": True, "files": [], "urls": [], "refresh_s": 1800},
    # Home Assistant on your network: its address and a long-lived access
    # token (your HA profile -> Security). Lights, switches and fans only;
    # locks, doors, garages, alarms and heaters are never operated by voice.
    "smarthome": {"enabled": True, "url": "", "token": ""},
    # Reminders, timers and alarms that actually go off ("remind me to call
    # mom in 20 minutes", "set a timer for 5 minutes", "wake me up at 7"),
    # snoozable, kept across restarts, and listed in the morning briefing.
    "reminders": {"enabled": True},
    # Play: "let's play a game" (a number game, or a quiz on your own people);
    # at a free moment ASH may say it is bored (only when your mood is good)
    # or offer an evening story. free_moments.disabled can drop bored/story.
    # voice: the TTS voice game lines are spoken in ("" for the usual one).
    "play": {"enabled": True, "voice": "bm_george"},
    # The remark log: everything ASH says and how it landed, kept two weeks
    # under state/remarks/, read back by `ashctl review [days]` with tips on
    # what to tune. keep_text false keeps only the kind and the reaction.
    "remarks": {"enabled": True, "keep_text": True},
    # Focus mode: "I'm focusing for an hour" / "do not disturb until 3" -- ASH
    # volunteers nothing until then (reminders and watches still go off) and
    # says what it held back when it ends.
    "focus": {"enabled": True},
    # "Stop" / "shh" cut ASH off; "the short version"; "keep it short".
    "speech": {"enabled": True},
    # Multi-step requests ("open Outlook and set a timer for 25 minutes") and
    # named tasks ("when I say start work, ..."), shown as a plan, run on a yes.
    "tasks": {"enabled": True},
    # gpu: false keeps the language model off the graphics card (Ollama runs it
    # on the CPU: slower, but nothing for a shaky GPU driver to trip over).
    "llm": {"gpu": True},
    # Email, read-only: the unread count and who sent what, in the briefing
    # and when asked. Nothing happens until an IMAP account is set; use an
    # app password. Bodies are never read, nothing is marked read or sent.
    # "How was my week?" -- and in the evening briefing on this day ("" for never).
    "weekly": {"enabled": True, "day": "sunday"},
    # Tells you about unexpected shutdowns (Windows event log), in the briefing
    # or when asked "did my computer crash?".
    "crashes": {"enabled": True},
    # Good moments brought back when you're worried; birthday, streaks and
    # long stretches celebrated.
    "cheer": {"enabled": True},
    # Water, a stretch, "you've been at it three hours" -- at a free moment,
    # only when the mood is fine, at most once an hour.
    "care": {"enabled": True},
    "email": {"enabled": True, "host": "", "port": 993, "user": "", "password": "", "folder": "INBOX"},
    # The weather in the briefings and for "what's the weather like today",
    # from Open-Meteo (no account or key). Set city (or lat and lon); until
    # then nothing is fetched. Only the place is sent, at most hourly.
    "weather": {"enabled": True, "city": "", "lat": None, "lon": None},
    # Media: "pause the music", "next song", "turn it up" (the media keys, any
    # player); a chime with reminders and alerts; "take a photo" (one webcam
    # frame to photo_dir, on request only, never while paused).
    "media": {"enabled": True, "music": True, "sounds": True, "photos": True,
              "photo_dir": "~/Pictures/ASH", "camera_index": 0},
    # Voice match: nothing is compared until you say "learn my voice". After
    # that a clearly different voice is a guest: answered, but not allowed to
    # act for you or teach ASH about you. A rough match, not security.
    "voice": {"enabled": True, "samples": 6},
    "watch": {"enabled": True, "downloads_dir": "~/Downloads", "expire_s": 86400, "disk_trend": True},
    # Briefings: a short summary the first time you are at the computer in
    # the morning (today's events, yesterday's work, questions waiting) and
    # at a free moment in the evening (what the day went on, tomorrow's
    # events). "Brief me" asks for one any time.
    "briefings": {"enabled": True, "morning": True, "evening": True,
                  "morning_from": 5, "morning_until": 12, "evening_from": 18, "evening_until": 23},
    # Routines: the apps opened at the start of a sitting, in order. One that
    # repeats on min_days different days is offered as a routine to keep
    # ("Outlook, then Teams, then Jira"). Only app names are kept.
    "routines": {"enabled": True, "min_days": 3, "window_s": 600, "away_s": 1800},
    "free_moments": {
        "enabled": True,
        "min_gap_s": 600,
        "max_per_hour": 2,
        "disabled": [],          # by name: gap_closure joke mood_check system_assessment follow_up
        "jokes": True,
        "mood_check": True,
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
            toks = arg.split()
            everything = "all" in [t.lower() for t in toks]          # "history all" / "history 100 all": observations too
            nums = [t for t in toks if t.isdigit()]
            n = int(nums[0]) if nums else 50
            return {"ok": True, "history": rt.ash.explain_history(n, observed=everything)}
        if cmd == "sleep":
            return {"ok": True, "report": rt._sleep_phase(force=True)}
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
        if cmd in ("wonders", "wonder"):
            return self._wonders(arg if cmd == "wonder" else "")
        if cmd == "review":
            log = getattr(rt, "remarks", None)
            if log is None:
                return {"ok": False, "error": "the remark log is off (daemon.json: remarks.enabled)"}
            try:
                days = max(1, min(14, int(arg.strip() or 1)))
            except ValueError:
                days = 1
            return {"ok": True, "report": log.report(days), "review": log.review(days)}
        if cmd == "watches":
            wt = getattr(rt, "watcher", None)
            if wt is None:
                return {"ok": False, "error": "watches are not enabled (daemon.json: watch.enabled)"}
            if arg.strip() in ("stop", "clear", "cancel"):
                wt.cancel("", rt.clock())
            elif arg.strip():
                return {"ok": False, "error": f"unknown watches command {arg.strip()!r}", "commands": ["stop"]}
            return {"ok": True, "report": wt.report(), "status": wt.status()}
        if cmd == "briefing":
            b = getattr(rt, "briefings", None)
            if b is None:
                return {"ok": False, "error": "briefings are not enabled (daemon.json: briefings.enabled)"}
            kind = arg.strip() if arg.strip() in ("morning", "evening") else ""
            return {"ok": True, "report": b.compose(kind) or "(nothing to brief)", "status": b.status()}
        if cmd == "routines":
            r = getattr(rt, "routines", None)
            if r is None:
                return {"ok": False, "error": "routines are not enabled (daemon.json: routines.enabled)"}
            if arg.strip() == "forget":
                r.forget()
            elif arg.strip():
                return {"ok": False, "error": f"unknown routines command {arg.strip()!r}", "commands": ["forget"]}
            kb = rt.wonder.kb if getattr(rt, "wonder", None) else None
            kept = [f"  {p['name']}: " + ", then ".join(p["steps"]) for p in (kb.procedures.values() if kb else [])]
            return {"ok": True, "report": r.report() + ("\nKept:\n" + "\n".join(kept) if kept else ""),
                    "status": r.status()}
        if cmd == "corrections":
            from src.brain import corrections
            c = corrections.default()
            if arg.strip() == "forget":
                c.forget()
            elif arg.strip():
                return {"ok": False, "error": f"unknown corrections command {arg.strip()!r}",
                        "commands": ["forget"]}
            return {"ok": True, "report": c.report(), "status": c.status()}
        if cmd == "followups":
            f = getattr(rt, "followups", None)
            if f is None:
                return {"ok": False, "error": "follow-ups are not enabled (daemon.json: follow_ups.enabled)"}
            parts = arg.split()
            if parts and parts[0] == "dismiss" and len(parts) > 1:
                if not f.dismiss(parts[1]):
                    return {"ok": False, "error": f"no open follow-up {parts[1]!r}"}
            elif parts:
                return {"ok": False, "error": f"unknown followups command {parts[0]!r}",
                        "commands": ["dismiss <id>"]}
            return {"ok": True, "report": f.report(), "pending": f.pending(), "status": f.status()}
        if cmd == "activity":
            a = getattr(rt, "activity", None)
            if a is None:
                return {"ok": False, "error": "activity memory is not enabled "
                                              "(daemon.json: activity.enabled)"}
            if arg.strip() == "forget":
                a.forget()
                return {"ok": True, "report": "activity memory cleared"}
            return {"ok": True, "report": a.report(), "status": a.status()}
        if cmd == "profile":
            t = getattr(rt, "taste", None)
            if t is None:
                return {"ok": False, "error": "taste is not enabled (daemon.json: taste.enabled)"}
            what = arg.strip()
            if what.startswith("forget"):
                gone = t.forget(what[len("forget"):].strip())
                return {"ok": True, "report": f"forgot {gone} taste(s)"}
            return {"ok": True, "report": t.report(), "status": t.status()}
        if cmd == "shutdown":
            threading.Thread(target=rt.stop, daemon=True).start()
            return {"ok": True, "stopping": True}
        return {"ok": False, "error": f"unknown command {cmd!r}",
                "commands": ["status", "pause", "resume", "mute", "unmute", "say",
                             "why", "history", "sleep", "rollup", "sensors", "face", "wonders", "wonder", "activity", "profile", "followups", "corrections", "routines", "briefing", "watches", "review",
                             "enable", "disable", "simulate", "shutdown"]}

    # ------------------------------------------------------------------
    def _wonders(self, arg: str) -> Dict[str, Any]:
        """wonders                list what ASH is wondering about
        wonder on|off          master switch (kept across restarts)
        wonder dismiss <id>    drop one
        wonder snooze [min]    no questions for a while
        wonder clear           drop everything pending"""
        w = getattr(self.runtime, "wonder", None)
        if w is None:
            return {"ok": False, "error": "wondering is not enabled (daemon.json: wondering.enabled)"}
        parts = arg.split()
        sub = parts[0].lower() if parts else ""
        if sub in ("on", "off"):
            w.set_enabled(sub == "on")
        elif sub == "dismiss" and len(parts) > 1:
            if not w.dismiss(parts[1]):
                return {"ok": False, "error": f"no open wonder {parts[1]!r}"}
        elif sub == "snooze":
            try:
                mins = float(parts[1]) if len(parts) > 1 else None
            except ValueError:
                mins = None
            w.snooze_all(None if mins is None else mins * 60)
        elif sub == "clear":
            w.clear()
        elif sub:
            return {"ok": False, "error": f"unknown wonder command {sub!r}",
                    "commands": ["on", "off", "dismiss <id>", "snooze [min]", "clear"]}
        return {"ok": True, "report": w.report(), "pending": w.pending(), "status": w.status()}

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
