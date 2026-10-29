#!/usr/bin/env python3
"""
ashd.py -- the ASH daemon. Run this and leave it running.

    python ashd.py                 start in the foreground
    python ashd.py --config X.json use a different config
    python ashd.py --dry           start with every actuator forced to dry-run
    python ashd.py --check         print what it can perceive/do, then exit

Talk to a running daemon with ashctl.py:

    python ashctl.py status
    python ashctl.py say "what have I been working on"
    python ashctl.py pause          # stop mic/camera/screen immediately
    python ashctl.py enable microphone

Startup order matters and is deliberate:

  1. Logging first, so a failure in step 2 is actually recorded somewhere.
  2. Single-instance lock, before touching any device.
  3. ASH itself (this is the slow part -- model loads).
  4. Periphery, then bind actuators to the VLA channel *and* the tool
     registry. Binding to the VLA is what puts hardware under the Executive's
     veto; binding to the registry is what lets the classifier route to it.
     Both are required, and doing only the second would give the brain motors
     with no safety review.
  5. Ambient loop, control socket, watchdog.

Running it as a real service
----------------------------
This process handles its own signals and cleanup but does not daemonize
itself; use the platform supervisor, which is better at restarts than
anything written here would be.

  systemd  -- ExecStart=/path/to/venv/bin/python /path/to/ashd.py
              Restart=always, RestartSec=10, After=graphical-session.target
  Windows  -- Task Scheduler, "At log on", restart on failure, or NSSM
  macOS    -- launchd plist with KeepAlive=true
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Windows' default console codepage (e.g. cp1256, cp1252 -- whatever the
# system locale is) can't encode plenty of ordinary text this daemon prints:
# an em dash in a narrated response, the narrow no-break space
# date_time_tool puts before AM/PM, etc. A naive sys.stdout.reconfigure(
# encoding="utf-8") "fixes" that but was found (on a real air-gapped
# machine) to crash sentence-transformers' import outright -- some native
# extension in that chain (huggingface_hub's download progress code, or
# hf_xet) touches the reconfigured stream in a way it doesn't handle. So
# instead of touching the stream globally, _safe_print() below catches the
# UnicodeEncodeError at the one or two print() call sites that actually
# need it, leaving sys.stdout/stderr completely untouched for everything
# else that imports during startup.
def _safe_print(text: str, **kwargs):
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"), **kwargs)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.daemon import (  # noqa: E402
    AmbientConfig, AmbientRuntime, AttentionGate, ControlServer, Journal,
    PrivacyPolicy, SingleInstance, Watchdog, install_signal_handlers,
    load_config, setup_logging,
)

logger = logging.getLogger("ashd")


def find_model_file(name: str) -> str:
    """Locate a model file. `models/` is git-ignored, so a git worktree has
    none of its own; fall back to the main checkout's copy rather than going
    silently mute when launched from a worktree or another directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [os.environ.get("ASH_MODELS_DIR"), os.path.join(os.getcwd(), "models"),
             os.path.join(here, "models")]
    try:
        gitfile = os.path.join(here, ".git")
        if os.path.isfile(gitfile):  # worktree: "gitdir: <main>/.git/worktrees/<name>"
            gd = open(gitfile, encoding="utf-8").read().split(":", 1)[1].strip()
            roots.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(gd))), "models"))
    except Exception:
        pass
    for r in roots:
        if r and os.path.isfile(os.path.join(r, name)):
            return os.path.join(r, name)
    return os.path.join("models", name)


def build_response_sink(periphery, prefer_speech: bool = True, window_open=None):
    """How ASH actually reaches the user.

    Notification first, speech second. When the console window is open it
    already shows the line, so the notification is skipped rather than
    doubled.

    `prefer_speech` decides whether the things ASH says *unprompted* are
    spoken as well as shown -- the free-moment question, the offer of help
    when you look stuck, an observation worth a sentence. It used to default
    to off here, on the reasoning that a notification is glanceable and
    ignorable while synthesized speech in a room with other people in it is
    not.

    That was the wrong place to make that decision, and the effect was worse
    than the problem it avoided. ASH would decide a moment was right, compose
    a question, put it on screen with quick replies -- and say nothing. The
    one feature whose whole point is that it speaks up without being asked
    did it silently, which reads less like tact than like a thing that cannot
    talk unless spoken to.

    Everything that gets here unprompted has already been rationed hard by
    the attention gate: six unsolicited remarks an hour at most, three offers,
    none during quiet hours, none while the refractory timer is up, none while
    the user is typing, and the whole lot down-weighted by what the social
    learner has seen them dismiss. Suppressing the voice here is a second
    veto on a decision that machinery already made carefully. So it defaults
    to on, and anyone who wants the quiet channel sets runtime.speak_unprompted
    to false.

    What is still never spoken: a notice that the model is down. That is ASH
    apologising for being broken, and it goes to the quiet channel only.
    """
    notify = periphery.actuators.get("notify")
    speak = periphery.actuators.get("speak")

    def sink(text: str, meta: dict):
        addressed = meta.get("addressed", False)
        delivered = []

        if (notify is not None and notify.state.value == "active"
                and not (window_open is not None and window_open())):
            r = notify(text)
            if r.get("ok"):
                delivered.append("notify")

        # Spoken when spoken to, and -- unless runtime.speak_unprompted says
        # otherwise -- when ASH decided on its own that this was worth saying.
        # Never an apology for being broken.
        if (speak is not None and speak.state.value == "active"
                and (addressed or prefer_speech) and not meta.get("degraded")):
            r = speak(text)
            if r.get("ok"):
                delivered.append("speak")

        tag = "ANSWER" if addressed else "REMARK"
        logger.info("[%s via %s] %s", tag, ",".join(delivered) or "log", text[:300])
        _safe_print(f"\nASH ({tag.lower()}): {text}\n", flush=True)

    return sink


def main() -> int:
    ap = argparse.ArgumentParser(description="ASH always-on daemon")
    ap.add_argument("--config", default=None, help="path to daemon.json")
    ap.add_argument("--dry", action="store_true", help="force every actuator to dry-run")
    ap.add_argument("--check", action="store_true", help="report capabilities and exit")
    ap.add_argument("--no-control", action="store_true", help="disable the control socket")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="log at DEBUG and print every degrade-gracefully failure "
                         "(model load, TTS, face, etc.) with its full traceback")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log_cfg = dict(cfg.get("log", {}))
    if args.verbose:
        log_cfg["level"] = "DEBUG"
    setup_logging(log_cfg)

    logger.info("=" * 62)
    logger.info("ASH daemon starting (pid %d)", os.getpid())

    lock = SingleInstance()
    if not lock.acquire():
        return 1

    # ---- privacy policy first, so it exists before any sensor does -----
    pcfg = cfg.get("privacy", {})
    privacy = PrivacyPolicy(
        redact=pcfg.get("redact", True),
        require_local_asr=pcfg.get("require_local_asr", True),
        raw_journal_days=pcfg.get("raw_journal_days", 7),
        frame_days=pcfg.get("frame_days", 2),
        audio_days=pcfg.get("audio_days", 0),
        blocked_apps=pcfg.get("blocked_apps", []),
    )
    print(privacy.banner(), flush=True)

    # ---- the brain -----------------------------------------------------
    # Everything from here through periphery/runtime wiring is wrapped so a
    # startup failure anywhere in it -- a bad import, a missing model file,
    # a memory-store init error -- is always logged with its full traceback
    # (via logger.critical(..., exc_info=True)) to BOTH the console and
    # logs/ashd.log, rather than relying on an uncaught exception's raw
    # stderr traceback, which never reaches the log file and can be lost.
    try:
        logger.info("Loading ASH (models, memory, tools) ...")
        t0 = time.time()
        from src.py.ash import ash  # noqa: E402  (slow import: model loads)
        logger.info("ASH loaded in %.1fs", time.time() - t0)

        # ---- periphery ------------------------------------------------------
        from src.senses import build_periphery  # noqa: E402

        tts = None
        try:
            from tools import TTSEngine
            tts = TTSEngine(model_path=find_model_file("kokoro-v1.0.onnx"),
                            voices_path=find_model_file("voices-v1.0.bin"))
        except Exception as e:
            logger.warning("TTS unavailable; speech actuator will be inert -- %s: %s",
                          type(e).__name__, e, exc_info=args.verbose)

        periphery = build_periphery(cfg, tts_engine=tts)

        if args.dry:
            for a in periphery.actuators.values():
                a.dry_run = True
            logger.warning("--dry: every actuator forced to dry-run")

        # Hardware inherits the brain's safety model here. Both bindings matter:
        # the VLA binding is the veto path, the registry binding is the routing
        # path. See ashd.py's module docstring.
        from tools.registry import REGISTRY as TOOL_REGISTRY, ToolEntry  # noqa: E402
        from src.brain import ActionClass  # noqa: E402

        periphery.bind_to_vla(ash.brain.vla, ActionClass)
        n = periphery.bind_to_registry(TOOL_REGISTRY, ToolEntry)
        logger.info("Exposed %d actuator(s) to the classifier", n)

        print(periphery.report(), flush=True)

        if args.check:
            logger.info("--check: exiting after capability report")
            lock.release()
            return 0

        # ---- runtime ---------------------------------------------------------
        acfg = cfg.get("attention", {})
        qh = acfg.get("quiet_hours")
        gate = AttentionGate(
            base_threshold=acfg.get("base_threshold", 0.62),
            refractory_seconds=acfg.get("refractory_seconds", 180),
            max_unsolicited_per_hour=acfg.get("max_unsolicited_per_hour", 6),
            quiet_hours=(tuple(qh) if qh else None),
        )

        ucfg = cfg.get("ui", {})
        bus = None
        if ucfg.get("enabled", True):
            from src.ui import UIBus  # noqa: E402
            bus = UIBus()

        rcfg = cfg.get("runtime", {})
        runtime = AmbientRuntime(
            ash=ash,
            periphery=periphery,
            journal=Journal(),
            gate=gate,
            privacy=privacy,
            config=AmbientConfig(
                tick_seconds=rcfg.get("tick_seconds", 1.0),
                sleep_after_idle_s=rcfg.get("sleep_after_idle_s", 1500),
                learn_social=rcfg.get("learn_social", True),
                flow_offers=rcfg.get("flow_offers", True),
                check_in_with_model=rcfg.get("check_in_with_model", False),
                flow_min_burst_s=rcfg.get("flow_min_burst_s", 600.0),
                social_path=os.path.join(
                    os.environ.get("ASH_BRAIN_STATE", os.path.join(os.getcwd(), "state", "brain")),
                    "social.json"),
            ),
            response_sink=build_response_sink(periphery,
                                              prefer_speech=rcfg.get("speak_unprompted", True),
                                              window_open=(lambda: bus.clients > 0) if bus else None),
        )

        # ---- "what do you see" ----------------------------------------------
        # The screen tool holds no sensor of its own; it is handed this one's
        # look function, behind the same privacy pause that stops the sensor.
        try:
            from tools import screen_tool  # noqa: E402

            shot = periphery.sensors.get("screenshot")
            if shot is not None:
                def _look(_shot=shot):
                    if not privacy.allows(_shot):
                        return {"ok": True, "withheld": "paused"}
                    if _shot.state.value == "disabled":
                        return {"ok": True, "withheld": "off"}
                    return _shot.look_now()
                screen_tool.set_looker(_look)
        except Exception:
            logger.exception("Could not attach the screen tool -- continuing without it")

        # ---- face -----------------------------------------------------------
        # Attached by wrapping the runtime's hooks, so the ambient loop has no
        # knowledge of the face and a display failure cannot take it down.
        fcfg = cfg.get("face", {})
        if bus is not None and fcfg.get("driver") is None:
            fcfg = dict(fcfg, driver="null")   # the console window shows the face
        if fcfg.get("enabled", True):
            try:
                from src.face import attach_to_daemon, build_face

                player = build_face(
                    width=int(fcfg.get("width", 128)),
                    height=int(fcfg.get("height", 64)),
                    driver=fcfg.get("driver"),
                    fps=int(fcfg.get("fps", 30)),
                    animations_dir=fcfg.get("animations_dir", "animations"),
                )
                player.start()
                runtime.face = attach_to_daemon(runtime, player)
                logger.info("Face: %d animation(s) on %s",
                            len(player.library), type(player.driver).__name__)
            except Exception:
                logger.exception("Face subsystem failed to start -- continuing without it")

        # ---- console window ----------------------------------------------
        # After the face, so the console can stream its poses.
        console = None
        if bus is not None:
            try:
                from src.ui import ConsoleServer, attach_ui, open_window  # noqa: E402
                state_dir = os.environ.get("ASH_BRAIN_STATE",
                                           os.path.join(os.getcwd(), "state", "brain"))
                console = ConsoleServer(bus, port=int(ucfg.get("port", 8788)),
                                        theme=ucfg.get("theme", "nebula"),
                                        prefs_path=os.path.join(os.path.dirname(state_dir), "ui.json"))
                profile = os.path.join(os.path.dirname(state_dir), "ui-window")
                opener = lambda: open_window(console.url, profile_dir=profile)  # noqa: E731
                console.on_say, console.on_control, console.state_extra = attach_ui(
                    runtime, bus, opener=opener,
                    open_on_message=ucfg.get("open_on_message", True))
                if not console.start():
                    console = None
            except Exception:
                logger.exception("Console failed to start -- continuing without it")
                console = None

        control = None
        ccfg = cfg.get("control", {})
        if ccfg.get("enabled", True) and not args.no_control:
            control = ControlServer(runtime, ccfg.get("host", "127.0.0.1"),
                                    int(ccfg.get("port", 8787)))

        watchdog = Watchdog(runtime)

        def shutdown():
            try:
                face = getattr(runtime, "face", None)
                if face is not None:
                    face.on_shutdown()
                    face.player.shutdown(play_outro=True)
            except Exception:
                pass
            try:
                watchdog.stop()
                if console:
                    console.stop()
                if control:
                    control.stop()
                runtime.stop()
            finally:
                lock.release()

        install_signal_handlers(shutdown)

        runtime.start()
        if control:
            control.start()
        watchdog.start()

        if privacy.announce_on_start:
            active = [s.name for s in periphery.active_sensors()]
            logger.warning("ASH is now listening/observing via: %s", ", ".join(active) or "nothing")

        if console and ucfg.get("open_on_start", True):
            how = opener()
            logger.info("Console window opened (%s): %s", how, console.url)
        logger.info("Daemon up. Ctrl-C to stop. Control: ashctl.py status")
    except Exception:
        logger.critical("ASH daemon failed to start -- see traceback below for "
                        "exactly where and why", exc_info=True)
        lock.release()
        return 1

    try:
        while runtime.alive():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
