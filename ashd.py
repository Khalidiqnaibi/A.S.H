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


def build_response_sink(periphery, prefer_speech: bool = True):
    """How ASH actually reaches the user.

    Notification first, speech second. A notification is glanceable and
    ignorable; synthesized speech in a room with other people in it is not,
    which is why an unprompted remark should default to the quiet channel.
    """
    notify = periphery.actuators.get("notify")
    speak = periphery.actuators.get("speak")

    def sink(text: str, meta: dict):
        addressed = meta.get("addressed", False)
        delivered = []

        if notify is not None and notify.state.value == "active":
            r = notify(text)
            if r.get("ok"):
                delivered.append("notify")

        # Speak when spoken to. Volunteered remarks stay silent unless the
        # user has explicitly asked for a talkative ASH.
        if speak is not None and speak.state.value == "active" and (addressed or prefer_speech):
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
            tts = TTSEngine(model_path="models/kokoro-v1.0.onnx",
                            voices_path="models/voices-v1.0.bin")
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
            ),
            response_sink=build_response_sink(periphery,
                                              prefer_speech=rcfg.get("speak_unprompted", False)),
        )

        # ---- face -----------------------------------------------------------
        # Attached by wrapping the runtime's hooks, so the ambient loop has no
        # knowledge of the face and a display failure cannot take it down.
        fcfg = cfg.get("face", {})
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
