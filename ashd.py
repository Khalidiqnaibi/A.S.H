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
import threading
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
            said = meta.get("spoken") or text              # in brief mode, long answers are spoken short
            r = speak({"text": said, "voice": meta["voice"]} if meta.get("voice") else said)
            if r.get("ok"):
                delivered.append("speak")

        tag = "ANSWER" if addressed else "REMARK"
        logger.info("[%s via %s] %s", tag, ",".join(delivered) or "log", text[:300])
        _safe_print(f"\nASH ({tag.lower()}): {text}\n", flush=True)

    return sink


def _file_embedder():
    """The sentence encoder ASH already loads, for file search by meaning; None (words only) if it can't be had."""
    try:
        from tools.classification import EMBEDDING_MODEL_NAME
        from tools.shared_model import load
        model = load(EMBEDDING_MODEL_NAME)
        return (lambda texts: model.encode(list(texts), show_progress_bar=False)) if model is not None else None
    except Exception:
        return None


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
    if cfg.get("llm", {}).get("gpu", True) is False:
        # The model stays off the graphics card (see tools/LLM.py).
        os.environ["ASH_OLLAMA_NUM_GPU"] = "0"
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

        # The manual: what ASH can and cannot do, in its own core memory (idempotent).
        try:
            from src.daemon import manual as _manual
            logger.info("Manual in core memory: %s", _manual.seed(getattr(ash, "core_memory", None)))
        except Exception:
            logger.exception("Could not seed the manual")

        # Load the language model now, in the background: the first question
        # after a start used to wait for the whole load. ASH_PRELOAD_MODEL=0 to skip.
        if os.environ.get("ASH_PRELOAD_MODEL", "1").strip().lower() not in ("0", "false", "no", "off"):
            def _warm(llm=getattr(ash, "llm", None)):
                t1 = time.time()
                try:
                    ok = bool(llm is not None and hasattr(llm, "warm") and llm.warm())
                except Exception:
                    ok = False
                logger.info("Model preload %s (%.1fs)", "done" if ok else "skipped", time.time() - t1)
            threading.Thread(target=_warm, name="ash-preload", daemon=True).start()
            _light = str((cfg.get("llm") or {}).get("light_model", "") or "")
            if _light and hasattr(getattr(ash, "llm", None), "set_light_model"):
                ash.llm.set_light_model(_light)
                threading.Thread(target=lambda: logger.info("Small-talk model %s: %s", _light, "ready" if ash.llm.warm(light=True) else "not loaded"),
                                 name="ash-preload-light", daemon=True).start()

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
        if cfg.get("corrections", {}).get("enabled", True):
            from src.brain import corrections  # noqa: E402
            corrections.configure(os.path.join(os.getcwd(), "state", "corrections.json"))
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
                loose_commands=bool(rcfg.get("loose_commands", True)),
                notes=bool(cfg.get("notes", {}).get("enabled", True)),
                notes_path=os.path.join(os.getcwd(), "state", "notes.json"),
                pcdoctor=bool(cfg.get("pcdoctor", {}).get("enabled", True)),
                situation=bool(cfg.get("situation", {}).get("enabled", True)),
                phone=bool(cfg.get("phone", {}).get("enabled", True)),
                phone_cfg=cfg.get("phone", {}),
                guard=bool(cfg.get("guard", {}).get("enabled", True)),
                free_after_s=float(cfg.get("free_moments", {}).get("free_after_s", 20.0)),
                dwell=bool(cfg.get("dwell", {}).get("enabled", True)),
                dwell_path=os.path.join(os.getcwd(), "state", "dwell.json"),
                dwell_away_s=float(cfg.get("dwell", {}).get("away_after_s", 600.0)),
                dwell_reading_s=float(cfg.get("dwell", {}).get("reading_after_s", 240.0)),
                dwell_welcome_min_s=float(cfg.get("dwell", {}).get("welcome_back_after_s", 600.0)),
                asks=bool(cfg.get("asks", {}).get("enabled", True)),
                asks_dir=os.path.join(os.getcwd(), "state", "asks"),
                asks_keep_replies=bool(cfg.get("asks", {}).get("keep_replies", True)),
                feelings=bool(cfg.get("feelings", {}).get("enabled", True)),
                feelings_path=os.path.join(os.getcwd(), "state", "feelings.json"),
                fun=bool(cfg.get("fun", {}).get("enabled", True)),
                fun_path=os.path.join(os.getcwd(), "state", "fun.json"),
                kids=bool(cfg.get("kids", {}).get("enabled", True)),
                kids_cfg=cfg.get("kids", {}),
                photobooth=bool(cfg.get("photobooth", {}).get("enabled", True)),
                photobooth_dir=str((cfg.get("photobooth") or {}).get("dir") or "~/Pictures/ASH"),
                filesearch=bool(cfg.get("filesearch", {}).get("enabled", True)),
                filesearch_folders=list((cfg.get("filesearch") or {}).get("folders") or ["~/Documents", "~/Desktop", "~/Downloads"]),
                filesearch_path=os.path.join(os.getcwd(), "state", "files.json"),
                filesearch_embed=_file_embedder(),
                wonder=bool(cfg.get("wondering", {}).get("enabled", True)),
                wonder_cfg=cfg.get("wondering", {}),
                lookup=bool((cfg.get("wondering", {}).get("lookup") or {}).get("enabled", True)),
                moments=bool(cfg.get("free_moments", {}).get("enabled", True)),
                moments_cfg=cfg.get("free_moments", {}),
                activity=bool(cfg.get("activity", {}).get("enabled", True)),
                activity_cfg=cfg.get("activity", {}),
                activity_path=os.path.join(os.getcwd(), "state", "activity", "topics.json"),
                taste=bool(cfg.get("taste", {}).get("enabled", True)),
                taste_cfg=cfg.get("taste", {}),
                corrections=bool(cfg.get("corrections", {}).get("enabled", True)),
                remarks_dir=(os.path.join(os.getcwd(), "state", "remarks")
                             if cfg.get("remarks", {}).get("enabled", True) else None),
                remarks_keep_text=bool(cfg.get("remarks", {}).get("keep_text", True)),
                cheer=bool(cfg.get("cheer", {}).get("enabled", True)),
                care=bool(cfg.get("care", {}).get("enabled", True)),
                story_path=os.path.join(os.getcwd(), "state", "story.json"),
                game_voice=str(cfg.get("play", {}).get("voice", "bm_george") or ""),
                cheer_path=os.path.join(os.getcwd(), "state", "cheer.json"),
                crashes=bool(cfg.get("crashes", {}).get("enabled", True)),
                crashes_path=os.path.join(os.getcwd(), "state", "crashes.json"),
                facts=True,
                facts_cfg={"enabled": bool(cfg.get("facts", {}).get("enabled", True)),
                           **{k: v for k, v in (cfg.get("facts") or {}).items() if k != "enabled"}},
                weekly=bool(cfg.get("weekly", {}).get("enabled", True)),
                weekly_path=os.path.join(os.getcwd(), "state", "weekly.json"),
                weekly_day=str(cfg.get("weekly", {}).get("day", "sunday")),
                email=bool(cfg.get("email", {}).get("enabled", True)),
                email_cfg=cfg.get("email", {}),
                weather=bool(cfg.get("weather", {}).get("enabled", True)),
                weather_cfg=cfg.get("weather", {}),
                focus=bool(cfg.get("focus", {}).get("enabled", True)),
                speech=bool(cfg.get("speech", {}).get("enabled", True)),
                tasks=bool(cfg.get("tasks", {}).get("enabled", True)),
                tasks_path=os.path.join(os.getcwd(), "state", "tasks.json"),
                speech_path=os.path.join(os.getcwd(), "state", "speech.json"),
                media=bool(cfg.get("media", {}).get("enabled", True)),
                media_cfg=cfg.get("media", {}),
                play=bool(cfg.get("play", {}).get("enabled", True)),
                reminders=bool(cfg.get("reminders", {}).get("enabled", True)),
                reminders_path=os.path.join(os.getcwd(), "state", "reminders.json"),
                smarthome=bool(cfg.get("smarthome", {}).get("enabled", True)),
                smarthome_cfg=cfg.get("smarthome", {}),
                voice=bool(cfg.get("voice", {}).get("enabled", True)),
                voice_cfg=cfg.get("voice", {}),
                voice_path=os.path.join(os.getcwd(), "state", "voice.json"),
                calendar=bool(cfg.get("calendar", {}).get("enabled", True)),
                calendar_cfg=cfg.get("calendar", {}),
                research=bool(cfg.get("research", {}).get("enabled", True)),
                research_cfg=cfg.get("research", {}),
                research_path=os.path.join(os.getcwd(), "state", "research.json"),
                desktop=bool(cfg.get("desktop", {}).get("enabled", True)),
                desktop_cfg=cfg.get("desktop", {}),
                rituals=bool(cfg.get("rituals", {}).get("enabled", True)) if isinstance(cfg.get("rituals", {}), dict) else True,
                personas=bool(cfg.get("personas", {}).get("enabled", True)) if isinstance(cfg.get("personas", {}), dict) else True,
                scenes=bool(cfg.get("scenes", {}).get("enabled", True)) if isinstance(cfg.get("scenes", {}), dict) else True,
                grammar=bool((cfg.get("address") or {}).get("grammar", True)),
                vision=True,    # always built: "look at me" is the camera's to answer, even when vision.enabled is false (it says so)
                vision_cfg=cfg.get("vision") or {},
                vision_path=os.path.join(os.getcwd(), "state", "vision.json"),
                clipboard=bool((cfg.get("clipboard") or {}).get("enabled", False)) if isinstance(cfg.get("clipboard"), dict) else False,
                arrivals=bool(cfg.get("arrivals", {}).get("enabled", True)) if isinstance(cfg.get("arrivals", {}), dict) else True,
                arrivals_path=os.path.join(os.getcwd(), "state", "arrivals.json"),
                arrivals_home=str((cfg.get("arrivals") or {}).get("home", "")) if isinstance(cfg.get("arrivals", {}), dict) else "",
                scenes_cfg={k: v for k, v in (cfg.get("scenes") or {}).items() if k != "enabled"},
                watch=bool(cfg.get("watch", {}).get("enabled", True)),
                watch_cfg=cfg.get("watch", {}),
                watch_path=os.path.join(os.getcwd(), "state", "watches.json"),
                briefings=bool(cfg.get("briefings", {}).get("enabled", True)),
                briefings_cfg=cfg.get("briefings", {}),
                briefings_path=os.path.join(os.getcwd(), "state", "briefing.json"),
                routines=bool(cfg.get("routines", {}).get("enabled", True)),
                routines_cfg=cfg.get("routines", {}),
                routines_path=os.path.join(os.getcwd(), "state", "routines", "routines.json"),
                followups=bool(cfg.get("follow_ups", {}).get("enabled", True)),
                followups_cfg=cfg.get("follow_ups", {}),
                followups_path=os.path.join(os.getcwd(), "state", "followups", "followups.json"),
                wonder_path=os.path.join(os.getcwd(), "state", "wonder", "wonders.json"),
                knowledge_path=os.path.join(os.getcwd(), "state", "wonder", "knowledge.json"),
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
                    if _shot.state.value == "unavailable":
                        return {"ok": False, "declined": True,
                                "error": "I can't look at the screen on this machine: "
                                         + str(getattr(_shot, "unavailable_reason", None) or "the capture isn't available") + "."}
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
                runtime.face = attach_to_daemon(runtime, player, sounds=bool(fcfg.get("sounds", False)))
                logger.info("Face: %d animation(s) on %s",
                            len(player.library), type(player.driver).__name__)
            except Exception:
                logger.exception("Face subsystem failed to start -- continuing without it")

        # ---- console window ----------------------------------------------
        # After the face, so the console can stream its poses.
        console = None
        if bus is not None:
            try:
                from src.daemon.service import CONFIG_FILE  # noqa: E402
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
                    open_on_message=ucfg.get("open_on_message", True),
                    config_path=args.config or CONFIG_FILE)
                if not console.start():
                    console = None
                else:
                    from src.ui.window import open_hud  # noqa: E402
                    runtime.hud_opener = lambda: open_hud(console.url, profile_dir=profile)  # noqa: E731
            except Exception:
                logger.exception("Console failed to start -- continuing without it")
                console = None

        control = None
        ccfg = cfg.get("control", {})
        if ccfg.get("enabled", True) and not args.no_control:
            control = ControlServer(runtime, ccfg.get("host", "127.0.0.1"),
                                    int(ccfg.get("port", 8787)))

        watchdog = Watchdog(runtime)

        done = threading.Event()
        quit_requested = threading.Event()
        # "Shut down" said to ASH: after the goodbye is spoken, the main loop
        # below ends and runs the same shutdown as Ctrl-C.
        runtime.request_shutdown = lambda: threading.Timer(3.0, quit_requested.set).start()

        def shutdown():
            if done.is_set():
                return                    # the signal handler and the main loop both get here
            done.set()
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
        while runtime.alive() and not quit_requested.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
