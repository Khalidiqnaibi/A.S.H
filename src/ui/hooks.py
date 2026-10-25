"""
src/ui/hooks.py

What the console shows, and when. Wired by wrapping the runtime, like
src/face/bridge.py: the ambient loop has no idea a window exists, and every
hook is guarded so a UI bug costs a missing chat line, never the daemon.

Every line in the window answers "who said this, how, and why":

  user     you, with the channel: typed here, or heard by the microphone
  ash      ASH, with the role: an ANSWER to you, a REMARK it volunteered, an
           OFFER (free moment / stuck) with quick replies, or a quiet NOTICE
           when its model is down
  action   something ASH did or tried: a tool it ran, an actuator it moved,
           or an action its Executive vetoed
  cue      a state that is on or off: thinking, speaking, heard you

Quick replies and the dismiss button on an offer go back in as ordinary
events, so the social learner is taught by the window exactly as it is by a
spoken "not now".
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("ash.ui.hooks")

OFFER_KINDS = ("free_moment", "stuck_help")
SPEECH_CHARS_PER_S = 14.0          # Kokoro at speed 1.0, roughly
REOPEN_MIN_S = 30.0                # never pop the window more often than this
PRESENCE_S = 300.0                 # reading the console counts as being here this long
QUICK_REPLIES = {
    "free_moment": ["Sure, let's chat", "Later", "No thanks"],
    "stuck_help": ["Yes, help me", "Not now", "No thanks"],
}


def _channel(ev) -> str:
    src = getattr(ev, "source", "")
    if src == "console":
        return "typed"
    if src in ("microphone", "mic"):
        return "voice"
    return src or "unknown"


def attach_ui(runtime, bus, opener: Optional[Callable[[], Any]] = None,
              open_on_message: bool = True, status_every_s: float = 3.0):
    """Wire `bus` into a running AmbientRuntime. Returns (on_say, on_control,
    state_extra) for the ConsoleServer."""
    orig_handle = runtime._handle
    orig_respond = runtime._respond
    orig_sink = runtime.response_sink
    last_open = [0.0]
    cycle = {"t0": None, "reported": None}      # the response being produced

    # ---- input --------------------------------------------------------
    def handle(ev):
        try:
            if getattr(ev, "addressed", False):
                bus.publish("user", text=ev.text, channel=_channel(ev))
            elif (getattr(ev, "meta", None) or {}).get("dismissed"):
                bus.publish("user", text="dismissed the offer", channel="button", quiet=True)
        except Exception:
            logger.exception("UI hook (handle) failed")
        return orig_handle(ev)

    # ---- thinking and actions -----------------------------------------
    def respond(ev, decision):
        offer = (getattr(ev, "meta", None) or {}).get("offer")
        label = ("answering you" if getattr(ev, "addressed", False)
                 else "noticed something" if offer else "considering a remark")
        try:
            bus.cue("thinking", True, label=label)
        except Exception:
            logger.exception("UI hook (thinking) failed")
        t0 = cycle["t0"] = time.time()
        try:
            return orig_respond(ev, decision)
        finally:
            cycle["t0"] = None
            try:
                if "thinking" in bus.state:
                    bus.cue("thinking", False, ms=int((time.time() - t0) * 1000))
                _report_trace(runtime, bus, t0, cycle)
            except Exception:
                logger.exception("UI hook (trace) failed")

    # ---- output -------------------------------------------------------
    def sink(text: str, meta: Dict[str, Any]):
        try:
            if cycle["t0"] is not None:          # what ASH did reads before what it says
                _report_trace(runtime, bus, cycle["t0"], cycle)
                # Done thinking as far as you can tell: the reply is here. The
                # cycle itself may run on for a while (memory, consolidation).
                if "thinking" in bus.state:
                    bus.cue("thinking", False, ms=int((time.time() - cycle["t0"]) * 1000))
            att = meta.get("attention") or {}
            kind = att.get("kind") or ""
            if meta.get("degraded"):
                role = "notice"
            elif meta.get("addressed"):
                role = "answer"
            elif kind in OFFER_KINDS:
                role = "offer"
            else:
                role = "remark"
            bus.publish("ash", text=text, role=role, offer=kind if role == "offer" else None,
                        pathway=meta.get("pathway"), source=meta.get("source"),
                        replies=QUICK_REPLIES.get(kind, []) if role == "offer" else [],
                        why=", ".join(att.get("reasons") or [])[:200])
            now = time.time()
            if (opener is not None and open_on_message and bus.clients == 0
                    and now - last_open[0] > REOPEN_MIN_S):
                last_open[0] = now
                opener()
        except Exception:
            logger.exception("UI hook (sink) failed")
        return orig_sink(text, meta)

    runtime._handle = handle
    runtime._respond = respond
    runtime.response_sink = sink
    presence = [0.0]
    _wrap_actuators(runtime, bus)
    _wrap_face(runtime, bus, presence)
    _start_status(runtime, bus, status_every_s)

    # ---- the server's callbacks ----------------------------------------
    def on_say(text: str):
        from src.senses.base import Modality, SensorEvent
        runtime.submit(SensorEvent(source="console", modality=Modality.TEXT, text=text,
                                   salience=0.9, urgency=0.6, addressed=True,
                                   meta={"typed": True}))

    def on_control(cmd: str, arg: str) -> Dict[str, Any]:
        cmd = cmd.lower().strip()
        if cmd == "presence":
            # The window is focused and being used. Keyboard/mouse idle time is
            # system-wide, so someone reading the chat looks "away" to it.
            presence[0] = time.time()
            face = getattr(runtime, "face", None)
            if face is not None:
                face.on_user_away(False)
            return {"ok": True}
        if cmd == "dismiss":
            from src.senses.base import Modality, SensorEvent
            runtime.submit(SensorEvent(source="console", modality=Modality.TEXT,
                                       text="(offer dismissed)", salience=0.0,
                                       meta={"dismissed": True}))
            return {"ok": True}
        if cmd == "pause":
            runtime.privacy.pause("console")
        elif cmd == "resume":
            runtime.privacy.resume()
        elif cmd == "mute":
            runtime.gate.mute(True)
        elif cmd == "unmute":
            runtime.gate.mute(False)
        elif cmd == "face":
            face = getattr(runtime, "face", None)
            if face is None:
                return {"ok": False, "error": "face not enabled"}
            return {"ok": bool(face.player.play(arg, force=True))}
        elif cmd in ("enable", "disable"):
            p = runtime.periphery
            target = p.sensors.get(arg) or p.actuators.get(arg)
            if target is None:
                return {"ok": False, "error": f"no peripheral named {arg!r}"}
            target.enable() if cmd == "enable" else target.disable()
        else:
            return {"ok": False, "error": f"unknown command {cmd!r}"}
        bus.publish("system", text=_CONTROL_TEXT.get(cmd, cmd).format(arg=arg))
        bus.broadcast("status", **_status(runtime))
        return {"ok": True}

    def state_extra() -> Dict[str, Any]:
        return {"status": _status(runtime)}

    return on_say, on_control, state_extra


_CONTROL_TEXT = {
    "pause": "Paused -- microphone, camera and screen are off.",
    "resume": "Resumed -- sensors are back on.",
    "mute": "Muted -- ASH won't volunteer anything until you unmute.",
    "unmute": "Unmuted -- ASH may speak up again.",
    "enable": "Turned on {arg}.",
    "disable": "Turned off {arg}.",
}


# ----------------------------------------------------------------------
def _report_trace(runtime, bus, since: float, cycle: Dict[str, Any]):
    """After a cycle: did ASH use a tool, or was an action refused?"""
    try:
        trace = runtime.ash.explain_last() or {}
    except Exception:
        return
    if not trace or float(trace.get("ts") or 0) < since - 0.5:
        return                                   # no new cycle ran (a canned line)
    if cycle.get("reported") == trace.get("ts"):
        return                                   # already shown, before the reply
    cycle["reported"] = trace.get("ts")
    verdict = trace.get("verdict") or {}
    chosen = verdict.get("chosen") or {}
    tool = chosen.get("tool_name")
    if verdict.get("vetoed_by"):
        bus.publish("action", name=tool or chosen.get("intent") or "action", status="vetoed",
                    detail="by " + ", ".join(verdict["vetoed_by"]))
    elif tool:
        out = trace.get("tool_output")
        ok = not (isinstance(out, dict) and out.get("ok") is False)
        bus.publish("action", name=tool, status="done" if ok else "failed",
                    detail=_short(chosen.get("args")), pathway=trace.get("pathway"))


def _short(x, n: int = 120) -> str:
    s = "" if x is None or x == "None" else str(x)
    return s if len(s) <= n else s[: n - 1] + "…"


def _wrap_actuators(runtime, bus):
    """Show hardware actions as they happen. `act` is wrapped per instance,
    so the base class's state and dry-run checks still run first."""
    for name, act in list(getattr(runtime.periphery, "actuators", {}).items()):
        if name == "notify":
            continue              # the window *is* the notification
        orig = act.act

        def wrapped(args, _orig=orig, _name=name):
            if _name == "speak":
                secs = max(1.2, len(str(args)) / SPEECH_CHARS_PER_S)
                bus.cue("speaking", True, seconds=round(secs, 1))
                res = _orig(args)
                if not (res or {}).get("ok"):
                    bus.cue("speaking", False)
                else:
                    threading.Timer(secs, lambda: bus.cue("speaking", False)).start()
                return res
            res = _orig(args)
            ok = bool((res or {}).get("ok"))
            bus.publish("action", name=_name, status="done" if ok else "failed",
                        detail=_short(args if ok else (res or {}).get("error")))
            return res

        act.act = wrapped


def _wrap_face(runtime, bus, presence):
    face = getattr(runtime, "face", None)
    player = getattr(face, "player", None)
    if player is None:
        return

    # Not sleepy at someone who is reading the console.
    orig_away = face.on_user_away
    face.on_user_away = lambda away: orig_away(
        bool(away) and time.time() - presence[0] > PRESENCE_S)
    prev = player.on_frame

    def on_frame(pose, img):
        if bus.clients:
            d = pose.as_dict()
            d["anim"] = player.current.name if player.current else None
            bus.publish_pose(d)
        if prev is not None:
            prev(pose, img)

    player.on_frame = on_frame


def _status(runtime) -> Dict[str, Any]:
    p = runtime.periphery
    sensors = [{"name": s.name, "on": getattr(getattr(s, "state", None), "value", "") == "active"}
               for s in p.sensors.values()]
    acts = [{"name": a.name, "on": getattr(getattr(a, "state", None), "value", "") == "active",
             "dry": bool(getattr(a, "dry_run", False))} for a in p.actuators.values()]
    gate = runtime.gate
    return {
        "sensors": sensors, "actuators": acts,
        "paused": bool(getattr(runtime.privacy, "paused", False)),
        "muted": bool(getattr(gate, "muted", False)),
        "busy": round(float(getattr(runtime, "_user_busy", 0.0) or 0.0), 2),
    }


def _start_status(runtime, bus, every: float):
    def loop():                      # daemon thread: ends with the process
        while True:
            try:
                if bus.clients:
                    bus.broadcast("status", **_status(runtime))
            except Exception:
                logger.debug("status broadcast failed", exc_info=True)
            time.sleep(every)

    threading.Thread(target=loop, daemon=True, name="ash-ui-status").start()
