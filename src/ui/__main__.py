"""
Preview the console without the daemon, models or sensors.

    python -m src.ui demo [--port 8799] [--theme blossom] [--no-window]

Runs the real face player and the real server with a scripted conversation
that walks through every kind of line and cue the console can show. Typing
gets a canned echo; clicking the face plays an animation. Useful for themes
and animations, where waiting for the real ASH to feel something is slow.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.face import FacePlayer, FaceRenderer, NullDriver  # noqa: E402
from src.ui import ConsoleServer, UIBus, open_window  # noqa: E402

SCRIPT = [
    (1.0, "face", "boot"),
    (2.5, "user", ("hey ash, what time is it?", "voice")),
    (0.3, "think", "answering you"),
    (1.4, "action", ("get_time", "done", "")),
    (0.2, "ash", ("It's 4:12 PM.", "answer", None, "fast")),
    (0.1, "speak", 2.0),
    (3.0, "face", "curious"),
    (2.0, "think", "considering a remark"),
    (1.5, "ash", ("Your build has failed three times with the same import error.", "remark", None, "slow")),
    (3.0, "face", "confused"),
    (1.0, "ash", ("Looks like that ImportError keeps coming back -- want me to take a look?",
                  "offer", "stuck_help", "offer")),
    (4.0, "action", ("launch", "vetoed", "by the safety check")),
    (2.0, "face", "happy"),
    (2.0, "ash", ("Nice focus -- 45 minutes in VS Code without stopping.", "offer", "free_moment", "offer")),
    (3.0, "face", "love"),
    (3.0, "face", "sleepy"),
    (4.0, "ash", ("My language model isn't responding, so I can't answer that right now.",
                  "notice", None, "degraded")),
    (2.0, "face", "very_happy"),
]
REPLIES = {"free_moment": ["Sure, let's chat", "Later", "No thanks"],
           "stuck_help": ["Yes, help me", "Not now", "No thanks"]}


def run_script(bus, player, loop: bool):
    while True:
        for delay, kind, arg in SCRIPT:
            time.sleep(delay)
            if kind == "face":
                player.play(arg, force=True)
            elif kind == "user":
                bus.publish("user", text=arg[0], channel=arg[1])
                player.play("listening", force=True)
            elif kind == "think":
                bus.cue("thinking", True, label=arg)
                player.play("thinking", force=True)
            elif kind == "action":
                bus.cue("thinking", False)
                bus.publish("action", name=arg[0], status=arg[1], detail=arg[2])
            elif kind == "ash":
                bus.cue("thinking", False)
                text, role, offer, path = arg
                bus.publish("ash", text=text, role=role, offer=offer, pathway=path,
                            replies=REPLIES.get(offer, []), why="demo")
                player.play("acknowledge")
            elif kind == "speak":
                bus.cue("speaking", True, seconds=arg)
                threading.Timer(arg, lambda: bus.cue("speaking", False)).start()
        if not loop:
            return
        time.sleep(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="demo")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--theme", default="nebula")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()

    bus = UIBus()
    player = FacePlayer(renderer=FaceRenderer(), driver=NullDriver(), fps=30)

    def on_frame(pose, img):
        if bus.clients:
            d = pose.as_dict()
            d["anim"] = player.current.name if player.current else None
            bus.publish_pose(d)

    player.on_frame = on_frame
    player.start()

    def on_say(text):
        bus.publish("user", text=text, channel="typed")
        bus.cue("thinking", True, label="answering you")
        player.play("thinking", force=True)

        def later():
            time.sleep(1.2)
            bus.cue("thinking", False)
            bus.publish("ash", text=f"(demo) You said: {text}", role="answer", pathway="fast")
            player.play("happy", force=True)
        threading.Thread(target=later, daemon=True).start()

    def on_control(cmd, arg):
        if cmd == "face":
            return {"ok": player.play(arg, force=True)}
        bus.publish("system", text=f"(demo) {cmd} {arg}".strip())
        return {"ok": True}

    srv = ConsoleServer(bus, port=a.port, theme=a.theme, on_say=on_say, on_control=on_control,
                        state_extra=lambda: {"status": {
                            "sensors": [{"name": "microphone", "on": True}, {"name": "camera", "on": False},
                                        {"name": "screenshot", "on": False}, {"name": "active_window", "on": True},
                                        {"name": "idle", "on": True}],
                            "paused": False, "muted": False}})
    if not srv.start():
        return 1
    print("Console demo on", srv.url, flush=True)
    if not a.no_window:
        open_window(srv.url)
    threading.Thread(target=run_script, args=(bus, player, a.loop), daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
        player.shutdown(play_outro=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
