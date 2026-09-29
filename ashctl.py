#!/usr/bin/env python3
"""
ashctl.py -- talk to a running ASH daemon.

    python ashctl.py status
    python ashctl.py sensors
    python ashctl.py say "what was I doing an hour ago"
    python ashctl.py why
    python ashctl.py pause / resume        # privacy: halt sensitive sensors
    python ashctl.py mute / unmute         # attention: stay quiet, keep watching
    python ashctl.py enable microphone
    python ashctl.py disable screenshot
    python ashctl.py sleep                 # force a consolidation pass
    python ashctl.py rollup                # force journal -> episodic memory
    python ashctl.py history               # last 50 decision traces (json)
    python ashctl.py history 200            # last N decision traces
    python ashctl.py history-export out.json  # write history to a file for
                                               # decision_explorer.html
    python ashctl.py shutdown

    python ashctl.py simulate mic hey ash what time is it
    python ashctl.py simulate active_window switched to Notepad
    python ashctl.py simulate idle 120
    python ashctl.py simulate battery_critical
    python ashctl.py simulate battery_low
    python ashctl.py simulate cpu_high

`pause` and `mute` are different on purpose: pause stops *perceiving*
(microphone, camera, screen), mute stops *speaking* while continuing to
observe and remember. In a meeting you usually want pause; while
concentrating you usually want mute.

`simulate` injects a synthetic sensor event through the exact same
AttentionGate/journal/response path a real sensor's event takes -- no
hardware, no quiet room, no draining a real battery required. `mic` runs
through the same addressed-detection heuristic real speech does; the
system presets (battery_low/battery_critical/cpu_high) match the real
salience/urgency numbers src/senses/devices.py uses for those conditions.
It's injected async (queued for the next tick, same as real sensors) --
follow up with `why` or `status` a moment later to see what happened.

`why` shows the most recent cognitive cycle only -- the next ambient tick
(idle/window-switch/etc. observations run every few seconds regardless)
overwrites it. `history` returns the rolling buffer of recent cycles
(bounded, oldest dropped) so a decision is still inspectable afterwards.
`history-export` writes that buffer to a JSON file that
decision_explorer.html (open it directly in a browser, no server needed)
can load to show a timeline of decisions with a click-through board-vote
breakdown per decision.
"""

import json
import sys

# A response containing an em dash or a narrow no-break space (e.g. from
# date_time_tool) can raise UnicodeEncodeError on a non-UTF-8 Windows
# console codepage. A global sys.stdout.reconfigure() "fixed" that but was
# found (on a real air-gapped machine, in ashd.py's own process) to crash
# an unrelated native library's import outright -- so instead of touching
# the stream globally, _safe_print() below catches it only where printed
# text is actually shown, at these two call sites.
def _safe_print(text: str, **kwargs):
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"), **kwargs)

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")

from src.daemon.service import control_client  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    if sys.argv[1] == "history-export":
        if len(sys.argv) < 3:
            print("usage: ashctl.py history-export <path.json> [n]", file=sys.stderr)
            return 2
        out_path = sys.argv[2]
        n = sys.argv[3] if len(sys.argv) > 3 else ""
        try:
            reply = control_client(f"history {n}".strip())
        except ConnectionRefusedError:
            print("No ASH daemon listening on 127.0.0.1:8787. Is ashd.py running?",
                  file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Control error: {e}", file=sys.stderr)
            return 1
        if not reply.get("ok"):
            print(json.dumps(reply, indent=2, default=str), file=sys.stderr)
            return 1
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reply.get("history", []), f, indent=2, default=str)
        _safe_print(f"Wrote {len(reply.get('history', []))} traces to {out_path}")
        return 0

    cmd = " ".join(sys.argv[1:])
    try:
        reply = control_client(cmd)
    except ConnectionRefusedError:
        print("No ASH daemon listening on 127.0.0.1:8787. Is ashd.py running?",
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Control error: {e}", file=sys.stderr)
        return 1

    if "report" in reply and isinstance(reply["report"], str):
        _safe_print(reply["report"])
    elif "text" in reply:
        _safe_print(reply["text"])
    else:
        print(json.dumps(reply, indent=2, default=str))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
