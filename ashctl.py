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
    python ashctl.py shutdown

`pause` and `mute` are different on purpose: pause stops *perceiving*
(microphone, camera, screen), mute stops *speaking* while continuing to
observe and remember. In a meeting you usually want pause; while
concentrating you usually want mute.
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
