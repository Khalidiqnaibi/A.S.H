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
        print(reply["report"])
    elif "text" in reply:
        print(reply["text"])
    else:
        print(json.dumps(reply, indent=2, default=str))
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
