"""
tst_service_fixes.py -- control socket and watchdog.

  V1  `ashctl shutdown` asks the daemon's single shutdown to run; it did not start a second stop alongside it
  V2  the watchdog restarts the loop only -- not the whole shutdown (sensors closed, learning saved, the run marked ended)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_service_fixes.py
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


class FakeRuntime:
    def __init__(self, ask=True):
        self.calls = []
        if ask:
            self.request_shutdown = lambda: self.calls.append("request_shutdown")

    def stop(self, timeout=5.0):
        self.calls.append("stop")

    def start(self):
        self.calls.append("start")

    def healthy(self):
        return False


def v1_shutdown():
    print("\n=== V1. ashctl shutdown ===")
    from src.daemon.service import ControlServer
    rt = FakeRuntime(ask=True)
    r = ControlServer(rt).dispatch("shutdown")
    time.sleep(0.1)
    check("it asks the one shutdown to run, and does not stop() on its own", r.get("stopping") and rt.calls == ["request_shutdown"], str(rt.calls))
    rt = FakeRuntime(ask=False)
    ControlServer(rt).dispatch("shutdown")
    time.sleep(0.2)
    check("with no such hook it still stops (as before)", rt.calls == ["stop"], str(rt.calls))


def v2_watchdog():
    print("\n=== V2. The watchdog restarts the loop, not the whole runtime ===")
    from src.daemon.service import Watchdog

    class RT(FakeRuntime):
        def restart_loop(self, timeout=3.0):
            self.calls.append("restart_loop")
            wd._stop.set()
    rt = RT(ask=False)
    wd = Watchdog(rt, interval=0.01)
    wd._loop()
    check("an unhealthy loop is restarted with restart_loop(), no stop()", rt.calls == ["restart_loop"], str(rt.calls))

    from src.daemon.ambient import AmbientRuntime
    import tst_effects as te
    ar = te.rig()
    closed, started = [], []
    ar.periphery.close_all = lambda: closed.append(1)
    ar.start = lambda: started.append(1)
    ar.restart_loop(timeout=0.1)
    check("restart_loop on the real runtime closes no sensors and starts the loop again", not closed and started == [1], f"{closed} {started}")


if __name__ == "__main__":
    v1_shutdown()
    v2_watchdog()
    print("\n" + ("ALL PASS" if OK else "FAILED"))
    sys.exit(0 if OK else 1)
