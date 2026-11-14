"""
tst_controlauth.py -- the control socket answers only to a caller that can read the token.

The control socket on 127.0.0.1:8787 took any line from any local process: `shutdown`, `pause`, `say ...`, and `simulate mic
...` (which is read as the user speaking). A browser page can only send HTTP, which was refused by accident (an unknown command),
not on purpose. Now a line is `ASHTOKEN <token> <command>`, the token lives in state/control.token (which `ashctl` reads), a
web request is refused as one, and a non-loopback host in the config is ignored.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_controlauth.py
"""

import json
import os
import socket
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402
from src.daemon import service  # noqa: E402
from src.daemon.service import ControlServer, control_client, load_or_create_control_token  # noqa: E402

check = tw.check


def raw(port, line):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall((line + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode() or "{}")


def c1_token_file():
    print("\n=== The token ===")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "state", "control.token")
    t1 = load_or_create_control_token(p)
    check("a token is made, long and random", len(t1) >= 32 and all(c in "0123456789abcdef" for c in t1), t1)
    check("...and kept: the same one next time", load_or_create_control_token(p) == t1)
    check("...in a file that holds just it", open(p).read().strip() == t1)
    open(p, "w").write("short")
    check("a damaged (too short) token file is replaced", len(load_or_create_control_token(p)) >= 32)
    check("two installs do not share one", load_or_create_control_token(os.path.join(d, "x", "t")) != t1)


def c2_server():
    print("\n=== The server ===")
    rt = te.rig()
    srv = ControlServer(rt, "127.0.0.1", 0, token="s3cret-token-0123456789abcdef")
    srv.start()
    port = srv._sock.getsockname()[1]
    try:
        check("no token: refused", raw(port, "status").get("ok") is False and "token" in raw(port, "status").get("error", ""))
        check("a wrong token: refused", raw(port, "ASHTOKEN nope status").get("error") == "wrong control token")
        check("...even a prefix of the right one", raw(port, "ASHTOKEN s3cret status").get("error") == "wrong control token")
        got = raw(port, "ASHTOKEN s3cret-token-0123456789abcdef status")
        check("the right token: the command runs", got.get("ok") is True and "status" in got, str(got)[:100])
        check("a dangerous command with no token does not run", raw(port, "shutdown").get("ok") is False)
        check("`simulate mic` with no token does not reach the loop (it is read as the user speaking)",
              raw(port, "simulate mic hey ash what is the time").get("ok") is False and rt.stats["events"] == 0)
        for line in ("GET / HTTP/1.1", "POST /status HTTP/1.1", "OPTIONS * HTTP/1.1", "CONNECT x:1 HTTP/1.1"):
            r = raw(port, line)
            check(f"a web request is refused as one: {line.split()[0]}", r.get("error") == "this is not a web server", str(r))
        check("a web request that carries the token is refused too", raw(port, "POST /x HTTP/1.1 ASHTOKEN s3cret-token-0123456789abcdef").get("ok") is False)
        check("a token with odd bytes does not crash the server", raw(port, "ASHTOKEN café status").get("ok") is False
              and raw(port, "ASHTOKEN s3cret-token-0123456789abcdef status").get("ok") is True)
    finally:
        srv.stop()

    srv2 = ControlServer(rt, "127.0.0.1", 0, token=None)
    srv2.start()
    port2 = srv2._sock.getsockname()[1]
    try:
        check("with no token configured it works as before", raw(port2, "status").get("ok") is True)
        check("...and tolerates a client that sends one anyway", raw(port2, "ASHTOKEN whatever status").get("ok") is True)
    finally:
        srv2.stop()


def c3_client_and_host():
    print("\n=== The client, and where it listens ===")
    rt = te.rig()
    d = tempfile.mkdtemp()
    old = os.getcwd()
    os.chdir(d)
    try:
        tok = load_or_create_control_token()
        check("the token goes in state/control.token under the working folder", os.path.exists(os.path.join(d, "state", "control.token")))
        srv = ControlServer(rt, "127.0.0.1", 0, token=tok)
        srv.start()
        port = srv._sock.getsockname()[1]
        try:
            got = control_client("status", port=port)
            check("ashctl's client reads the token itself and is answered", got.get("ok") is True, str(got)[:80])
            bad = control_client("status", port=port, token="wrong")
            check("...and one given the wrong token is not", bad.get("ok") is False)
        finally:
            srv.stop()
    finally:
        os.chdir(old)
    for host in ("0.0.0.0", "192.168.1.5", "example.com", ""):
        s = ControlServer(rt, host, 0)
        check(f"a config host of {host!r} is not honoured: loopback only", s.host == "127.0.0.1", s.host)
    check("127.0.0.1, localhost and ::1 are", all(ControlServer(rt, h, 0).host == h for h in ("127.0.0.1", "localhost", "::1")))
    check("the default config requires the token", service.DEFAULT_CONFIG["control"]["require_token"] is True)


if __name__ == "__main__":
    c1_token_file()
    c2_server()
    c3_client_and_host()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
