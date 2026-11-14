"""
tst_legacyapp.py -- the older web app (app.py) is for this machine and keeps the voice here.

It listened on 0.0.0.0:5000 with Socket.IO origins "*" and no authentication, so anyone on the network -- and any web page open in
the user's browser -- could send it messages, which ran through ASH's tools and memory; and its voice path sent the recording to
Google (recognize_google) although ASH promises local-only speech recognition.

app.py cannot be imported without Flask-SocketIO and the whole of ASH, so this reads its source.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_legacyapp.py
"""

import ast
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check
SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), encoding="utf-8").read()
TREE = ast.parse(SRC)


def const_env_default(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets):
            for call in ast.walk(node.value):
                if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "get" and len(call.args) >= 2:
                    if isinstance(call.args[1], ast.Constant):
                        return call.args[1].value
    return None


def t1_network():
    print("\n=== Where it listens, and who it answers ===")
    check("it listens on loopback by default", const_env_default("APP_HOST") == "127.0.0.1", str(const_env_default("APP_HOST")))
    origins = const_env_default("APP_ORIGINS") or ""
    check("it accepts its own pages only (not '*')", "127.0.0.1:5000" in origins and "*" not in origins, origins)
    check("the Socket.IO server is given those origins", "cors_allowed_origins=APP_ORIGINS" in SRC)
    check("no wildcard origin is left anywhere", not re.search(r"cors_allowed_origins\s*=\s*[\"']\*[\"']", SRC))
    check("it runs on APP_HOST, not 0.0.0.0", "host=APP_HOST" in SRC and '"0.0.0.0"' not in SRC)


def t2_voice():
    print("\n=== The voice stays here ===")
    i = SRC.index("recognizer.recognize_google")
    guard = SRC.rfind("ASH_ALLOW_ONLINE_STT", 0, i)
    check("the upload to Google is behind a deliberate switch", guard != -1 and i - guard < 1200)
    check("...which is off unless set to 1", '"ASH_ALLOW_ONLINE_STT", "0") != "1"' in SRC)
    check("...and says why, in the app", "keeps your voice on this" in SRC)


if __name__ == "__main__":
    t1_network()
    t2_voice()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
