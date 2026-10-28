"""
tst_face_editor.py -- the face editor's save/load server.

The editor writes files the robot loads, from a web request, so the things
worth testing are: what lands on disk is exactly what the robot reads, bad
input never reaches the disk, a web page on another site can't write, and the
Python interpolator the editor is checked against is the real one.
"""
import http.client
import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.face import Animation, BUILTINS, load_library  # noqa: E402
from src.face.editor_server import make_server, validate_animation  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_editor.html")
tmp = tempfile.mkdtemp()
adir = os.path.join(tmp, "animations")
srv = make_server(HTML, adir, port=0)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()


def call(method, path, body=None, headers=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": f"127.0.0.1:{port}"}
    h.update(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    if data is not None:
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    txt = r.read().decode()
    c.close()
    try:
        return r.status, json.loads(txt)
    except ValueError:
        return r.status, txt


def good(name="wiggle", **kw):
    d = {"schema": 1, "name": name, "duration": 1.0, "loop": False, "priority": 60,
         "interruptible": True, "blink_suppress": False, "hold_last": True, "tags": ["t"],
         "keyframes": [{"t": 0, "ease": "in_out"},
                       {"t": 0.5, "ease": "out", "both": {"h": 20}, "left": {"radius_tl": 2},
                        "face": {"tilt": 5}},
                       {"t": 1.0, "ease": "linear", "both": {"h": 36}}]}
    d.update(kw)
    return d


def files():
    return sorted(os.listdir(adir)) if os.path.isdir(adir) else []


# ---------------------------------------------------------------- basics
st, j = call("GET", "/api/ping")
check("ping", st == 200 and j["ok"])
st, page = call("GET", "/")
check("serves the editor", st == 200 and "A.S.H Face Editor" in page)
st, j = call("GET", "/api/library")
names = {a["name"]: a for a in j["animations"]}
check("library lists every builtin", set(BUILTINS) <= set(names) and
      all(a["source"] == "builtin" for a in names.values()))
check("no directory created by reading", not os.path.isdir(adir))
check("all builtins validate", all(not validate_animation(a.as_dict()) for a in BUILTINS.values()))
st, j = call("GET", "/api/animation/happy")
check("get builtin", st == 200 and j["name"] == "happy" and j["keyframes"])
check("get unknown is 404", call("GET", "/api/animation/nope")[0] == 404)

# ---------------------------------------------------------------- save
st, j = call("PUT", "/api/animation/wiggle", good())
check("save new", st == 200 and j["saved"] == "wiggle.json" and files() == ["wiggle.json"], str((st, j)))
lib = load_library(adir)
a = lib["wiggle"]
check("robot loader reads it", a.priority == 60 and a.tags == ["t"])
p = a.sample(0.5)
check("values survive", p.left.h == 20 and p.left.radius_tl == 2 and p.right.radius_tl is None
      and p.tilt == 5)
st, j = call("GET", "/api/library")
check("listed as user", {x["name"]: x["source"] for x in j["animations"]}["wiggle"] == "user")
st, j = call("GET", "/api/animation/wiggle")
check("get returns saved", st == 200 and j["priority"] == 60)
check("no temp files left", all(f.endswith(".json") for f in files()))

# override a builtin and revert
d = BUILTINS["happy"].as_dict()
d["priority"] = 77
st, j = call("PUT", "/api/animation/happy", d)
check("override builtin", st == 200)
check("loader prefers file", load_library(adir)["happy"].priority == 77)
st, j = call("GET", "/api/library")
check("listed as override", {x["name"]: x["source"] for x in j["animations"]}["happy"] == "override")
st, j = call("DELETE", "/api/animation/happy")
check("delete reverts to builtin", st == 200 and j["reverts_to_builtin"] and
      load_library(adir)["happy"].priority == BUILTINS["happy"].priority)
check("delete missing is 404", call("DELETE", "/api/animation/happy")[0] == 404)
check("delete leaves others", "wiggle.json" in files())

# a person's own file with another filename is replaced, not duplicated
with open(os.path.join(adir, "custom_name.json"), "w") as fh:
    json.dump(good("hand_made"), fh)
st, j = call("PUT", "/api/animation/hand_made", good("hand_made", priority=11))
check("replaces file that defines the name", st == 200 and j["saved"] == "custom_name.json"
      and not os.path.exists(os.path.join(adir, "hand_made.json")))

# ---------------------------------------------------------------- bad input
before = files()
bad_bodies = {
    "bad ease": good(keyframes=[{"t": 0, "ease": "wobble"}]),
    "string value": good(keyframes=[{"t": 0, "both": {"h": "20"}}]),
    "inf value": good(keyframes=[{"t": 0, "both": {"h": 1e999}}]),
    "unknown param": good(keyframes=[{"t": 0, "both": {"bogus": 1}}]),
    "unknown face param": good(keyframes=[{"t": 0, "face": {"left": 1}}]),
    "negative time": good(keyframes=[{"t": -1}]),
    "missing time": good(keyframes=[{"ease": "in"}]),
    "no keyframes": good(keyframes=[]),
    "duration short": good(duration=0.2),
    "duration zero": good(duration=0),
    "bad priority": good(priority="high"),
    "bad flag": good(loop="yes"),
    "bad tags": good(tags="x"),
    "future schema": good(schema=99),
    "too many keys": good(keyframes=[{"t": i * 0.001} for i in range(600)], duration=1),
}
for label, body in bad_bodies.items():
    st, j = call("PUT", "/api/animation/wiggle", body)
    check(f"rejects {label}", st == 422 and "errors" in j, str((st, j)))
st, j = call("PUT", "/api/animation/wiggle", good("Wiggle"))
check("rejects uppercase name", st == 422)
st, j = call("PUT", "/api/animation/other", good("wiggle"))
check("rejects name/URL mismatch", st == 422)
st, j = call("PUT", "/api/animation/wiggle", raw=b"{not json")
check("rejects non-JSON", st == 400)
st, j = call("PUT", "/api/animation/wiggle", raw=b"[]")
check("rejects non-object", st == 422)
check("nothing written by bad requests", files() == before)

for label, n in {"traversal": "..%2f..%2fevil", "dots": "..", "slash": "a/b",
                 "upper": "Happy", "empty": "", "long": "a" * 49, "space": "a%20b",
                 "dot-json": "x.json"}.items():
    st, _ = call("PUT", f"/api/animation/{n}", good())
    st2, _ = call("GET", f"/api/animation/{n}")
    st3, _ = call("DELETE", f"/api/animation/{n}")
    check(f"bad name refused ({label})", st in (400, 404) and st2 in (400, 404) and st3 in (400, 404),
          str((st, st2, st3)))
check("no file escaped the directory", files() == before and
      not any(f.startswith("evil") for f in os.listdir(tmp)))

big = json.dumps(good(tags=["x" * 30] * 20000)).encode()
st, _ = call("PUT", "/api/animation/wiggle", raw=big)
check("oversize body refused", st == 413)

# ---------------------------------------------------------------- who may write
st, _ = call("PUT", "/api/animation/wiggle", good(), headers={"Host": "evil.example"})
check("foreign Host refused", st == 403)
st, _ = call("PUT", "/api/animation/wiggle", good(), headers={"Host": "127.0.0.1.evil.example"})
check("lookalike Host refused", st == 403)
st, _ = call("PUT", "/api/animation/wiggle", good(), headers={"Origin": "https://evil.example"})
check("cross-site Origin refused", st == 403)
st, _ = call("PUT", "/api/animation/wiggle", good(),
             headers={"Origin": f"http://127.0.0.1:{port}"})
check("same-origin accepted", st == 200)
st, _ = call("PUT", "/api/animation/wiggle", good(),
             headers={"Origin": "http://localhost:9999"})
check("other local port refused", st == 403)
st, _ = call("GET", "/api/library", headers={"Origin": "https://evil.example"})
check("cross-site read refused", st == 403)

# ---------------------------------------------------------------- sample
anim = good()
times = [0, 0.1, 0.25, 0.5, 0.77, 1.0, 1.5]
st, j = call("POST", "/api/sample", {"animation": anim, "times": times})
ref = Animation.from_dict(anim)
check("sample matches the Python interpolator",
      st == 200 and all(j["poses"][i] == ref.sample(t).as_dict() for i, t in enumerate(times)))
check("sample rejects bad animation", call("POST", "/api/sample",
      {"animation": good(keyframes=[]), "times": [0]})[0] == 422)
check("sample rejects bad times", call("POST", "/api/sample",
      {"animation": anim, "times": ["a"]})[0] == 422)
check("sample caps times", call("POST", "/api/sample",
      {"animation": anim, "times": [0] * 1000})[0] == 422)
check("unknown route", call("GET", "/nothing")[0] == 404 and call("POST", "/x", {})[0] == 404)

lp = good("looper", loop=True)
st, j = call("POST", "/api/sample", {"animation": lp, "times": [0.25, 1.25]})
check("loop wraps in sample", j["poses"][0] == j["poses"][1])

srv.shutdown()
print("\nFAILED:", FAILS if FAILS else "none")
sys.exit(1 if FAILS else 0)
