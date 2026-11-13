"""
tst_vision_page.py -- the console's /vision page: what the camera is doing, why it is off if it is, the latest detections
as a table with confidence bars, per-label counts, the last look drawn as boxes (SVG from numbers, never a photo), and a
"look now" button. The server runs on a loopback port with a fake camera and fake detectors; no real camera is touched.

  P1  the page: served with the console's themes, token and theme; light and dark; mobile; no unsafe HTML
  P2  /api/vision: the numbers behind it, JSON-clean, labels and boxes only
  P3  "look now": through the token-protected control route; privacy, kids and no-camera say why
  P4  the checks that keep another page from driving it

Run:  PYTHONIOENCODING=utf-8 python tests/tst_vision_page.py
"""

import http.client
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_vision as tv  # noqa: E402
import tst_vision_camera as vc  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
HTML_PATH = os.path.join(HERE, "src", "ui", "static", "vision.html")


class Page:
    def __init__(self, srv):
        self.port = int(srv.url.rstrip("/").rsplit(":", 1)[1])
        self.srv = srv

    def req(self, method, path, body=None, host=None, token=True):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        if token:
            h["X-ASH-Token"] = self.srv.token
        data = json.dumps(body).encode() if body is not None else None
        if data:
            h["Content-Type"] = "application/json"
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        return r.status, r.getheader("Content-Type") or "", raw


def served(**kw):
    from src.ui import ConsoleServer, UIBus, attach_ui
    rt = vc.rt_with(**kw)
    bus = UIBus()
    on_say, on_control, extra = attach_ui(rt, bus)
    srv = ConsoleServer(bus, port=0, theme="paper", on_say=on_say, on_control=on_control, state_extra=extra)
    ok = srv.start()
    return rt, srv, ok


def p1_page():
    print("\n=== P1. The page ===")
    html_src = open(HTML_PATH, encoding="utf-8").read()
    check("the file exists, with a name for a title", "<title>ASH Camera</title>" in html_src)
    check("mobile: a viewport and a single column on a narrow screen", 'name="viewport"' in html_src and "max-width: 720px" in html_src and "grid-template-columns: minmax(0, 1fr)" in html_src)
    check("it uses the console's theme tokens (so light and dark come with the theme)", all(t in html_src for t in ("var(--text", "var(--accent", "var(--line", "var(--warn", "var(--bad")))
    check("color-scheme follows the theme: dark by default, light for paper and blossom", "color-scheme: dark" in html_src and '[data-theme="paper"], [data-theme="blossom"] { color-scheme: light; }' in html_src)
    check("no innerHTML anywhere: a label from the camera can never be markup", "innerHTML" not in html_src and "document.write" not in html_src and "eval(" not in html_src)
    check("it auto-refreshes, and not while hidden", "setInterval(" in html_src and "document.hidden" in html_src and "/api/vision" in html_src)
    check("it has the look-now button, posting with the token", 'id="look"' in html_src and '"X-ASH-Token": TOKEN' in html_src and 'cmd: "vision_look"' in html_src)
    check("the frame is an SVG drawn from numbers; there is no <img>, <canvas> or <video>", 'id="frame"' in html_src and not re.search(r"<(img|canvas|video)\b", html_src))
    check("it shows a table with confidence bars, per-label counts and the camera's status with its reason",
          ('class="bar' in html_src or 'class: "bar' in html_src) and 'id="counts"' in html_src and 'id="det"' in html_src and 'id="why"' in html_src)
    check("labelled regions for a screen reader", html_src.count("aria-labelledby") >= 5 and 'role="status"' in html_src and 'role="img"' in html_src)
    rt, srv, ok = served(things=[tv.thing("watch", .8, vc.MID)])
    check("the server started", ok)
    if not ok:
        return
    try:
        pg = Page(srv)
        st, ctype, raw = pg.req("GET", "/vision", token=False)
        page = raw.decode("utf-8")
        check("GET /vision: 200 html", st == 200 and ctype.startswith("text/html"))
        check("the themes, token and theme are filled in", "/*{{THEMES}}*/" not in page and "{{TOKEN}}" not in page and srv.token in page
              and 'data-theme="paper"' in page and all(f'[data-theme="{t}"]' in page for t in ("nebula", "blossom", "paper", "forest")))
        check("also at /vision.html", pg.req("GET", "/vision.html", token=False)[0] == 200)
        check("the console and hud still work", pg.req("GET", "/", token=False)[0] == 200 and pg.req("GET", "/hud", token=False)[0] == 200)
        check("a made-up path is still 404", pg.req("GET", "/vision/nothing", token=False)[0] == 404)
        scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
        check("one inline script, balanced braces", len(scripts) == 1 and scripts[0].count("{") == scripts[0].count("}") and scripts[0].count("(") == scripts[0].count(")"))
    finally:
        srv.stop()


def p2_json():
    print("\n=== P2. /api/vision ===")
    rt, srv, ok = served(things=[tv.thing("watch", .8, vc.MID), tv.thing("cup", .45, (20, 20, 60, 60)), tv.thing("person", .9, (0, 0, 640, 480))],
                         faces=[tv.face()], vectors=[tv.vec(2)])
    if not ok:
        check("server started", False)
        return
    try:
        pg = Page(srv)
        st, ctype, raw = pg.req("GET", "/api/vision", token=False)
        j = json.loads(raw)
        v = j.get("vision") or {}
        check("200 JSON with ok and a vision block", st == 200 and ctype.startswith("application/json") and j.get("ok") and v)
        check("before any look: status, nothing seen, the reason the camera is 'ready'", v["camera"]["state"] == "ready" and v["detections"] == [] and v["scene"] is None
              and v["counts"] == [] and "until you ask" in v["camera"]["why"], str(v["camera"]))
        check("it says how it names things", v["backend"]["objects"] is True and v["backend"]["faces"] is True and v["backend"]["min_conf"] == 0.35 and v["backend"]["sure_conf"] == 0.6)
        rt.say("Ash, look around")
        v = json.loads(pg.req("GET", "/api/vision", token=False)[2])["vision"]
        labels = {(d["kind"], d["label"]) for d in v["detections"]}
        check("after a look: the detections, newest first, with kind, label, confidence, box and source",
              {("object", "watch"), ("object", "cup"), ("person", "person"), ("face", "face")} == labels and v["detections"][0]["t"] >= v["detections"][-1]["t"]
              and all(set(d) == {"t", "kind", "label", "conf", "box", "source"} for d in v["detections"]), str(labels))
        check("per-label counts and average confidence", any(c["label"] == "watch" and c["count"] == 1 and c["avg"] == 0.8 for c in v["counts"]), str(v["counts"]))
        sc = v["scene"]
        check("the last look as boxes: the frame size and one box per thing and face", sc and (sc["w"], sc["h"]) == (640, 480) and len(sc["things"]) == 3 and len(sc["faces"]) == 1
              and all(len(t["box"]) == 4 for t in sc["things"]))
        check("a box is four plain numbers; the whole payload is numbers and short labels (no pixels, no names)",
              all(isinstance(n, int) for t in sc["things"] for n in t["box"]) and "Khalid" not in json.dumps(v) and len(json.dumps(v)) < 20000)
        check("presence and watching blocks are there", v["presence"] == {"on": False, "here": [], "stranger": False} and v["watching"]["react"] is True)
        check("the last look is reported", v["last_look"]["why"] == "" and v["last_look"]["say"].startswith("I see"))
        for i in range(400):
            rt.log.add("object", f"x{i}", .5, None, "asked")
        v = json.loads(pg.req("GET", "/api/vision", token=False)[2])["vision"]
        check("only the latest 60 are sent, but the ring holds 300", len(v["detections"]) == 60 and v["kept"] == 300)
        rt.log.add("object", "<script>alert(1)</script>", .9, None, "asked")
        raw = pg.req("GET", "/api/vision", token=False)[2]
        check("a hostile label is just a string in JSON (the page uses textContent)", json.loads(raw)["vision"]["detections"][0]["label"] == "<script>alert(1)</script>")
        rt.privacy.pause("t")
        v = json.loads(pg.req("GET", "/api/vision", token=False)[2])["vision"]
        check("privacy mode: the page says so and why", v["camera"]["state"] == "privacy" and "Privacy mode" in v["camera"]["why"])
        rt.privacy.resume()
        rt.looking.cfg.enabled = False
        v = json.loads(pg.req("GET", "/api/vision", token=False)[2])["vision"]
        check("switched off: the page says so and which setting", v["camera"]["state"] == "off" and "vision.enabled" in v["camera"]["why"])
    finally:
        srv.stop()
    from src.ui import ConsoleServer, UIBus
    bare = ConsoleServer(UIBus(), port=0)
    if bare.start():
        try:
            st, _c, raw = Page(bare).req("GET", "/api/vision", token=False)
            check("a console with no ASH behind it: 503, not a crash", st == 503 and json.loads(raw)["ok"] is False)
        finally:
            bare.stop()
    import tst_effects as te
    from src.ui import attach_ui
    plain = te.rig()
    _s, ctl, _e = attach_ui(plain, UIBus())
    check("vision off in the runtime: the endpoint says so", ctl("vision_info", "") == {"ok": False, "error": "vision is not running"})


def p3_look_now():
    print("\n=== P3. 'Look now' ===")
    rt, srv, ok = served(things=[tv.thing("watch", .8, vc.MID)])
    if not ok:
        check("server started", False)
        return
    try:
        pg = Page(srv)
        n, k = len(rt.ash.calls), len(rt.grabs)
        st, _c, raw = pg.req("POST", "/api/control", {"cmd": "vision_look", "arg": ""})
        j = json.loads(raw)
        check("one explicit look: one frame, the answer, the snapshot", st == 200 and j["ok"] and len(rt.grabs) == k + 1 and j["say"] == "I see a person and a watch."
              and j["vision"]["scene"]["things"] and len(rt.ash.calls) == n, str(j)[:200])
        check("...and it is in the detections", any(d["label"] == "watch" for d in j["vision"]["detections"]))
        check("it keeps nothing of the picture on disk", os.listdir(rt.dir) == ["detections.jsonl"] or sorted(os.listdir(rt.dir)) == ["detections.jsonl"])
        rt.privacy.pause("t")
        k = len(rt.grabs)
        j = json.loads(pg.req("POST", "/api/control", {"cmd": "vision_look", "arg": ""})[2])
        check("privacy mode: no frame, and the line says why", not j["ok"] and j["why"] == "privacy" and "Privacy mode is on" in j["say"] and len(rt.grabs) == k, str(j)[:160])
        rt.privacy.resume()
        rt.looking.eye._grab = lambda: None
        j = json.loads(pg.req("POST", "/api/control", {"cmd": "vision_look", "arg": ""})[2])
        check("no camera: not ok, the reason in one line, and the page's state follows", not j["ok"] and j["why"] == "no camera" and "no camera I can open" in j["say"]
              and j["vision"]["camera"]["state"] == "unavailable", str(j)[:200])
        rt.looking.eye._grab = lambda: vc.tv.frame()
    finally:
        srv.stop()
    rt, srv, ok = served(things=[tv.thing("watch", .8, vc.MID)], kids=True)
    try:
        rt.say("Ash, kids mode for 20 minutes")
        k = len(rt.grabs)
        j = json.loads(Page(srv).req("POST", "/api/control", {"cmd": "vision_look", "arg": ""})[2])
        check("kids mode: the button does not open the camera", not j["ok"] and j["why"] == "kids" and "Kids mode" in j["say"] and len(rt.grabs) == k, str(j)[:160])
    finally:
        srv.stop()


def p4_guards():
    print("\n=== P4. Only this computer, and only this page ===")
    rt, srv, ok = served(things=[tv.thing("watch", .8, vc.MID)])
    if not ok:
        check("server started", False)
        return
    try:
        pg = Page(srv)
        k = len(rt.grabs)
        check("'look now' without the token: 403, no frame", pg.req("POST", "/api/control", {"cmd": "vision_look"}, token=False)[0] == 403 and len(rt.grabs) == k)
        c = http.client.HTTPConnection("127.0.0.1", pg.port, timeout=10)
        c.request("POST", "/api/control", body=b'{"cmd":"vision_look"}', headers={"Host": f"127.0.0.1:{pg.port}", "X-ASH-Token": "wrong"})
        check("a wrong token: 403", c.getresponse().status == 403 and len(rt.grabs) == k)
        check("a foreign Host header (DNS rebinding): 403 for the page and the data", pg.req("GET", "/vision", host="evil.example", token=False)[0] == 403
              and pg.req("GET", "/api/vision", host="evil.example", token=False)[0] == 403)
        check("the data is labels: nothing in the payload could be a picture", b"\xff\xd8" not in pg.req("GET", "/api/vision", token=False)[2])
        src = open(os.path.join(HERE, "src", "ui", "server.py"), encoding="utf-8").read()
        check("the server binds loopback by default", 'host: str = "127.0.0.1"' in src)
    finally:
        srv.stop()


if __name__ == "__main__":
    for fn in (p1_page, p2_json, p3_look_now, p4_guards):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
