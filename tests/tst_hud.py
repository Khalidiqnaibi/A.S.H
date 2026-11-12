"""
tst_hud.py -- the heads-up widget: a small always-on-top panel.

  H1  the page: served at /hud on the console server, with the console's own themes and the session token
  H2  what it shows: the status stream's cards, sensors, privacy, quiet -- plus what is playing and who is here
  H3  pinned on top (Windows): found by its title, then SetWindowPos(TOPMOST); gives up after a while
  H4  "show the heads-up display" / `ashctl hud`: opened through the opener ashd gives the runtime (a fake here --
      no window is opened by this test)

Run:  PYTHONIOENCODING=utf-8 python tests/tst_hud.py
"""

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402

check = tw.check


def h1_page():
    print("\n=== H1. The page ===")
    from src.ui import ConsoleServer, UIBus
    from src.ui.server import theme_css
    css = theme_css()
    check("the console's themes are found (all four)", all(f'[data-theme="{t}"]' in css for t in ("nebula", "blossom", "paper", "forest"))
          and "--accent" in css, css[:80])
    srv = ConsoleServer(UIBus(), port=0, theme="paper")
    html = srv.hud().decode("utf-8")
    check("served with the themes, the token and the theme filled in", "/*{{THEMES}}*/" not in html and "{{TOKEN}}" not in html
          and srv.token in html and 'data-theme="paper"' in html and '[data-theme="forest"]' in html)
    check("its title is what the window is pinned by", "<title>ASH HUD</title>" in html)
    check("it listens to the same stream and posts with the token", 'new EventSource("/events")' in html and '"X-ASH-Token": TOKEN' in html)
    if srv.start():
        try:
            port = srv.url.rstrip("/").rsplit(":", 1)[1]
            body = urllib.request.urlopen(f"http://127.0.0.1:{port}/hud", timeout=5).read().decode("utf-8")
            check("GET /hud on the running server", "<title>ASH HUD</title>" in body)
        finally:
            srv.stop()
    else:
        check("the console server started for the check", False)


def h2_status():
    print("\n=== H2. What it shows ===")
    import tst_effects as te
    from src.ui.hooks import _status
    rt = te.rig()
    rt._playing = {"title": "Clair de Lune", "artist": "Debussy"}
    st = _status(rt)
    check("what is playing", st["hud"]["playing"] == "Clair de Lune -- Debussy", str(st["hud"]))
    check("nobody listed without the camera's presence", st["hud"]["here"] == [])
    check("and the cards, sensors, privacy and quiet the console already gets", all(k in st for k in ("headsup", "sensors", "paused", "muted")))
    rt._playing = None
    check("nothing playing: empty", _status(rt)["hud"]["playing"] == "")


class FakeUser32:
    def __init__(self, appears_after=2):
        self.calls, self.n, self.appears_after, self.pos = [], 0, appears_after, None

    def FindWindowW(self, cls, title):
        self.n += 1
        self.calls.append(title)
        return 4242 if self.n >= self.appears_after else 0

    def SetWindowPos(self, hwnd, after, x, y, w, h, flags):
        self.pos = (hwnd, after, flags)
        return 1


def h3_pinned():
    print("\n=== H3. Kept on top ===")
    from src.ui.window import make_topmost, open_hud
    u = FakeUser32(appears_after=3)
    ok = make_topmost("ASH HUD", timeout_s=5, user32=u, sleep=lambda s: None)
    check("found by its title once drawn, then made topmost without moving or stealing focus",
          ok and u.calls == ["ASH HUD"] * 3 and u.pos == (4242, -1, 0x0001 | 0x0002 | 0x0010), str(u.pos))
    never = FakeUser32(appears_after=10 ** 6)
    check("no such window: gives up", make_topmost("ASH HUD", timeout_s=2, user32=never, sleep=lambda s: None) is False and never.n <= 6)
    opened, pinned = [], []
    how = open_hud("http://127.0.0.1:8788/", opener=lambda url, profile_dir=None, size=None: opened.append((url, size)) or "app",
                   pin=lambda title: pinned.append(title))
    import time as _t
    _t.sleep(0.2)
    check("opens /hud small, and pins it", how == "app" and opened == [("http://127.0.0.1:8788/hud", (340, 470))] and pinned == ["ASH HUD"],
          f"{opened} {pinned}")
    pinned.clear()
    how = open_hud("http://127.0.0.1:8788", opener=lambda url, profile_dir=None, size=None: "tab", pin=lambda t: pinned.append(t))
    _t.sleep(0.1)
    check("in a plain tab there is nothing to pin", how == "tab" and pinned == [])


def h4_asking():
    print("\n=== H4. Asking for it ===")
    import tst_effects as te
    rt = te.rig()
    asked = []
    rt.hud_opener = lambda: asked.append(1) or "app"
    n = len(rt.ash.calls)
    for said in ("Ash, show the heads-up display", "open the hud", "can you bring up the widget"):
        rt.say(said)
    check("opened by rule, three ways of asking, no model", len(asked) == 3 and len(rt.ash.calls) == n and "stays on top" in rt.out[-1], str(rt.out[-2:]))
    rt.say("show the hud", addressed=False)
    check("overheard: not opened", len(asked) == 3)
    rt.hud_opener = None
    rt.say("Ash, show the hud")
    check("no console: says why", "needs the console" in rt.out[-1], rt.out[-1])
    from src.daemon.service import ControlServer
    cs = ControlServer.__new__(ControlServer)
    cs.runtime = rt
    check("`ashctl hud` with no console: an error saying why", cs.dispatch("hud").get("ok") is False)
    rt.hud_opener = lambda: "app"
    check("`ashctl hud`", cs.dispatch("hud") == {"ok": True, "opened": "app"}, json.dumps(cs.dispatch("hud")))


if __name__ == "__main__":
    for fn in (h1_page, h2_status, h3_pinned, h4_asking):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
