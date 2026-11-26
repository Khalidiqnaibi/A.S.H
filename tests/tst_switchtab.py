"""tst_switchtab.py -- "switch to Gemini's tab": the open tab found by name and handed to the extension to bring forward. Fake executors only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_switchtab.py"""
import json
import os
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402
from src.daemon.ambient import AmbientRuntime  # noqa: E402
from src.senses.browsertabs import BrowserTabSensor  # noqa: E402

check = tw.check
TABS = [{"id": 11, "url": "https://gemini.google.com/app/1", "title": "Custom TPU Prototyping - Google Gemini", "active": True, "audible": False, "incognito": False},
        {"id": 12, "url": "https://www.youtube.com/watch?v=x", "title": "lo-fi beats - YouTube", "active": False, "audible": True, "incognito": False},
        {"id": 13, "url": "https://example.com/secret", "title": "Private stuff", "active": False, "audible": False, "incognito": True}]


def main():
    print("\n=== S1. What is asked ===")
    T = AmbientRuntime._TAB
    for t, w in (("Switch to Gemini's tab.", "Gemini"), ("go to the YouTube tab", "YouTube"), ("Ash, switch back to my gmail tab please", "gmail"), ("open the Gemini tab", "Gemini")):
        m = T.match(t)
        check(f"{t!r} -> {w!r}", m is not None and (m.group("w") or m.group("w2")) == w, str(m and m.groupdict()))
    for t in ("switch to Word", "open a new tab", "switch tabs", "what tab am I on"):
        m = T.match(t)
        check(f"{t!r} is not a tab by name", m is None or (m.group("w") or m.group("w2") or "").lower() in ("a new",), str(m and m.groupdict()))

    print("\n=== S2. The sensor ===")
    tok = "t" * 24
    s = BrowserTabSensor(enabled=True, token=tok, port=0)
    s.submit({"focused": True, "tabs": TABS})
    s.read()
    check("tabs are found by a word of the title or the site", (s.find_tab("Gemini") or {}).get("id") == 11 and (s.find_tab("youtube") or {}).get("id") == 12)
    check("a private tab is never found", s.find_tab("private stuff") is None)
    check("nothing there: None", s.find_tab("gmail") is None)
    check("the listener starts", s.start_listener())

    def post():
        r = urllib.request.Request(f"http://127.0.0.1:{s.port}/tabs", data=json.dumps({"focused": True, "tabs": TABS}).encode(),
                                   headers={"Content-Type": "application/json", "X-ASH-Token": tok, "Origin": "chrome-extension://abc"}, method="POST")
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, (resp.read() or b"").decode()
    check("no command: 204, as before", post()[0] == 204)
    s.request_activate(12)
    st, body = post()
    check("a switch asked for is handed over in the reply", st == 200 and json.loads(body) == {"activate": 12}, f"{st} {body}")
    check("once only", post()[0] == 204)
    s.stop_listener()
    js = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extension", "ash-tabs", "background.js"), encoding="utf-8").read()
    check("the extension acts on it, and on nothing else", "chrome.tabs.update(cmd.activate, { active: true })" in js and "Number.isInteger(cmd.activate)" in js)

    print("\n=== S3. In the runtime ===")
    rt = tm.rig()
    rt.desktop = td.drig()
    rt.desktop.ex_.procs.add("opera.exe")
    s2 = BrowserTabSensor(enabled=True)
    s2.submit({"focused": True, "tabs": TABS})
    s2.read()
    real = rt.periphery.sensors

    class Sensors(dict):
        def get(self, k, d=None):
            return s2 if k == "browser_tabs" else real.get(k, d)
    rt.periphery.sensors = Sensors(real)
    used, said = rt.say("Switch to Gemini's tab.")
    check("said, and the switch is waiting for the extension", not used and said and said[-1] == "Switching to the Gemini tab." and s2.take_command() == {"activate": 11}, f"{used} {said}")
    check("the browser was brought forward (so the extension reports at once)", rt.desktop.ex_.focused == ["opera.exe"], str(rt.desktop.ex_.focused))
    n = len(rt.desktop.ex_.focused)
    used, said = rt.say("open a new tab")
    check("'open a new tab' is not taken as a tab called 'a new'", not (said and "a new tab open" in said[-1]), str(said))
    used, said = rt.say("switch to the gmail tab")
    check("not open: said so", said and said[-1] == "I don't see a gmail tab open.", str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
