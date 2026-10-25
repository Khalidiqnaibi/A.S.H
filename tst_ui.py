"""
tst_ui.py -- the console window: what it shows, and what it sends back.

  U1  UIBus: history, sticky cues, lossy poses, records forced through a full queue
  U2  ConsoleServer over real HTTP: page + token, loopback-only Host check,
      token check on every POST, say/control/theme, the SSE stream (hello,
      history replay, live records, poses), theme persistence
  U3  Hooks on the real AmbientRuntime: your words labelled typed/voice, the
      thinking cue around every response, every ASH line labelled answer /
      remark / offer / notice, offers with quick replies, the window reopened
      for a message when closed (and not more often than allowed)
  U4  The loop back: a quick reply and the dismiss button teach the social
      learner exactly as a spoken answer does
  U5  Actions: tools from the brain's trace, vetoes, actuators (speech cue,
      launch success/failure), notify left alone, stale traces ignored
  U6  The daemon's sink skips the notification while the window is open
  U7  The page: tokens substituted, message text never written as HTML,
      every record kind the hooks send is handled

Run:  python tst_ui.py
"""

import http.client
import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_social as ts  # noqa: E402  (reuses its fake brain and runtime rig)
from src.senses.base import SensorEvent  # noqa: E402
from src.ui import ConsoleServer, UIBus, attach_ui  # noqa: E402
from src.ui.hooks import QUICK_REPLIES  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


def kinds(bus, kind):
    return [r for r in bus.history() if r["kind"] == kind]


# ----------------------------------------------------------------------
def u1_bus():
    print("\nU1  UIBus")
    bus = UIBus(history=5)
    for i in range(8):
        bus.publish("ash", text=str(i))
    h = bus.history()
    check("history is bounded and keeps the newest", [r["text"] for r in h] == list("34567"))
    check("records are numbered in order", [r["id"] for r in h] == [4, 5, 6, 7, 8])
    check("history(after) returns only newer records", [r["id"] for r in bus.history(6)] == [7, 8])

    bus.cue("thinking", True, label="answering you")
    check("an 'on' cue is remembered for late windows", "thinking" in bus.state)
    bus.cue("thinking", False)
    check("an 'off' cue is forgotten", "thinking" not in bus.state)

    q = bus.subscribe(maxsize=3)
    check("subscribers are counted", bus.clients == 1)
    for i in range(10):
        bus.publish_pose({"look_x": i})
    check("poses never overfill a slow window", q.qsize() == 3)
    check("poses are not kept in history", all(r["kind"] != "pose" for r in bus.history()))
    rec = bus.publish("ash", text="important")
    got = [q.get_nowait() for _ in range(q.qsize())]
    check("a record still gets through a queue full of poses", rec in got)
    bus.unsubscribe(q)
    check("unsubscribing removes the window", bus.clients == 0)

    q = bus.subscribe(maxsize=3)
    line = bus.publish("ash", text="do not lose me")
    for i in range(20):
        bus.publish_pose({"look_x": i})
    got = [q.get_nowait() for _ in range(q.qsize())]
    check("a flood of poses never pushes a chat line out", line in got)
    bus.unsubscribe(q)


# ----------------------------------------------------------------------
class Client:
    def __init__(self, srv):
        self.srv = srv

    def req(self, method, path, body=None, host=None, token=True):
        c = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=5)
        headers = {"Host": host or f"127.0.0.1:{self.srv.port}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if token:
            headers["X-ASH-Token"] = self.srv.token if token is True else token
        c.request(method, path, body=data, headers=headers)
        r = c.getresponse()
        raw = r.read()
        c.close()
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw.decode("utf-8", "replace")

    def stream(self, n, timeout=5.0):
        """Read the first n SSE messages."""
        c = http.client.HTTPConnection("127.0.0.1", self.srv.port, timeout=timeout)
        c.request("GET", "/events", headers={"Host": f"127.0.0.1:{self.srv.port}"})
        r = c.getresponse()
        out, buf = [], b""
        deadline = time.time() + timeout
        while len(out) < n and time.time() < deadline:
            line = r.fp.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                out.append(json.loads(line[6:]))
        c.close()
        return out


def u2_server():
    print("\nU2  ConsoleServer")
    tmp = tempfile.mkdtemp()
    prefs = os.path.join(tmp, "ui.json")
    bus = UIBus()
    said, ctl = [], []
    srv = ConsoleServer(bus, port=0, theme="blossom", prefs_path=prefs,
                        on_say=said.append,
                        on_control=lambda c, a: ctl.append((c, a)) or {"ok": True})
    check("server starts on a free port", srv.start() and srv.port > 0)
    cl = Client(srv)

    code, page = cl.req("GET", "/")
    check("the page is served", code == 200 and "<canvas" in page)
    check("the session token is written into the page", srv.token in page and "{{TOKEN}}" not in page)
    check("the saved theme is applied before first paint", 'data-theme="blossom"' in page)

    code, _ = cl.req("GET", "/", host="evil.example:80")
    check("a non-loopback Host is refused (DNS rebinding)", code == 403)
    code, _ = cl.req("POST", "/api/say", {"text": "hi"}, token=False)
    check("a POST without the token is refused", code == 403 and not said)
    code, _ = cl.req("POST", "/api/say", {"text": "hi"}, token="guess")
    check("a POST with a wrong token is refused", code == 403 and not said)

    code, r = cl.req("POST", "/api/say", {"text": "  what time is it  "})
    check("typing reaches ASH, trimmed", code == 200 and said == ["what time is it"])
    code, r = cl.req("POST", "/api/say", {"text": "   "})
    check("an empty message is rejected", code == 400 and len(said) == 1)
    code, r = cl.req("POST", "/api/control", {"cmd": "mute", "arg": ""})
    check("controls reach the runtime", r.get("ok") and ctl == [("mute", "")])

    code, r = cl.req("POST", "/api/theme", {"theme": "nebula"})
    check("the theme can be changed", r.get("ok") and srv.theme == "nebula")
    check("the theme is saved", json.load(open(prefs)) == {"theme": "nebula"})
    code, r = cl.req("POST", "/api/theme", {"theme": "neon-vomit"})
    check("an unknown theme is rejected", code == 400 and srv.theme == "nebula")
    again = ConsoleServer(UIBus(), port=0, theme="paper", prefs_path=prefs)
    check("the saved theme wins over the configured one on restart", again.theme == "nebula")
    with open(prefs, "w") as f:
        f.write("{not json")
    check("a corrupt prefs file is ignored", ConsoleServer(UIBus(), port=0, theme="forest",
                                                           prefs_path=prefs).theme == "forest")

    code, st = cl.req("GET", "/api/state")
    check("state reports theme and themes", st.get("theme") == "nebula" and "blossom" in st.get("themes", []))

    bus.publish("user", text="earlier", channel="typed")
    bus.cue("thinking", True, label="answering you")
    got = []
    t = threading.Thread(target=lambda: got.extend(cl.stream(6)))
    t.start()
    for _ in range(50):
        if bus.clients:
            break
        time.sleep(0.02)
    bus.publish_pose({"look_x": 3, "anim": "happy"})
    bus.publish("ash", text="live line", role="answer")
    t.join(6)
    k = [m["kind"] for m in got]
    check("the stream opens with hello", k[:1] == ["hello"], str(k))
    hello = got[0] if got else {}
    check("hello carries boot id, token and sticky cues",
          hello.get("boot") == srv.boot and hello.get("token") == srv.token
          and "thinking" in hello.get("cues", {}))
    check("history is replayed before live messages",
          any(m.get("text") == "earlier" for m in got[1:4]))
    check("live poses arrive", any(m["kind"] == "pose" and m.get("anim") == "happy" for m in got))
    check("live records arrive", any(m.get("text") == "live line" for m in got))
    for _ in range(200):                              # keepalive is every 2s
        if not bus.clients:
            break
        time.sleep(0.02)
    check("a closed window unsubscribes", bus.clients == 0)
    srv.stop()


# ----------------------------------------------------------------------
def wired(**cfg):
    h = ts.rig(**cfg)
    bus = UIBus()
    opened = []
    on_say, on_control, state = attach_ui(h.rt, bus, opener=lambda: opened.append(1),
                                          status_every_s=3600)
    h.bus, h.opened, h.on_say, h.on_control, h.state = bus, opened, on_say, on_control, state
    return h


def u3_hooks():
    print("\nU3  Hooks on the real runtime")
    h = wired()
    h.on_say("what time is it")
    h.rt._tick()
    users = kinds(h.bus, "user")
    check("typed words appear as yours, labelled typed",
          users and users[-1]["text"] == "what time is it" and users[-1]["channel"] == "typed")
    ash = kinds(h.bus, "ash")
    check("ASH's reply appears, labelled as an answer", ash and ash[-1]["role"] == "answer"
          and ash[-1]["text"] == "Sure.")
    cues = [(r["cue"], r["on"]) for r in kinds(h.bus, "cue")]
    check("a thinking cue wraps the response", cues[:2] == [("thinking", True), ("thinking", False)], str(cues))
    first = kinds(h.bus, "cue")[0]
    check("the cue says what ASH is thinking about", first.get("label") == "answering you")
    check("the thinking cue is cleared afterwards", "thinking" not in h.bus.state)
    seq = [(r["kind"], r.get("cue"), r.get("on")) for r in h.bus.history() if r["kind"] in ("cue", "ash")]
    check("thinking ends when the reply arrives, not when the cycle's chores do",
          seq[:3] == [("cue", "thinking", True), ("cue", "thinking", False), ("ash", None, None)], str(seq[:3]))
    check("and is turned off only once", [c for c in seq if c[1] == "thinking"].count(("cue", "thinking", False)) == 1)
    check("the runtime's own sink still ran", h.out and h.out[-1][0] == "Sure.")

    h.say("hey ash are you there")                      # the rig's mic
    check("spoken words are labelled voice", kinds(h.bus, "user")[-1]["channel"] == "voice")

    n = len(kinds(h.bus, "ash"))
    offers = ts.free_moment(h)
    new = kinds(h.bus, "ash")[n:]
    check("a free moment produces an offer in the window", offers and new and new[-1]["role"] == "offer",
          str([r.get("role") for r in new]))
    if new:
        check("the offer names its kind", new[-1]["offer"] == "free_moment")
        check("the offer carries quick replies", new[-1]["replies"] == QUICK_REPLIES["free_moment"])
        check("the offer says why ASH spoke", bool(new[-1].get("why")))

    check("the window was opened for ASH's lines while closed", len(h.opened) == 1,
          f"opened {len(h.opened)}x")
    q = h.bus.subscribe()
    h.on_say("again")
    h.rt._tick()
    check("an open window is not reopened", len(h.opened) == 1)
    h.bus.unsubscribe(q)

    # Remark vs notice.
    h2 = wired()
    h2.rt.response_sink("Your build failed.", {"addressed": False, "source": "screen",
                                               "attention": {"kind": "screen", "reasons": ["urgent"]},
                                               "pathway": "slow"})
    h2.rt.response_sink("My language model isn't responding.", {"addressed": True, "degraded": True,
                                                                "attention": {}, "pathway": "degraded"})
    roles = [r["role"] for r in kinds(h2.bus, "ash")]
    check("a volunteered line is a remark, a model outage a notice", roles == ["remark", "notice"], str(roles))
    check("a remark has no quick replies", kinds(h2.bus, "ash")[0]["replies"] == [])
    check("reopening is rate-limited", len(h2.opened) == 1)

    h3 = wired()
    h3.rt.response_sink("x", {"addressed": True, "attention": {}})
    for c in ("pause", "mute"):
        h3.on_control(c, "")
    st = h3.state()["status"]
    check("pause and mute reach the runtime", st["paused"] and st["muted"] and h3.rt.privacy.paused)
    check("each control is confirmed in the window",
          [r["text"].split(" ")[0] for r in kinds(h3.bus, "system")] == ["Paused", "Muted"])
    for c in ("resume", "unmute"):
        h3.on_control(c, "")
    st = h3.state()["status"]
    check("and are undone", not st["paused"] and not st["muted"])
    check("an unknown command is refused", h3.on_control("format_disk", "")["ok"] is False)
    check("a missing face is reported, not crashed on", h3.on_control("face", "happy")["ok"] is False)
    check("presence without a face is harmless", h3.on_control("presence", "")["ok"] is True)

    class Bridge:
        def __init__(self):
            self.away = []
            self.player = type("P", (), {"on_frame": None, "current": None})()

        def on_user_away(self, away):
            self.away.append(away)

    h4 = ts.rig()
    h4.rt.face = Bridge()
    _, ctl4, _ = attach_ui(h4.rt, UIBus(), status_every_s=3600)
    h4.rt.face.on_user_away(True)
    check("with nobody at the console, away makes ASH sleepy", h4.rt.face.away[-1] is True)
    ctl4("presence", "")
    check("using the console wakes it", h4.rt.face.away[-1] is False)
    h4.rt.face.on_user_away(True)
    check("and the system idle timer cannot put it back to sleep while you read",
          h4.rt.face.away[-1] is False)


# ----------------------------------------------------------------------
def u4_learning():
    print("\nU4  Quick replies and dismiss teach the learner")
    h = wired()
    ts.free_moment(h)
    before = dict(h.rt.reactions.counts)
    h.on_say("No thanks")
    h.rt._tick()
    after = h.rt.reactions.counts
    check("'No thanks' from a button counts as a rejection",
          after.get("rejected", 0) == before.get("rejected", 0) + 1, str(after))
    check("and ASH answers it without the model",
          kinds(h.bus, "ash")[-1]["text"] == "Okay, I'll stay out of your way.")

    h = wired()
    ts.free_moment(h)
    h.on_say("Later")
    h.rt._tick()
    check("'Later' counts as deferred", h.rt.reactions.counts.get("deferred", 0) == 1)

    h = wired()
    ts.free_moment(h)
    h.on_say("Sure, let's chat")
    h.rt._tick()
    check("'Sure, let's chat' counts as accepted", h.rt.reactions.counts.get("accepted", 0) == 1)

    h = wired()
    ts.free_moment(h)
    n_ash = len(kinds(h.bus, "ash"))
    r = h.on_control("dismiss", "")
    h.rt._tick()
    check("dismiss is accepted", r["ok"])
    check("dismiss counts as dismissed", h.rt.reactions.counts.get("dismissed", 0) == 1,
          str(h.rt.reactions.counts))
    check("dismiss does not make ASH talk", len(kinds(h.bus, "ash")) == n_ash)
    check("dismiss is shown quietly", kinds(h.bus, "user")[-1].get("quiet") is True)


# ----------------------------------------------------------------------
class Act:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = []

    def act(self, args):
        self.calls.append(args)
        return self.result


def u5_actions():
    print("\nU5  Actions")
    h = ts.rig()
    speak, launch, bad, notify = (Act("speak", {"ok": True}), Act("launch", {"ok": True}),
                                  Act("shell", {"ok": False, "error": "not allowed"}), Act("notify", {"ok": True}))
    for a in (speak, launch, bad, notify):
        h.rt.periphery.actuators[a.name] = a
    orig_notify = notify.act
    bus = UIBus()
    attach_ui(h.rt, bus, status_every_s=3600)

    import src.ui.hooks as hooks
    hooks.SPEECH_CHARS_PER_S = 1000.0                 # make the speech timer short
    speak.act("hello there")
    cue = [(r["cue"], r["on"]) for r in kinds(bus, "cue")]
    check("speaking turns the speaking cue on", cue == [("speaking", True)])
    check("the cue says roughly how long", kinds(bus, "cue")[0]["seconds"] >= 1.2)
    check("the actuator really ran", speak.calls == ["hello there"])
    time.sleep(1.4)
    check("and the cue turns itself off", ("speaking", False) in [(r["cue"], r["on"]) for r in kinds(bus, "cue")])

    launch.act("notepad")
    bad.act("rm -rf /")
    acts = kinds(bus, "action")
    check("a hardware action is shown with what it did",
          acts[0]["name"] == "launch" and acts[0]["status"] == "done" and acts[0]["detail"] == "notepad")
    check("a failed action is shown with why", acts[1]["status"] == "failed" and acts[1]["detail"] == "not allowed")
    check("notify is not wrapped (the window is the notification)", notify.act == orig_notify)

    silent = Act("speak", {"ok": False, "error": "no TTS engine attached"})
    h2 = ts.rig()
    h2.rt.periphery.actuators["speak"] = silent
    b2 = UIBus()
    attach_ui(h2.rt, b2, status_every_s=3600)
    silent.act("hi")
    check("speech that fails does not leave the speaking cue on", "speaking" not in b2.state)

    # Tool use and vetoes come from the brain's trace.
    h3 = wired()
    trace = {}
    h3.ash.explain_last = lambda: trace
    trace.update(ts=time.time() + 1, pathway="fast",
                 verdict={"chosen": {"tool_name": "get_time", "args": None}, "vetoed_by": []},
                 tool_output={"ok": True})
    h3.on_say("time?")
    h3.rt._tick()
    a = kinds(h3.bus, "action")
    check("a tool the brain used is shown", a and a[-1]["name"] == "get_time" and a[-1]["status"] == "done")
    order = [r["kind"] for r in h3.bus.history() if r["kind"] in ("action", "ash")]
    check("the action is shown before the reply, and only once", order[-2:] == ["action", "ash"]
          and order.count("action") == 1, str(order))
    trace.update(ts=time.time() + 1, verdict={"chosen": {"tool_name": "shell", "args": "rm"},
                                              "vetoed_by": ["executive"]})
    h3.on_say("delete it")
    h3.rt._tick()
    a = kinds(h3.bus, "action")
    check("a vetoed action is shown as held back", a[-1]["status"] == "vetoed" and "executive" in a[-1]["detail"])
    n = len(kinds(h3.bus, "action"))
    trace.update(ts=time.time() - 3600)
    h3.on_say("again")
    h3.rt._tick()
    check("a stale trace from an earlier cycle is not shown again", len(kinds(h3.bus, "action")) == n)


# ----------------------------------------------------------------------
def u6_sink():
    print("\nU6  The daemon's sink and the open window")
    import ashd

    class Notify:
        class state:
            value = "active"

        def __init__(self):
            self.n = 0

        def __call__(self, text):
            self.n += 1
            return {"ok": True}

    class Per:
        def __init__(self):
            self.actuators = {"notify": Notify()}

    p = Per()
    open_ = [False]
    sink = ashd.build_response_sink(p, prefer_speech=False, window_open=lambda: open_[0])
    sink("hello", {"addressed": True})
    check("closed window: the notification is shown", p.actuators["notify"].n == 1)
    open_[0] = True
    sink("hello", {"addressed": True})
    check("open window: no duplicate notification", p.actuators["notify"].n == 1)


# ----------------------------------------------------------------------
def u7_page():
    print("\nU7  The page")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "ui", "static", "console.html")
    html = open(path, encoding="utf-8").read()
    check("token and theme placeholders exist", "{{TOKEN}}" in html and "{{THEME}}" in html)
    script = html.split("<script>", 1)[1]
    inner = [ln.strip() for ln in script.splitlines() if "innerHTML" in ln]
    check("message text is never written as HTML (only the flower drawing is)",
          len(inner) == 1 and "flowerSVG" in inner[0], str(inner))
    for kind in ("hello", "pose", "status", "user", "ash", "action", "system", "theme", "cue"):
        check(f"the page handles '{kind}'", f'case "{kind}"' in script)
    for theme in ("nebula", "blossom", "paper", "forest"):
        check(f"theme '{theme}' is defined in CSS and in the picker",
              f'[data-theme="{theme}"]' in html and f"{theme}:" in script)
    for role in ("answer", "remark", "offer", "notice"):
        check(f"ASH role '{role}' has a label", f"{role}: [" in script)
    check("the POSTs send the token header", '"X-ASH-Token": TOKEN' in script)
    check("reduced motion is respected", "prefers-reduced-motion" in html)
    check("the page reports presence while used", 'cmd: "presence"' in script and "visibilityState" in script)
    check("replayed cues never override what hello says is happening now",
          "if (!live) return;" in script.split("function onCue", 1)[1].split("}", 1)[0])
    check("a hidden pill stays hidden", "[hidden] { display: none !important; }" in html)


def main():
    print("tst_ui.py -- the console window")
    u1_bus()
    u2_server()
    u3_hooks()
    u4_learning()
    u5_actions()
    u6_sink()
    u7_page()
    print("\nALL CHECKS PASSED" if OK else "\nSOME CHECKS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
