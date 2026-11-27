"""tst_devicehelp.py -- Bluetooth help ("I'm having a trouble with my live buds" took 75 s on the model, 26 Nov), the internet check, keeping the PC
awake, the exact volume, and "run a diagnostic". Every reader and setter is a fake: nothing here touches Bluetooth, the network, sleep or the
volume of this machine.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_devicehelp.py"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402
from src.daemon import devicehelp as dh  # noqa: E402

check = tw.check
PAIRED = [("Galaxy Buds Live (0922)", False), ("MX Master 3", True), ("JBL Flip 5", False)]


class FakeVol:
    def __init__(self):
        self.level, self.sets = 18, []

    def get(self):
        return self.level

    def set(self, n):
        self.sets.append(n)
        self.level = n
        return True


def rig(paired=PAIRED, net=None):
    rt = tm.rig()
    rt.desktop = td.drig()
    vol = FakeVol()
    awake_calls = []
    rt._bt = dh.Bluetooth(reader=lambda: list(paired), clock=lambda: rt.clock())
    rt._net = net or dh.Network(wifi=lambda: ("Home", 78), gw=lambda: "192.168.1.1", ping_fn=lambda h: 3.0, dns=lambda: True, out=lambda: 40.0)
    rt._awake = dh.Awake(setter=lambda on: awake_calls.append(on) or True, clock=time.time, threaded=False)
    rt._volume = dh.Volume(get=vol.get, set_=vol.set)
    return rt, vol, awake_calls


def main():
    print("\n=== D1. Bluetooth ===")
    for t, kind in (("I'm having a trouble with my live buds.", "trouble"), ("I'm having trouble with connecting my earbuds", "trouble"),
                    ("my headphones won't connect", "trouble"), ("are my buds connected?", "status"), ("connect my buds", "connect"),
                    ("what's connected over bluetooth", "list"), ("open bluetooth settings", "settings")):
        p = dh.Bluetooth.parse(t)
        check(f"{t!r} -> {kind}", p is not None and p[0] == kind, str(p))
    for t in ("I'm having trouble with my code", "connect the dots", "I have a problem with my boss"):
        check(f"not a device: {t!r}", dh.Bluetooth.parse(t) is None, str(dh.Bluetooth.parse(t)))
    rt, vol, _ = rig()
    used, said = rt.say("I'm having a trouble with my live buds.")
    check("the buds are found by what was heard, and said as they are -- no model", not used and said and
          said[-1] == "Your Galaxy Buds Live are paired but not connected right now. Want me to open Bluetooth settings to connect them?", f"{used} {said}")
    used, said = rt.say("yes")
    check("a yes opens Bluetooth settings", rt.desktop.ex_.opened == ["ms-settings:bluetooth"] and said[-1] == "Opening Bluetooth settings.",
          f"{rt.desktop.ex_.opened} {said}")
    rt, vol, _ = rig()
    used, said = rt.say("is my mouse connected")
    check("a connected one: said so, and the sound/settings offer fits", said and said[-1].startswith("Your MX Master 3 shows as connected"), str(said))
    rt, vol, _ = rig(paired=[])
    used, said = rt.say("my headphones won't connect")
    check("nothing paired: said so, with the offer", said and said[-1].startswith("I don't see your headphones paired"), str(said))
    used, said = rt.say("no")
    check("a no: nothing opened", rt.desktop.ex_.opened == [], str(rt.desktop.ex_.opened))
    rt, vol, _ = rig()
    used, said = rt.say("what bluetooth devices do I have")
    check("the list, connected and not", said and said[-1] == "Connected: MX Master 3. Paired but not connected: Galaxy Buds Live, JBL Flip 5.", str(said))
    check("paired names become hearing hints, without Windows' code", "Galaxy Buds Live" in rt.vocab_hint(), str(rt.vocab_hint()))

    print("\n=== D2. The internet ===")
    for t in ("is the internet down?", "is my wifi working", "am I online", "why is the internet so slow", "check my connection"):
        check(f"{t!r} is a network question", dh.Network.parse(t))
    ok = dh.Network(wifi=lambda: ("Home", 78), gw=lambda: "192.168.1.1", ping_fn=lambda h: 3.0, dns=lambda: True, out=lambda: 40.0).check()
    check("working", ok == "The internet is working: Wi-Fi “Home” at 78%, and the outside answers in 40 ms.", ok)
    down = dh.Network(wifi=lambda: ("Home", 78), gw=lambda: "192.168.1.1", ping_fn=lambda h: 3.0, dns=lambda: False, out=lambda: None).check()
    check("the router answers, nothing past it", down.startswith("Your Wi-Fi “Home” at 78% is fine as far as the router, but nothing answers past it"), down)
    dns = dh.Network(wifi=lambda: ("Home", 78), gw=lambda: "x", ping_fn=lambda h: 3.0, dns=lambda: False, out=lambda: 30.0).check()
    check("up but names don't resolve: DNS", "a DNS problem" in dns, dns)
    local = dh.Network(wifi=lambda: ("Home", 20), gw=lambda: "192.168.1.1", ping_fn=lambda h: None, dns=lambda: False, out=lambda: None).check()
    check("the router itself unreachable", local.startswith("You're on Wi-Fi “Home” at 20%, but I can't reach the router"), local)
    none = dh.Network(wifi=lambda: ("", None), gw=lambda: "", ping_fn=lambda h: None, dns=lambda: False, out=lambda: None).check()
    check("no network at all", none.startswith("You're not connected to any network"), none)
    rt, vol, _ = rig()
    used, said = rt.say("is the internet down?")
    check("asked in the runtime: by rule", not used and said and said[-1].startswith("The internet is working"), f"{used} {said}")

    print("\n=== D3. Keeping it awake ===")
    rt, vol, calls = rig()
    used, said = rt.say("keep my pc awake for 2 hours")
    check("kept awake, until a time said", calls == [True] and said[-1].startswith("Okay -- I'll keep the computer awake until ") and rt._awake.active(), str(said))
    used, said = rt.say("let it sleep")
    check("let go", calls == [True, False] and said[-1] == "Okay -- it can sleep again." and not rt._awake.active(), f"{calls} {said}")
    rt, vol, calls = rig()
    used, said = rt.say("don't let the computer sleep until the download finishes")
    check("until a download: up to three hours", "for up to 3 hours" in said[-1], str(said))
    rt, vol, calls = rig()
    rt.say("keep the laptop awake for 20 hours")
    check("never more than eight hours", rt._awake.until - time.time() <= dh.MAX_AWAKE_S + 5, str(rt._awake.until - time.time()))
    rt, vol, calls = rig()
    rt.say("keep my pc awake for 2 hours", source="microphone")
    used, said = rig()[0].say("keep my pc awake for 2 hours")
    check("'for 30 minutes' read right", abs(dh._for("for 30 minutes", time.time()) - 1800) < 1 and abs(dh._for("for an hour", time.time()) - 3600) < 1)

    print("\n=== D4. The volume ===")
    for t, want in (("volume to 30", ("set", 30)), ("set the volume to 50 percent", ("set", 50)), ("turn the volume down to twenty", ("set", 20)),
                    ("what's the volume?", ("get", None)), ("volume to 300", None)):
        check(f"{t!r} -> {want}", dh.Volume.parse(t) == want, str(dh.Volume.parse(t)))
    rt, vol, _ = rig()
    used, said = rt.say("volume to 30")
    check("set exactly, by rule", vol.sets == [30] and said[-1] == "Volume at 30%." and not used, f"{vol.sets} {said}")
    used, said = rt.say("what's the volume")
    check("read back", said[-1] == "The volume is at 30%.", str(said))
    rt, vol, _ = rig()
    used, said = rt.say("turn the volume up")
    check("'turn the volume up' (no number) is still the media key's", vol.sets == [] and said and said[-1] != "Volume at 0%.", str(said))

    print("\n=== D5. A self-check ===")
    for t in ("run a diagnostic", "run diagnostics", "self check", "are your systems okay?", "is everything working", "status report"):
        check(f"{t!r} is a self-check", dh.is_diagnostic(t))
    check("'are you okay' is about feelings, not systems", not dh.is_diagnostic("are you okay"))
    mic = types.SimpleNamespace(state=types.SimpleNamespace(value="active"), _last_asr_conf=-0.2)
    fake = types.SimpleNamespace(periphery=types.SimpleNamespace(sensors={"microphone": mic}),
                                 ash=types.SimpleNamespace(llm=types.SimpleNamespace(_ollama_model="qwen2.5:3b")),
                                 governor=types.SimpleNamespace(mem=61.0, cpu=12.0), unrouted=types.SimpleNamespace(rows=[{"took_s": 4.0}, {"took_s": 6.0}]),
                                 stats={"errors": 0})
    line = dh.diagnose(fake)
    check("the microphone, the model and its speed, memory, errors", line.startswith("Microphone: active -- the last thing I heard, I was sure of.")
          and "Language model: qwen2.5:3b, recent answers about 5 s." in line and "Memory 61%, CPU 12%." in line and "No errors since I started." in line
          and "What would help" not in line, line)
    fake.governor, fake.unrouted.rows, fake.stats = types.SimpleNamespace(mem=93.0, cpu=40.0), [{"took_s": 48.0}, {"took_s": 83.0}], {"errors": 2}
    line = dh.diagnose(fake)
    check("under load: what would help", "What would help: a smaller model in the settings would answer faster; closing " in line
          and "2 internal errors since I started" in line, line)
    rt, vol, _ = rig()
    used, said = rt.say("run a diagnostic")
    check("asked in the runtime: answered by rule", not used and said and said[-1].startswith("Microphone"), f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
