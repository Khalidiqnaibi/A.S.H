"""
tst_inventory.py -- what is installed (apps and games), answered from the machine.

25 Nov: the missing installed apps/games inventory -- "what games do I have" had nothing to read. Run:  PYTHONIOENCODING=utf-8 python tests/tst_inventory.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import inventory  # noqa: E402

check = tw.check

ITEMS = [{"name": "Cyberpunk 2077", "kind": "game", "open": "steam://rungameid/1091500"},
         {"name": "Minecraft Launcher", "kind": "game", "open": "com.epicgames.launcher://apps/x"},
         {"name": "Blender", "kind": "app", "open": r"C:\ProgramData\Start\Blender.lnk"},
         {"name": "Visual Studio Code", "kind": "app", "open": r"C:\x\Code.lnk"},
         {"name": "Stremio", "kind": "app", "open": r"C:\x\Stremio.lnk"}]


def mk():
    opened = []
    calls = []
    t = [1000.0]

    def scan():
        calls.append(1)
        return list(ITEMS)

    d = tempfile.mkdtemp()
    inv = inventory.Inventory(path=os.path.join(d, "apps.json"), scan=scan, opener=lambda x: opened.append(x) or True, clock=lambda: t[0])
    inv.t, inv.opened, inv.calls = t, opened, calls
    return inv


def i1_questions():
    print("\n=== I1. What is installed ===")
    inv = mk()
    say = inv.hear("what games do I have")
    check("games, by name, with the count", say == "2 games: Cyberpunk 2077, Minecraft Launcher.", say)
    say = inv.hear("what apps do I have installed")
    check("apps", say == "3 apps: Blender, Stremio, Visual Studio Code.", say)
    check("do I have X installed: yes", inv.hear("do I have Blender installed") == "Yes -- Blender is installed.")
    check("a close name matches", inv.hear("is Visual Studio installed") == "Yes -- Visual Studio Code is installed.", inv.hear("is Visual Studio installed"))
    check("not installed: said", inv.hear("do I have Photoshop installed") == "I can't find Photoshop installed.")
    check("read once, not on every question", len(inv.calls) == 1, str(len(inv.calls)))
    inv.t[0] += 90000
    inv.hear("what games do I have")
    check("a day later it reads again", len(inv.calls) == 2)
    check("rescan on request", inv.hear("rescan my apps") == "Scanned: 3 apps and 2 games." and len(inv.calls) == 3)
    again = inventory.Inventory(path=inv.path, scan=lambda: [], clock=lambda: inv.t[0])
    check("kept on disk for the next start", len(again.items) == 5)
    for t in ("do I have any time", "do I have to go", "what time is it", "who is Linda", "do I have a meeting today"):
        check(f"not an installed-apps question: {t!r}", inv.hear(t) is None, str(inv.hear(t)))


def i2_open():
    print("\n=== I2. Opening what only the inventory knows ===")
    inv = mk()
    check("a game by name", inv.hear("open Cyberpunk") == "Opening Cyberpunk 2077." and inv.opened[-1] == "steam://rungameid/1091500", str(inv.opened))
    check("an app", inv.hear("launch blender") == "Opening Blender." and inv.opened[-1].endswith("Blender.lnk"))
    n = len(inv.opened)
    check("the desktop rules' own app is left to them", inv.hear("open Stremio", known_app=lambda x: x.lower() == "stremio") is None and len(inv.opened) == n)
    check("an unknown name is left alone (no guess)", inv.hear("open the readme") is None and inv.hear("open zxqv") is None and len(inv.opened) == n)
    check("not 'open my notes'", inv.hear("open my notes") is None)


def i2b_switch():
    print("\n=== I2b. Switching to an installed app by its name (28 Nov: 'switch to claude', 68 s on the model) ===")
    inv = mk()
    inv.items = inv.items + [{"name": "Claude", "kind": "app", "open": "shell:AppsFolder\\Claude_x!Claude"}]
    inv.at = inv.t[0]
    seen = []
    check("brought forward when its window is there", inv.hear("switch to claude", switch=lambda hit: seen.append(hit["name"]) or True) == "Switching to Claude." and seen == ["Claude"])
    check("'switch to Claude desktop' and a misheard 'calude' find the same app", inv.hear("switch to calude", switch=lambda hit: True) == "Switching to Claude.")
    check("not open: says so and how to open it", inv.hear("switch to claude", switch=lambda hit: None) == "Claude isn't open. Say “open Claude” and I will.")
    check("open but Windows won't bring it: said, not claimed", "wouldn't bring it forward" in inv.hear("switch to claude", switch=lambda hit: False))
    check("an app the desktop rules know is theirs", inv.hear("switch to claude", known_app=lambda n: True, switch=lambda hit: True) is None)
    check("not an installed thing: left alone", inv.hear("switch to the other tab", switch=lambda hit: True) is None and inv.hear("switch to zxqv", switch=lambda hit: True) is None)
    import tst_desktop as td
    h = td.rt_rig()
    rt = h.rt
    inv2 = mk()
    inv2.clock = rt.clock
    inv2.items = inv2.items + [{"name": "Claude", "kind": "app", "open": "shell:x"}]
    inv2.at = rt.clock()
    rt.inventory = inv2
    h.ex.windows = lambda: [("Claude", "claude.exe")]
    h.say("switch to claude")
    check("on the runtime: the window is found by its title and brought forward, no model", h.ex.focused[-1:] == ["claude.exe"] and h.said()[-1] == "Switching to Claude.", f"{h.ex.focused} {h.said()}")


def i3_runtime():
    print("\n=== I3. On the runtime ===")
    import tst_effects as te
    h = te.rig()
    rt = getattr(h, "rt", h)
    rt.inventory = mk()
    n = len(rt.ash.calls)
    h.say("what games do I have", source="console")
    check("said to ASH: answered from the inventory, no model", h.out[-1].startswith("2 games:") and len(rt.ash.calls) == n, h.out[-1])
    check("in the status", rt.status()["inventory"]["games"] == 2)


if __name__ == "__main__":
    i1_questions()
    i2_open()
    i2b_switch()
    i3_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
