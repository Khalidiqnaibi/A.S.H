"""tst_machine.py -- "what's using my memory?", "what are my specs?", "can my PC run GTA 6?", "unload the model": by rule, from the machine.
Injected readers only; nothing real is unloaded.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_machine.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import machine  # noqa: E402

check = tw.check
GB = 1024 ** 3
PROCS = [("opera.exe", int(2.0 * GB), 10), ("opera.exe", int(3.1 * GB), 11), ("ollama_llama_server.exe", int(3.2 * GB), 12),
         ("Code.exe", int(1.4 * GB), 13), ("python.exe", int(0.9 * GB), 99), ("Discord.exe", int(0.5 * GB), 14), ("svchost.exe", int(0.2 * GB), 15)]
SPEC = {"cpu": "AMD Ryzen 7 5800H with Radeon Graphics", "threads": 16, "cores": 8, "ram": 16 * GB,
        "gpus": ["NVIDIA GeForce RTX 3060 Laptop GPU", "Microsoft Basic Display Adapter"]}


def rig(online=False, req=None, spec=None):
    """Never touches the network: `online` and `req` are the test's own."""
    return machine.Machine(procs=lambda: PROCS, mem=lambda: (92.0, int(14.7 * GB), 16 * GB), spec=lambda: dict(spec or SPEC), own_pid=99,
                           online=lambda: online, req=req or (lambda name: None))


def main():
    print("\n=== M1. What is asked ===")
    for t, want in [("what's using my memory", "memory"), ("what is eating my ram", "memory"), ("why is my computer so slow today", "memory"),
                    ("what's hogging all my memory right now?", "memory"), ("what are my specs", "specs"), ("what gpu do I have", "specs"),
                    ("how much ram do I have", "specs"), ("what's my processor", "specs")]:
        check(f"{t!r} -> {want}", (machine.parse(t) or ("",))[0] == want, str(machine.parse(t)))
    check("'can my pc run gta 6' -> run, GTA 6 named", machine.parse("can my pc run gta 6") == ("run", "gta 6"))
    for t in ("what's using my phone", "why is the sky blue", "can you run a scan", "what are my plans", "my computer is slow"):
        check(f"{t!r} is not this", machine.parse(t) is None, str(machine.parse(t)))

    print("\n=== M2. The answers ===")
    m = rig()
    s = m.memory()
    check("memory: the level, the top users summed by name, ASH's own share, and the way out",
          s.startswith("Memory is at 92% (15 GB of 16 GB). The most: Opera 5.1 GB, the language model (Ollama) 3.2 GB, VS Code 1.4 GB, Discord 0.5 GB, svchost 0.2 GB.")
          and "I'm using 0.9 GB myself, and the language model holds 3.2 GB." in s and "unload the model" in s, s)
    s = m.specs()
    check("specs: processor, threads, memory, the real graphics card (not the basic display adapter)",
          s == "This computer has an AMD Ryzen 7 5800H (16 threads), 16 GB of memory and an RTX 3060.", s)
    s = m.can_run("gta 6")
    check("can it run X, offline: the specs, and plainly that it can't look it up", s.startswith("This computer has") and "I'm offline right now" in s and "GTA 6" in s, s)
    check("nothing readable: said so", machine.Machine(procs=lambda: [], mem=lambda: (0.0, 0, 0), spec=lambda: {}).specs() == "I couldn't read this computer's hardware.")

    print("\n=== M3. In the runtime ===")
    rt = tm.rig()
    rt._machine = rig()
    used, said = rt.say("what's using my memory")
    check("answered by rule, no model", not used and said and said[-1].startswith("Memory is at 92%"), f"{used} {said}")
    used, said = rt.say("can my pc run gta 6")
    check("'can my pc run gta 6' by rule (it waited 300 s twice)", not used and said and "I'm offline right now" in said[-1] and "GTA 6" in said[-1], f"{used} {said}")
    unloaded = []
    rt._safe_unload = lambda: unloaded.append(1)
    rt._model_unloaded = False
    used, said = rt.say("unload the model")
    import time
    time.sleep(0.2)
    check("'unload the model' lets it go, and says so", not used and said and said[-1].startswith("Unloading the language model") and unloaded == [1], f"{said} {unloaded}")
    used, said = rt.say("free up some memory")
    check("asked again while unloaded: said so, nothing done twice", said and said[-1] == "The language model isn't loaded right now." and unloaded == [1], str(said))
    print("\n=== M4. Can this PC run it: looked up online, only when online ===")
    from src.daemon import canrun
    for t in ["Can this little stupid old device run GTA 6?", "can my pc run cyberpunk", "can I run minecraft on this laptop", "will elden ring run on my pc", "can this device handle valorant"]:
        check(f"asked: {t!r}", machine.parse(t) is not None and machine.parse(t)[0] == "run", str(machine.parse(t)))
    check("not asked: 'can you run a timer'", machine.parse("can you run a timer for me") is None)
    GAME = {"name": "Cyberpunk 2077", "minimum": {"ram_gb": 12.0, "gpu": ["GeForce GTX 1060 6GB", "Radeon RX 580 8GB", "Arc A380"], "gfx_text": "x"},
            "recommended": {"ram_gb": 16.0, "gpu": ["GeForce RTX 2060 SUPER", "Radeon RX 5700 XT"], "gfx_text": "y"}}
    weak = {"cpu": "Intel i5", "ram": 20 * GB, "gpus": ["NVIDIA GeForce GTX 1630"]}
    strong = {"cpu": "Ryzen 7", "ram": 32 * GB, "gpus": ["NVIDIA GeForce RTX 3070"]}
    off = rig(online=False, spec=weak).can_run("cyberpunk")
    check("offline: the specs, and that it can't look it up -- no guess", "offline" in off and "GTX 1630" in off and "Probably" not in off and "should run" not in off, off)
    asked = []
    on = rig(online=True, req=lambda n: asked.append(n) or GAME, spec=weak).can_run("cyberpunk")
    check("online, a weak card: probably not, with the card it asks for and the card you have", on.startswith("Probably not: Cyberpunk 2077") and "GeForce GTX 1060 6GB" in on and "GTX 1630" in on and asked == ["cyberpunk"], on)
    check("online, a strong card: should run well", rig(online=True, req=lambda n: GAME, spec=strong).can_run("cyberpunk").startswith("Cyberpunk 2077 should run well"))
    short = rig(online=True, req=lambda n: GAME, spec={"ram": 8 * GB, "gpus": ["NVIDIA GeForce RTX 3070"]}).can_run("cyberpunk")
    check("short of memory: said with the numbers", short.startswith("Probably not") and "12 GB" in short and "8 GB" in short, short)
    check("not on Steam: said, not made up", "doesn't list" in rig(online=True, req=lambda n: None, spec=weak).can_run("gta 6"))

    def boom(n):
        raise OSError("down")
    check("online but the store can't be reached: said", "couldn't check what x needs -- down" in rig(online=True, req=boom, spec=weak).can_run("x"))
    unk = {"name": "X", "minimum": {"ram_gb": 4.0, "gpu": ["Some Future Card 9000"], "gfx_text": "Some Future Card 9000"}, "recommended": None}
    check("a card the table doesn't know is said, not guessed", "can't rank" in rig(online=True, req=lambda n: unk, spec=weak).can_run("x"))
    check("the GPU table ranks roughly", canrun.gpu_score("GeForce GTX 1060 6GB") > canrun.gpu_score("GeForce GTX 1630") and canrun.gpu_score("RTX 3060 Laptop GPU") > canrun.gpu_score("GTX 1060") and canrun.gpu_score("no such card") is None)
    block = "<strong>Minimum:</strong><br><ul><li><strong>Memory:</strong> 12 GB RAM<br></li><li><strong>Graphics:</strong> GeForce GTX 1060 6GB or Radeon RX 580 8GB or Arc A380<br></li><li><strong>Storage:</strong> 70 GB available space<br></li></ul>"
    r = canrun.read_requirements(block)
    check("a Steam requirements block is read", r["ram_gb"] == 12.0 and len(r["gpu"]) == 3 and "70 GB" in r["storage"], str(r))
    items = [{"id": 1, "name": "Cyberpunk 2077: Phantom Liberty"}, {"id": 2, "name": "Cyberpunk 2077"}, {"id": 3, "name": "Cyberpunk 2077 REDmod"}]
    check("the base game is picked, not the add-on", canrun.pick_game("cyberpunk 2077", items)["id"] == 2 and canrun.pick_game("zzzz qqq", items) is None)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
