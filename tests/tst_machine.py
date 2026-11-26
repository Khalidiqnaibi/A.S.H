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


def rig():
    return machine.Machine(procs=lambda: PROCS, mem=lambda: (92.0, int(14.7 * GB), 16 * GB), spec=lambda: dict(SPEC), own_pid=99)


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
          s == "This computer has an AMD Ryzen 7 5800H with Radeon Graphics (16 threads), 16 GB of memory and NVIDIA GeForce RTX 3060 Laptop GPU.", s)
    s = m.can_run("gta 6")
    check("can it run X: the specs, and plainly that the requirements are not checked here", s.startswith("This computer has") and "can't check what gta 6 needs" in s
          and "look up gta 6 system requirements" in s, s)
    check("nothing readable: said so", machine.Machine(procs=lambda: [], mem=lambda: (0.0, 0, 0), spec=lambda: {}).specs() == "I couldn't read this computer's hardware.")

    print("\n=== M3. In the runtime ===")
    rt = tm.rig()
    rt._machine = rig()
    used, said = rt.say("what's using my memory")
    check("answered by rule, no model", not used and said and said[-1].startswith("Memory is at 92%"), f"{used} {said}")
    used, said = rt.say("can my pc run gta 6")
    check("'can my pc run gta 6' by rule (it waited 300 s twice)", not used and said and "can't check what gta 6 needs" in said[-1], f"{used} {said}")
    unloaded = []
    rt._safe_unload = lambda: unloaded.append(1)
    rt._model_unloaded = False
    used, said = rt.say("unload the model")
    import time
    time.sleep(0.2)
    check("'unload the model' lets it go, and says so", not used and said and said[-1].startswith("Unloading the language model") and unloaded == [1], f"{said} {unloaded}")
    used, said = rt.say("free up some memory")
    check("asked again while unloaded: said so, nothing done twice", said and said[-1] == "The language model isn't loaded right now." and unloaded == [1], str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
