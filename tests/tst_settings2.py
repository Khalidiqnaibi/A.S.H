"""
tst_settings2.py -- the settings panel's model dropdown, threads, governor limits, second speech pass, search engine and live health line.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_settings2.py
"""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.ui import settings as st  # noqa: E402
from src.daemon.governor import Governor, GovernorConfig  # noqa: E402

check = tw.check


def rig():
    llm = types.SimpleNamespace(_ollama_model="qwen2.5:3b", _ollama_url="http://127.0.0.1:1/api/chat")     # nothing listens there
    mic = types.SimpleNamespace(second_pass_model="small.en", state=types.SimpleNamespace(value="active"))
    game = types.SimpleNamespace(squeezed=False)
    gov = Governor(game, GovernorConfig(), clock=lambda: 0.0)
    gov.mem, gov.cpu = 61.0, 30.0
    return types.SimpleNamespace(ash=types.SimpleNamespace(llm=llm), periphery=types.SimpleNamespace(sensors={"microphone": mic}),
                                 cfg=types.SimpleNamespace(search_engine="google"), governor=gov, clock=lambda: 0.0)


def main():
    print("\n=== P1. Reading ===")
    check("a number in range", st.parse("governor.memory_high", "90") == 90.0)
    check("threads are a whole number", st.parse("llm.threads", "4") == 4 and isinstance(st.parse("llm.threads", "4.0"), int))
    for key, bad in (("governor.memory_high", "120"), ("governor.memory_high", "ten"), ("llm.threads", "-1"), ("llm.threads", "99"),
                     ("search.engine", "altavista"), ("llm.model", "bad name; rm"), ("llm.model", "")):
        try:
            st.parse(key, bad)
            check(f"{key} refuses {bad!r}", False)
        except ValueError:
            check(f"{key} refuses {bad!r}", True)
    check("an engine, any case", st.parse("search.engine", "Bing") == "bing")
    check("a model tag", st.parse("llm.model", "qwen2.5:3b") == "qwen2.5:3b" and st.parse("llm.model", "library/phi3:mini") == "library/phi3:mini")

    print("\n=== P2. Applying live ===")
    rt = rig()
    path = os.path.join(tempfile.mkdtemp(), "daemon.json")
    with open(path, "w") as f:
        json.dump({"keep": {"me": 1}}, f)
    r = st.change(rt, path, "llm.model", "phi3:mini")
    check("the model is switched in the running daemon", r["ok"] and r["live"] and rt.ash.llm._ollama_model == "phi3:mini", str(r))
    r = st.change(rt, path, "llm.threads", "3")
    check("threads: set for the next call", r["ok"] and os.environ.get("ASH_OLLAMA_NUM_THREAD") == "3")
    st.change(rt, path, "llm.threads", "0")
    check("0 means automatic again", "ASH_OLLAMA_NUM_THREAD" not in os.environ)
    r = st.change(rt, path, "mic.second_pass", "off")
    check("the second speech pass off", r["ok"] and rt.periphery.sensors["microphone"].second_pass_model == "")
    st.change(rt, path, "mic.second_pass", "on")
    check("and on", rt.periphery.sensors["microphone"].second_pass_model == "small.en")
    r = st.change(rt, path, "search.engine", "duckduckgo")
    check("the search engine", r["ok"] and rt.cfg.search_engine == "duckduckgo")
    r = st.change(rt, path, "governor.memory_high", "80")
    check("a governor limit", r["ok"] and r["live"] and rt.governor.cfg.memory_high == 80.0, str(r))
    r = st.change(rt, path, "governor.enabled", "off")
    check("the governor off", r["ok"] and rt.governor.cfg.enabled is False)
    saved = json.load(open(path))
    check("saved, nested, the rest of the file kept",
          saved["llm"]["model"] == "phi3:mini" and saved["governor"]["memory_high"] == 80.0 and saved["keep"] == {"me": 1}, str(saved))

    print("\n=== P3. The listing ===")
    rows = {r["key"]: r for r in st.listing(rt, path)}
    check("the model is a dropdown that always includes the one in use", rows["llm.model"]["type"] == "choice" and "phi3:mini" in rows["llm.model"]["options"], str(rows["llm.model"]))
    check("the engine's options", rows["search.engine"]["options"] == ["google", "bing", "duckduckgo"] and rows["search.engine"]["value"] == "duckduckgo")
    check("numbers carry their range", (rows["governor.cpu_high"]["min"], rows["governor.cpu_high"]["max"]) == (50, 100))
    check("all of them are live", all(rows[k]["live"] for k in ("llm.model", "llm.threads", "mic.second_pass", "search.engine", "governor.enabled")))

    print("\n=== P4. The health line, and the saved values at start ===")
    rt = rig()
    h = st.health(rt)
    check("memory, CPU, model and microphone in one line", h.startswith("Now: memory 61%, CPU 30%") and "model qwen2.5:3b" in h and "microphone active" in h, h)
    rt.governor.game.squeezed, rt.governor.game.squeeze_why = True, "memory is at 91%"
    check("and when it is keeping out of the way, why", "keeping out of the way: memory is at 91%" in st.health(rt), st.health(rt))
    check("a runtime with nothing to report: an empty line, no crash", st.health(types.SimpleNamespace()) == "")
    rt = rig()
    import time as _t
    now = _t.time()
    rt.clock = lambda: now
    rt.unrouted = types.SimpleNamespace(rows=[{"t": now - 60, "took_s": 83.0, "note": "the machine was squeezed (memory at 92%)"},
                                              {"t": now - 120, "took_s": 48.0, "note": ""}, {"t": now - 180, "took_s": 3.0},
                                              {"t": now - 3 * 86400, "took_s": 300.0, "note": "memory was at 93%"}])
    h = st.health(rt)
    check("the slow answers today, and how many under load (not other days, not quick ones)", h.endswith("; 2 slow answers today (1 while the machine was under load)."), h)
    rt = rig()
    st.apply_saved(rt, {"llm": {"model": "phi3:mini", "threads": 2}, "mic": {"second_pass": False}})
    check("saved values are applied when the daemon starts", rt.ash.llm._ollama_model == "phi3:mini" and os.environ.get("ASH_OLLAMA_NUM_THREAD") == "2"
          and rt.periphery.sensors["microphone"].second_pass_model == "", os.environ.get("ASH_OLLAMA_NUM_THREAD", ""))
    os.environ.pop("ASH_OLLAMA_NUM_THREAD", None)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(here, "src", "ui", "static", "console.html"), encoding="utf-8").read()
    check("the page draws a dropdown and a number box", 's.type === "choice"' in html and 's.type === "number"' in html and "settingsHealth" in html)
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
