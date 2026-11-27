"""tst_modelhint.py -- 27 Nov: the 7B mistral made every answer 40-160 s; a slow answer now says so once a day with the way out ("use the small model");
"what did I tell you today?" is read back from this session; "code me a small python game and run it" gets an honest answer (not "I can't" invented),
and a bare "ask Gemini" afterwards takes the request. Fakes only; the settings file is a temp file and Ollama is not asked.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_modelhint.py"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tst_desktop as td  # noqa: E402
from src.daemon.ambient import AmbientRuntime  # noqa: E402
from src.ui import settings as st  # noqa: E402

check = tw.check


def rig():
    rt = tm.rig()
    rt.ash.llm = types.SimpleNamespace(_ollama_model="mistral:latest", _ollama_url="http://127.0.0.1:1/api/chat", _mode="ollama")
    rt.cfg.notices_path = os.path.join(tempfile.mkdtemp(), "n.json")
    rt.config_path = os.path.join(tempfile.mkdtemp(), "daemon.json")
    json.dump({}, open(rt.config_path, "w"))
    st.choices = lambda runtime, key: ["mistral:latest", "qwen2.5:3b", "nomic-embed-text:latest"] if key == "llm.model" else []
    return rt


def main():
    print("\n=== H1. A big model on a squeezed machine ===")
    P = AmbientRuntime._model_params_b
    check("sizes", P("qwen2.5:3b") == 3.0 and P("mistral:latest") == 7.0 and P("llama3.2:1b") == 1.0 and P("phi3:mini") == 3.0 and P("gemma2:9b") == 9.0)
    rt = rig()
    h = rt._big_model_hint()
    check("said with the way out", h == " mistral:latest is a big model for this machine -- say “use the small model” to switch to qwen2.5:3b.", h)
    check("once a day", rt._big_model_hint() == "")
    rt2 = rig()
    rt2.ash.llm._ollama_model = "qwen2.5:3b"
    check("a small model: nothing said", rt2._big_model_hint() == "")

    print("\n=== H2. Switching by voice ===")
    rt = rig()
    used, said = rt.say("use the small model")
    check("the small one, saved, no model call", not used and rt.ash.llm._ollama_model == "qwen2.5:3b" and said[-1].startswith("Switched to qwen2.5:3b."), f"{used} {said}")
    check("kept in the settings file", json.load(open(rt.config_path))["llm"]["model"] == "qwen2.5:3b")
    used, said = rt.say("switch to mistral")
    check("by name (without the tag)", rt.ash.llm._ollama_model == "mistral:latest", f"{rt.ash.llm._ollama_model} {said}")
    used, said = rt.say("use google")
    check("a word that is not a model is left alone", rt.ash.llm._ollama_model == "mistral:latest")

    print("\n=== H3. What was told this session ===")
    rt = rig()
    rt.say("I'm working on a custom machine that runs a language model all the time")
    rt.say("my brother Omar is studying medicine in Jordan this year")
    used, said = rt.say("what did I tell you today")
    check("read back from this session, no model", not used and said and "custom machine" in said[-1] and "Omar" in said[-1], f"{used} {said}")
    rt = rig()
    used, said = rt.say("what did I tell you in this session")
    check("nothing told: said so", said and said[-1].startswith("You haven't told me anything this session"), str(said))

    print("\n=== H4. 'Code me a small python game and run it' ===")
    rt = rig()
    rt.desktop = td.drig()
    rt.desktop.ex_.procs.add("opera.exe")
    used, said = rt.say("code me a small python game and run it")
    check("honest, with what it can do, no model", not used and said and "I don't write programs and run them" in said[-1] and "ask Gemini" in said[-1], f"{used} {said}")
    clip = []
    real_a = rt.periphery.actuators

    class A(dict):
        def get(self, k, d=None):
            return types.SimpleNamespace(act=lambda x: (clip.append(x), {"ok": True})[1]) if k == "clipboard" else real_a.get(k, d)
    rt.periphery.actuators = A(real_a)
    used, said = rt.say("ask Gemini")
    check("'ask Gemini' then takes that request", clip and clip[-1].lower().startswith("code me a small python game"), f"{clip} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
