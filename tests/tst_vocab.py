"""
tst_vocab.py -- the recogniser is told which words to lean towards (names, classes, "port"), locally only.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_vocab.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.senses.mic import MicrophoneSensor  # noqa: E402

check = tw.check


def v1_prompt():
    print("\n=== V1. The prompt ===")
    m = MicrophoneSensor(enabled=False)
    check("with no words, the prompt is what it always was", m.prompt_hint() == "Talking to Ash, my assistant.")
    m.hint_words = lambda: ["Khalid", "Linda", "NoSQL", "linda", "", "x" * 40]
    p = m.prompt_hint()
    check("words are listed, once each, long junk dropped", p == "Talking to Ash, my assistant. Words I use: Khalid, Linda, NoSQL.", p)
    m.hint_words = lambda: [f"Word{i}" for i in range(60)]
    check("at most 25 of them", len(m.prompt_hint().split(":")[1].split(",")) == 25, m.prompt_hint())
    m.hint_words = lambda: 1 / 0
    check("a failing source does not break hearing", m.prompt_hint() == "Talking to Ash, my assistant.")


def v2_runtime():
    print("\n=== V2. From the runtime ===")
    import tst_effects as te
    d = tempfile.mkdtemp()
    rt = te.rig(wonder=True, knowledge_path=os.path.join(d, "k.json"), wonder_path=os.path.join(d, "w.json"),
                timetable=True, timetable_path=os.path.join(d, "tt.json"))
    rt.brain.user = "Khalid"
    rt.say("remember linda is my sister", source="console")
    rt.say("my NoSQL class is on Sunday from 10 to 11:30", source="console")
    words = rt.vocab_hint()
    check("the user, the people told about, the classes, and the words that were misheard",
          "Khalid" in words and "Linda" in words and "NoSQL" in words and "port" in words, str(words))
    mic = MicrophoneSensor(enabled=False)
    rt.periphery.sensors["microphone"] = mic
    rt._hint_the_microphone()
    check("the microphone is given the source", "Linda" in mic.prompt_hint(), mic.prompt_hint())


if __name__ == "__main__":
    v1_prompt()
    v2_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
