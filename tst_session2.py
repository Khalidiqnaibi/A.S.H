"""
tst_session2.py -- the smaller fixes from the session of 2026-11-06.

  S1  "Ass system status" is the name misheard; "my ass hurts" is not
  S2  Whisper is told the name to expect
  S3  "What's your recommendation for the storage allocation?" is a disk question
  S4  Entities: "Smile", "Turn", "Nope" at the start of a sentence are not
      names; real names still are; ASH's own replies link known entities
      but never create new ones
  S5  Intent shapes are read once, not three times per utterance
  S6  Speech: lists and line breaks become one line

Run:  PYTHONIOENCODING=utf-8 python tst_session2.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402

check = tw.check


def s1_name():
    print("\n=== S1. The name, misheard ===")
    from src.senses import address
    from src.senses.mic import fix_name
    for t, want in (("Ass system status", "Ash system status"), ("ass, what time is it", "Ash, what time is it"),
                    ("Hey ass turn off the mic", "Hey Ash turn off the mic"), ("Asch what is up", "Ash what is up")):
        check(f"{t!r} -> {want!r}", fix_name(t) == want, fix_name(t))
        check(f"...and it is addressed", address.score(fix_name(t)).confidence >= 0.9)
    for t in ("my ass hurts", "Ass.", "He is an ass about it", "assume nothing", "", None):
        check(f"left alone: {t!r}", fix_name(t) == (t or ""))
    import threading
    import numpy as np
    from src.senses.mic import MicrophoneSensor
    m = MicrophoneSensor.__new__(MicrophoneSensor)
    m.voiceprint, m.keep_audio, m._speech_seen = False, False, 0
    m._lock, m._out = threading.Lock(), []
    m.transcribe = lambda audio: "Ass system status"
    m._finish(np.zeros(16000 * 2, dtype=np.int16))
    check("the microphone passes it on as the name", m._out and m._out[0].text == "Ash system status"
          and m._out[0].addressed, str([e.text for e in m._out]))


def s2_whisper():
    print("\n=== S2. Whisper is told the name ===")
    import numpy as np
    from src.senses.mic import MicrophoneSensor
    got = {}

    class W:
        def transcribe(self, audio, **kw):
            got.update(kw)
            return [type("S", (), {"text": " Ass system status"})()], None

    m = MicrophoneSensor.__new__(MicrophoneSensor)
    m._transcriber, m._whisper = None, W()
    try:
        text = m.transcribe(np.zeros(16000, dtype=np.int16))
    except Exception as e:
        text = f"error: {e}"
    check("the name is in the prompt", "Ash" in str(got.get("initial_prompt", "")), str(got))
    check("(transcribed)", "system status" in text, text)


def s3_storage():
    print("\n=== S3. Storage questions ===")
    from src.daemon import ambient as a
    for t in ("What's your recommendation for the storage allocation in this system?", "any tips on disk space",
              "what is my storage usage", "how's the disk situation", "can you suggest how to clean my SSD"):
        check(f"{t!r}", bool(a._STORAGE.search(t)))
    for t in ("recommend a good movie", "I bought storage boxes", "suggest a name for my dog", "space is big"):
        check(f"not: {t!r}", not a._STORAGE.search(t))


def s4_entities():
    print("\n=== S4. Entities ===")
    from src.memory.ner import NERExtractor
    n = NERExtractor()
    for t in ("Smile for me, Ash.", "Turn off the mic.", "Nope, you didn't.", "Given that our last interaction was "
              "recent:\n1. Answering questions\n2. Task management", "Good night. Let me know."):
        names = [e["text"] for e in n.extract(t) if e["text"] not in ("Ash", "A.S.H")]
        check(f"no junk from {t[:30]!r}", names == [], str(names))
    n.nlp, keep = None, n.nlp                      # the fallback alone, without spaCy
    check("a two-word name at the start still counts; one capitalised word does not",
          [e["text"] for e in n.extract("Project Falcon starts today. Smile now.")] == ["Project Falcon"])
    n.nlp = keep
    for t, want in (("I had lunch with Omar today", "Omar"), ("Omar called. Sara Ali too.", "Sara Ali"),
                    ("it runs on a Raspberry Pi", "Raspberry Pi")):
        check(f"a real name: {want}", want in [e["text"] for e in n.extract(t)])

    from src.memory.memory_router import MemoryRouter
    made, linked = [], []

    class Known:
        entity_id = "known-1"

    class Ent:
        def ingest(self, payload):
            made.append(payload["primary_identifiers"]["name"])
            return type("E", (), {"entity_id": "new"})()

        def find_matching_entities(self, mentions):
            return [Known()] if mentions[0].lower() == "omar" else []

    class Ep:
        def add_episode(self, **kw):
            linked.append(kw["related_entities"])
            return type("E", (), {"episode_id": "ep"})()

    class Ner:
        def extract(self, text):
            return [{"text": "Omar", "label": "person"}, {"text": "Entertainment", "label": "proper"}]

    r = MemoryRouter.__new__(MemoryRouter)
    r.ner, r.entity, r.episodic, r.core = Ner(), Ent(), Ep(), None
    r.config = {"core_force_importance": 2.0}
    r._summarize_for_episode = lambda text, ents, actor: text
    r.route_utterance("Omar and Entertainment", source="ash")
    check("ASH's reply: nothing created", made == [], str(made))
    check("...but the known one is linked", linked[-1] == ["known-1"], str(linked))
    r._classify_core_candidate = lambda text: None
    r.route_utterance("Omar and Entertainment", source="chat")
    check("the user's words still create", made == ["omar", "entertainment"], str(made))


def s5_shapes():
    print("\n=== S5. Shapes read once ===")
    import importlib
    cl = importlib.import_module("tools.classification")
    from tools.shapes import shape_loader
    calls = []
    real = shape_loader.load_all_shapes

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)
    shape_loader.load_all_shapes = counting
    cl._SHAPES.update({"mtime": None, "blocks": None, "warned": False})
    try:
        for _ in range(5):
            try:
                cl._ensure_catalogs_loaded()
            except Exception:
                pass
        check("five calls, one read", len(calls) == 1, str(len(calls)))
        cl._SHAPES["mtime"] = -1                       # a shape file changed
        try:
            cl._ensure_catalogs_loaded()
        except Exception:
            pass
        check("read again when a file changes", len(calls) == 2, str(len(calls)))
    finally:
        shape_loader.load_all_shapes = real


def s6_speech():
    print("\n=== S6. Speech ===")
    from src.senses.actuators import speakable
    s = speakable("Here are a few things:\n\n1. Answering questions: I can help.\n2. **System** status\n- a bullet")
    check("one line, no numbers or stars", "\n" not in s and "1." not in s and "**" not in s, s)
    check("still reads in order", s == "Here are a few things: Answering questions: I can help. System status. a bullet", s)
    check("a plain line is untouched", speakable("It's 5 pm.") == "It's 5 pm.")
    check("version numbers are not list markers", speakable("Use Python 3.11 here.") == "Use Python 3.11 here.")
    check("nothing in, nothing out", speakable("") == "" and speakable(None) == "")
    import time
    from src.senses.actuators import SpeakActuator
    got = []

    class Tts:
        engine = True

        def stream_audio(self, text, voice=None):
            got.append(text)
            yield b"RIFF"
    sp = SpeakActuator(tts_engine=Tts(), player=lambda w, st: None, dry_run=False)
    sp.enable()
    sp.act("Two things:\n1. one\n2. two")
    time.sleep(0.1)
    check("the speaker speaks the flattened line", got == ["Two things: one. two"], str(got))


if __name__ == "__main__":
    for fn in (s1_name, s2_whisper, s3_storage, s4_entities, s5_shapes, s6_speech):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
