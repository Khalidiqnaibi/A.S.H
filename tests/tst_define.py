"""
tst_define.py -- "define segue" from a dictionary, the source named; only the word leaves the machine.

"What's the definition of segue?" was answered "I couldn't find a Wikipedia page for 'definition of segue'" (a definition taken for an article
title), or went to the model, which defines words fluently whether or not it knows them.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_define.py
"""

import os
import sys
import tempfile
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import define  # noqa: E402
from src.daemon.define import Define, DefineConfig  # noqa: E402

check = tw.check
SEGUE = {"en": [{"partOfSpeech": "Noun", "definitions": [{"definition": "A <a href='x'>smooth</a> transition from one thing to another."}]},
                {"partOfSpeech": "Verb", "definitions": [{"definition": "To move &amp; continue without interruption."}]}]}


def d1_parse():
    print("\n=== D1. What is asked ===")
    for t, w in [("Define segue", "segue"), ("what's the definition of segue?", "segue"), ("what does ubiquitous mean", "ubiquitous"),
                 ("What is the meaning of serendipity", "serendipity"), ("definition of ice cream", "ice cream"), ("what do you mean by ephemeral", "ephemeral"),
                 ("what does the word obfuscate mean", "obfuscate")]:
        check(f"{t!r} -> {w}", define.parse(t) == w, str(define.parse(t)))
    for t in ("what does this mean", "what does it mean", "what is the meaning of life", "define my schedule", "what is a segue", "who is Ada Lovelace",
              "what does that error message mean on my screen right now", "", None):
        check(f"{str(t)[:40]!r} is not a word to look up", define.parse(t) is None)


def d2_answer():
    print("\n=== D2. The answer ===")
    asked = []
    d = Define(DefineConfig(enabled=True), fetch=lambda url, to: asked.append(url) or SEGUE)
    s = d.hear("what's the definition of segue?")
    check("the definitions, tidy, the source named",
          s == "segue: (noun) A smooth transition from one thing to another; (verb) To move & continue without interruption. (Wiktionary)", str(s))
    check("only the word was sent, to Wiktionary", asked == ["https://en.wiktionary.org/api/rest_v1/page/definition/segue"], str(asked))
    check("switched off (a bare Define), nothing is asked", Define().hear("define segue") is None)

    def missing(url, to):
        raise urllib.error.HTTPError(url, 404, "nf", None, None)
    s = Define(DefineConfig(enabled=True), fetch=missing).hear("define blorptastic")
    check("a word with no entry says so, and offers the spelling", s and "couldn't find “blorptastic”" in s and "spell" in s, str(s))
    s = Define(DefineConfig(enabled=True), fetch=lambda u, t: {"fr": []}).hear("define chat")
    check("no English entry says so", s and "English definition" in s, str(s))

    def offline(url, to):
        raise TimeoutError("slow")
    check("offline: nothing said, the model answers as before", Define(DefineConfig(enabled=True), fetch=offline).hear("define segue") is None)
    cut = define.render("x", {"en": [{"partOfSpeech": "noun", "definitions": [{"definition": "word " * 200}]}]}, 100)
    check("a long definition is cut at a word", len(cut) < 130, str(len(cut)))


def d3_facts_and_runtime():
    print("\n=== D3. The facts rule and the runtime ===")
    from src.daemon.facts import topic_of
    check("a definition is not a Wikipedia article title any more", topic_of("what's the definition of segue") is None)
    check("a person still is one", topic_of("who was Ada Lovelace") == "Ada Lovelace")
    import tst_modellast as tm
    rt = tm.rig()
    rt.define = Define(DefineConfig(enabled=True), fetch=lambda u, t: SEGUE)
    used, said = rt.say("What's the definition of segue?")
    check("in the runtime: answered from the dictionary, no model", not used and said and said[-1].endswith("(Wiktionary)"), f"{used} {said}")
    off = tm.rig()
    used, said = off.say("what does ubiquitous mean")
    check("with no dictionary configured it goes where it did", used or said, str(said))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    d1_parse()
    d2_answer()
    d3_facts_and_runtime()
