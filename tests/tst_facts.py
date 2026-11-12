"""
tst_facts.py -- short factual questions from Wikipedia (opt-in).

  F1  Which questions count: "who/what is <a thing>", and nothing else
  F2  Off by default: nothing is fetched, the model answers
  F3  On: a summary with its source; the first sentences, tidied, bounded
  F4  Disambiguation asks which; a missing page says so; no connection is silent
  F5  Runtime: only when said to ASH, never for talk overheard; settings switch it live
  F6  Fuzz: it never raises and never fetches for a non-question

No network is used: the fetcher is replaced.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_facts.py
"""

import os
import random
import sys
import tempfile
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.facts import Facts, FactsConfig, topic_of  # noqa: E402

check = tw.check


def f1_questions():
    print("\n=== F1. Which questions ===")
    yes = {"Who was Ada Lovelace?": "Ada Lovelace", "what is photosynthesis": "photosynthesis",
           "what is a black hole": "black hole", "tell me about the Roman Empire": "Roman Empire",
           "what does ephemeral mean": "ephemeral", "where is Lisbon": "Lisbon", "who's Alan Turing": "Alan Turing",
           "hey ash, what's a quasar?": "quasar", "Ash, who was Marie Curie": "Marie Curie",
           "can you tell me who is Nelson Mandela": "Nelson Mandela", "define entropy": "entropy"}
    for q, want in yes.items():
        check(f"{q!r} -> {want!r}", topic_of(q) == want, str(topic_of(q)))
    no = ["what is the time", "what time is it", "what is the date", "what is your name", "who are you", "what is this",
          "what's up", "what is going on", "what is 15 times 23", "who is my boss", "what is the weather like",
          "what is the capital of France", "what is the population of Jordan", "what is the best way to learn python",
          "what is wrong", "what is the meaning of life itself, really, tell me please", "who is that", "where is it",
          "where is my phone", "what is the score", "", None, "tell me a joke", "what is on my screen",
          "what is today", "what's my battery at", "who was I talking to", "what is the cpu doing"]
    for q in no:
        check(f"not a topic: {str(q)[:44]!r}", topic_of(q) is None, str(topic_of(q)))


class Fetch:
    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        title = url.rsplit("/", 1)[1]
        v = self.table.get(title)
        if isinstance(v, Exception):
            raise v
        if v is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return v


ADA = {"type": "standard", "extract": "Augusta Ada King, Countess of Lovelace (10 December 1815 – 27 November 1852) "
       "was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical "
       "general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had "
       "applications beyond pure calculation. She published the first algorithm intended to be carried out by such a "
       "machine."}


def f2_off():
    print("\n=== F2. Off by default ===")
    f = Fetch({"Ada_Lovelace": ADA})
    facts = Facts(fetch=f)
    check("the default is off", FactsConfig().enabled is False and not facts.enabled)
    check("a factual question: not answered, not fetched", facts.hear("who was Ada Lovelace") is None and f.calls == [])


def f3_answers():
    print("\n=== F3. Answers ===")
    f = Fetch({"Ada_Lovelace": ADA})
    facts = Facts(FactsConfig(enabled=True), fetch=f)
    say = facts.hear("who was Ada Lovelace?")
    check("a summary, with its source", say is not None and say.endswith("(Wikipedia)"), str(say))
    check("brackets (dates, pronunciations) are taken out", "(10 December" not in say and "Augusta Ada King, Countess of "
          "Lovelace was an English mathematician" in say, say)
    check("two sentences at most, and bounded", say.count(". ") <= 2 and "She was the first to recognise" in say and "She published" not in say and len(say) <= 345, f"{len(say)}: {say}")
    check("it asked for the right page", f.calls == ["https://en.wikipedia.org/api/rest_v1/page/summary/Ada_Lovelace"],
          str(f.calls))
    f2 = Fetch({"Black_hole": {"type": "standard", "extract": "A black hole is a region of spacetime where gravity is so "
                "strong that nothing can escape."}})
    say = Facts(FactsConfig(enabled=True), fetch=f2).hear("what is a black hole")
    check("an article is dropped from the title", f2.calls[0].endswith("/Black_hole") and say.startswith("A black hole is"),
          str(f2.calls) + str(say))
    f3 = Fetch({"Roman_empire": None, "Roman_Empire": {"type": "standard", "extract": "The Roman Empire was the state ruled "
                "by the Romans."}})
    say = Facts(FactsConfig(enabled=True), fetch=f3).hear("tell me about the roman empire")
    check("a second spelling is tried (each word capitalised)", say and say.startswith("The Roman Empire was"),
          f"{f3.calls} {say}")
    long = {"type": "standard", "extract": "X is " + "a very long thing " * 60 + "."}
    say = Facts(FactsConfig(enabled=True, max_chars=100), fetch=Fetch({"X": long})).hear("what is X")
    check("one huge sentence is cut at a word, with an ellipsis", say.endswith("...(Wikipedia)".replace("...(", "... (")) and
          len(say) < 130, say)


def f4_edges():
    print("\n=== F4. Edges ===")
    facts = Facts(FactsConfig(enabled=True), fetch=Fetch({"Mercury": {"type": "disambiguation", "extract": "Mercury may "
                                                                    "refer to:"}}))
    say = facts.hear("what is mercury")
    check("several meanings: asks which", say is not None and "several things" in say and "which one" in say, str(say))
    facts = Facts(FactsConfig(enabled=True), fetch=Fetch({}))
    say = facts.hear("who is Zyxwvut Qqqq")
    check("no page: says so", say == "I couldn't find a Wikipedia page for “Zyxwvut Qqqq”.", str(say))
    check("...and counts it", facts.stats["missing"] == 1)
    facts = Facts(FactsConfig(enabled=True), fetch=Fetch({"Ada_Lovelace": OSError("no route"),
                                                         "Ada_lovelace": OSError("no route")}))
    check("no connection: silent, so the model still answers", facts.hear("who was Ada Lovelace") is None
          and facts.stats["offline"] == 1)
    facts = Facts(FactsConfig(enabled=True), fetch=Fetch({"Ada_Lovelace": TimeoutError("late")}))
    check("too slow: silent", facts.hear("who was Ada Lovelace") is None)
    facts = Facts(FactsConfig(enabled=True), fetch=Fetch({"Ada_Lovelace": urllib.error.HTTPError("u", 503, "x", {}, None)}))
    check("a server error is not 'no such page'", facts.hear("who was Ada Lovelace") is None)
    facts = Facts(FactsConfig(enabled=True), fetch=Fetch({"Ada_Lovelace": {"type": "standard", "extract": ""}}))
    check("an empty summary is 'not found'", "couldn't find" in (facts.hear("who was Ada Lovelace") or ""))


def rig(enabled=True, fetch=None):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    out = []

    class Ash(tw.Ash):
        def think(self, query, image=None, context=""):
            self.calls.append(query)
            r = tw.Resp("The model's own answer.")
            r.pathway = type("P", (), {"value": "slow"})()
            return r
    ash = Ash()
    rt = AmbientRuntime(ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(),
                        privacy=PrivacyPolicy(), config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9,
                                                                       facts=True, facts_cfg={"enabled": enabled}),
                        response_sink=lambda t, m: out.append(t))
    if fetch is not None:
        rt.facts.fetch = fetch

    def say(text, source="console", addressed=True):
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed, salience=0.9,
                              urgency=0.6, ts=time.time()))
        rt._tick()
    rt.say, rt.out, rt.ash_ = say, out, ash
    return rt


def f5_runtime():
    print("\n=== F5. The runtime ===")
    f = Fetch({"Ada_Lovelace": ADA})
    rt = rig(True, f)
    rt.say("who was Ada Lovelace?")
    check("answered from Wikipedia, with no model call", rt.out and rt.out[-1].endswith("(Wikipedia)") and rt.ash_.calls == [],
          str(rt.out) + str(rt.ash_.calls))
    n = len(f.calls)
    rt.say("who was Ada Lovelace", source="microphone", addressed=False)
    check("said in the room, not to ASH: nothing is fetched", len(f.calls) == n)
    rt.say("who was Ada Lovelace", source="screen", addressed=False)
    check("read off the screen: nothing is fetched", len(f.calls) == n)
    rt.say("what time is it?")
    check("a question it does not take goes on as before", len(f.calls) == n)
    off = rig(False, Fetch({"Ada_Lovelace": ADA}))
    off.say("who was Ada Lovelace?")
    check("off: the model answers, nothing fetched", off.facts.fetch.calls == [] and off.ash_.calls)
    from src.ui import settings
    res = settings.change(off, None, "facts.enabled", "true")
    check("the setting switches it on live", res.get("ok") and res.get("live") and off.facts.enabled, str(res))
    off.ash_.calls.clear()
    off.say("who was Ada Lovelace?")
    check("and it answers now", off.out[-1].endswith("(Wikipedia)") and not off.ash_.calls, str(off.out[-1:]))
    row = [r for r in settings.listing(off, None) if r["key"] == "facts.enabled"]
    check("the setting is listed, live, and says the topic goes online", row and row[0]["live"] and row[0]["value"] is True
          and "internet" in row[0]["label"], str(row))
    check("status reports it", off.status()["facts"]["enabled"] is True and off.status()["facts"]["answered"] == 1)
    bare = rig(True)
    bare.facts = None
    bare.say("who was Ada Lovelace?")
    check("no facts part: the model answers", bare.ash_.calls)


def f6_fuzz():
    print("\n=== F6. Fuzz ===")
    rnd = random.Random(7)
    words = ["who", "what", "is", "was", "the", "a", "ash", "tell", "me", "about", "Ada", "Lovelace", "time", "my", "your",
             "weather", "define", "mean", "?", "!", "...", "17", "x", "of", "capital", "where"]
    f = Fetch({})
    facts = Facts(FactsConfig(enabled=True), fetch=f)
    bad = 0
    for _ in range(3000):
        q = " ".join(rnd.choice(words) for _ in range(rnd.randint(0, 11)))
        try:
            facts.hear(q)
            t = topic_of(q)
            if t is None and f.calls:
                bad += 1
            f.calls.clear()
        except Exception:
            bad += 1
    check("3000 random sentences: no error, no fetch without a topic", bad == 0, str(bad))
    for q in ("\x00who is x", "who is " + "a" * 5000, "who is ‮evil", "what is <script>"):
        try:
            facts.hear(q)
            ok = True
        except Exception:
            ok = False
        check(f"odd input survived: {q[:20]!r}", ok)


if __name__ == "__main__":
    for fn in (f1_questions, f2_off, f3_answers, f4_edges, f5_runtime, f6_fuzz):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
