"""
tst_lookup.py -- looking something up before asking about it.

  L1  Reading a definition out of text: "X means", "X stands for", an acronym
      spelled out next to it, a glossary line, "X is a"; and what is not one
  L2  What a file says about itself (docstring, README line, comment)
  L3  The sources: memory, files (with caps and skipped folders), the web
      (only for acronyms and quoted terms, never for codenames or files)
  L4  Wondering asks to confirm a guess, and reads yes / no / a correction
  L5  The runtime end to end, and the switches
  L6  Fuzz

Run:  PYTHONIOENCODING=utf-8 python tst_lookup.py
"""

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy  # noqa: E402
from src.daemon.lookup import (Finding, Lookup, LookupConfig, _initials_match,  # noqa: E402
                               definitions_in, self_description)
from src.daemon.wondering import Environment, Knowledge, Wonderer, WonderingConfig  # noqa: E402
from src.senses.base import PeripheryRegistry  # noqa: E402

check = tw.check
T0, HOUR = tw.T0, tw.HOUR
FREE = dict(idle_s=60.0, free_moment=True)


def defs(text, term):
    return [d for d, _ in definitions_in(text, term)]


class Ep:
    def __init__(self, s):
        self.summary = s


class Epi:
    def __init__(self, *summaries):
        self.eps = [Ep(s) for s in summaries]
        self.queries = []

    def retrieve(self, q, top_k=5):
        self.queries.append(q)
        return self.eps[:top_k]


def folder(files):
    d = tempfile.mkdtemp()
    for rel, text in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
    return d


# ---------------------------------------------------------------------------
def l1_definitions():
    print("\n=== L1. Reading a definition ===")
    check("X stands for", defs("ZXQ stands for Zero Extra Queue.", "ZXQ") == ["Zero Extra Queue"])
    check("X means", defs("By the way, ZXQ means the zero-copy queue we built", "ZXQ")
          == ["the zero-copy queue we built"])
    check("spelled out before it", "Zone eXchange Queue" in defs("We use the Zone eXchange Queue (ZXQ) for this.", "ZXQ"))
    check("only the words that spell it are taken",
          defs("Today we shipped the Zone eXchange Queue (ZXQ).", "ZXQ")[:1] == ["Zone eXchange Queue"])
    check("a camel-cased word gives two letters", "JavaScript Object Notation" in defs(
        "data in JavaScript Object Notation (JSON) files", "JSON"))
    check("small words may be skipped", "Bureau of Labor Statistics" in defs(
        "per the Bureau of Labor Statistics (BLS)", "BLS"), str(defs("per the Bureau of Labor Statistics (BLS)", "BLS")))
    check("spelled out after it", "Zone eXchange Queue" in defs("ZXQ (Zone eXchange Queue) is fast", "ZXQ"))
    check("words whose initials do not spell it are not taken", defs("the Zero Extra Queue (ZXQ)", "ZXQ") == [])
    check("nor after it", defs("ZXQ (some other words) is fast", "ZXQ") == [])
    check("letters that do not spell it are not a definition", defs("we like it (ZXQ)", "ZXQ") == [])
    check("a glossary line", defs("- **ZXQ**: the queue the scheduler drains\n", "ZXQ")
          == ["the queue the scheduler drains"])
    check("'X is a' is the weakest form", definitions_in("Orion is a tool for deploys.", "Orion")
          == [("a tool for deploys", 0.5)])
    check("an acronym is matched by case", defs("zxq stands for nothing", "ZXQ") == [])
    check("a word is matched without case", defs("orion means the deploy tool", "Orion") == ["the deploy tool"])
    check("inside another word is not it", defs("XZXQ stands for something", "ZXQ") == [])
    check("nothing in plain text", defs("we talked about ZXQ all day", "ZXQ") == [])
    check("initials: plain", _initials_match(["Zero", "Extra", "Queue"], "ZXQ") is False and
          _initials_match(["Zero", "Xtra", "Queue"], "ZXQ") and _initials_match(["eXtensible", "Markup",
          "Language"], "XML") and not _initials_match(["Zero", "Xtra"], "ZXQ"))


def l2_self():
    print("\n=== L2. What a file says about itself ===")
    py = '"""\nsrc/brain/brain.py\n\nThe central loop that ties memory and drives together.\n\nMore text.\n"""\nimport os\n'
    check("a module docstring, skipping its own name line",
          self_description("x/brain.py", py) == "The central loop that ties memory and drives together",
          str(self_description("x/brain.py", py)))
    check("a README's first line", self_description("README.md", "# Orion\n\nDeploy tool") == "Orion")
    check("a leading comment", self_description("run.sh", "# starts the nightly backup job\necho") ==
          "starts the nightly backup job")
    check("a shebang is not a description", self_description("run.sh", "#!/bin/bash\necho") is None)
    check("nothing to say", self_description("data.json", "{}") is None)


def l3_sources():
    print("\n=== L3. The sources ===")
    epi = Epi("User said ZXQ stands for Zero Extra Queue", "unrelated chatter")
    lk = Lookup(LookupConfig(files=False), episodic=epi)
    f = lk.find("ZXQ", "acronym")
    check("memory: found", f and f.source == "memory" and f.text == "Zero Extra Queue", str(f))

    d = folder({"proj/README.md": "Our Zone eXchange Queue (ZXQ) keeps jobs.\n",
                "proj/node_modules/x.md": "ZXQ stands for Wrong Answer\n",
                "proj/.git/y.md": "ZXQ stands for Also Wrong\n",
                "proj/pic.png": "ZXQ stands for Binary\n",
                "proj/src/brain.py": '"""The central loop."""\n'})
    lk = Lookup(LookupConfig(memory=False, roots=(d,)))
    f = lk.find("ZXQ", "acronym")
    check("files: found, with the file", f and f.source == "files" and f.text == "Zone eXchange Queue"
          and f.where == "README.md", str(f))
    hidden = folder({"proj/node_modules/x.md": "ZXQ stands for Wrong Answer\n",
                     "proj/.git/y.md": "ZXQ stands for Also Wrong\n",
                     "proj/__pycache__/z.md": "ZXQ stands for Cached\n"})
    check("definitions inside skipped folders are not read",
          Lookup(LookupConfig(memory=False, roots=(hidden,))).find("ZXQ", "acronym") is None)
    binary = folder({"pic.png": "ZXQ stands for Binary\n", "blob.bin": "ZXQ stands for Blob\n"})
    check("files that are not text are not read",
          Lookup(LookupConfig(memory=False, roots=(binary,))).find("ZXQ", "acronym") is None)
    f = lk.find("brain.py", "activity")
    check("a file name: what the file says about itself", f and f.text == "The central loop" and f.where == "brain.py",
          str(f))
    check("no roots, no file search", Lookup(LookupConfig(memory=False)).find("ZXQ", "acronym") is None)
    big = folder({"a.txt": "x" * 2000 + " ZXQ stands for Too Big"})
    check("files over the size cap are skipped",
          Lookup(LookupConfig(memory=False, roots=(big,), max_bytes=1000)).find("ZXQ", "acronym") is None)
    many = folder({f"f{i}.txt": "nothing" for i in range(30)} | {"zz.txt": "ZXQ stands for Late"})
    check("the file cap is honoured", Lookup(LookupConfig(memory=False, roots=(many,), max_files=5)).find(
        "ZXQ", "acronym") is None)
    t = [0.0]

    def slow():
        t[0] += 1.0
        return t[0]
    check("the time budget is honoured",
          Lookup(LookupConfig(memory=False, roots=(d,), budget_s=0.5), clock=slow).find("ZXQ", "acronym") is None)

    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return {"type": "standard", "extract": "Kubernetes is an open-source container orchestration system. More."}
    lk = Lookup(LookupConfig(memory=False, files=False, web=True), fetch=fetch)
    f = lk.find("Kubernetes", "quoted")
    check("web: the first sentence's predicate", f and f.source == "web"
          and f.text == "an open-source container orchestration system", str(f))
    check("only the term is sent", calls and calls[0].endswith("/page/summary/Kubernetes"))
    calls.clear()
    check("never for a project codename", lk.find("Orion", "named") is None and calls == [])
    check("never for a file name", lk.find("brain.py", "activity") is None and calls == [])
    check("web off: not called", Lookup(LookupConfig(memory=False, files=False, web=False),
                                       fetch=fetch).find("ZXQ", "acronym") is None)

    def boom(url, timeout):
        raise TimeoutError("slow")
    lk = Lookup(LookupConfig(memory=False, files=False, web=True), fetch=boom)
    check("a web failure is quiet", lk.find("ZXQ", "acronym") is None and lk.stats["web_errors"] == 1)
    lk = Lookup(LookupConfig(memory=False, files=False, web=True),
                fetch=lambda u, t: {"type": "disambiguation", "extract": "ZXQ may refer to:"})
    check("a disambiguation page is no answer", lk.find("ZXQ", "acronym") is None)

    epi = Epi("ZXQ stands for Zero Extra Queue")
    lk = Lookup(LookupConfig(roots=(d,), web=True), episodic=epi, fetch=fetch)
    calls.clear()
    f = lk.find("ZXQ", "acronym")
    check("a sure local answer skips the web", calls == [] and f.source in ("memory", "files"))
    epi = Epi("ZXQ is a thing")
    lk = Lookup(LookupConfig(files=False, min_confidence=0.6), episodic=epi)
    check("below min_confidence is no guess", lk.find("ZXQ", "acronym") is None)
    check("off is off", Lookup(LookupConfig(enabled=False), episodic=Epi("ZXQ stands for A B C")).find("ZXQ") is None)


def wrig(lookup):
    t = [T0]
    w = Wonderer(WonderingConfig(min_gap_s=0), Knowledge(), clock=lambda: t[0])
    w.lookup = lookup
    w.t = t
    return w


def ask(w):
    """Pick the next question, remember how it was said, mark it asked."""
    q = w.next_question(Environment(**FREE), w.t[0])
    w.said = None
    if q is not None:
        w.said = w.render(q, w.t[0])
        w.mark_asked(q.id, w.t[0])
    return q


def l4_wondering():
    print("\n=== L4. Wondering asks to confirm a guess ===")
    lk = Lookup(LookupConfig(files=False), episodic=Epi("ZXQ stands for Zero Extra Queue"))
    w = wrig(lk.for_wonder)
    w.observe("we should move the jobs to ZXQ before Friday")
    w.observe("ZXQ is getting slow")
    q = ask(w)
    check("the question is a guess to confirm", q and w.said == "I think ZXQ means Zero Extra Queue. Is that right?",
          str(w.said))
    r = w.hear("yes", w.t[0] + 5)
    check("yes stores the guess", r.outcome == "answered" and w.kb.terms["zxq"]["definition"] == "Zero Extra Queue",
          str(r))
    check("and is learned", r.learned == ["ZXQ means Zero Extra Queue."])

    w = wrig(lk.for_wonder)
    w.observe("ZXQ again"); w.observe("ZXQ is slow")
    ask(w)
    r = w.hear("no, it's the zero-copy exchange queue", w.t[0] + 5)
    check("a correction is stored instead", r.outcome == "answered"
          and w.kb.terms["zxq"]["definition"] == "the zero-copy exchange queue", str(w.kb.terms))

    w = wrig(lk.for_wonder)
    w.observe("ZXQ again"); w.observe("ZXQ is slow")
    ask(w)
    r = w.hear("no", w.t[0] + 5)
    check("a bare no asks once what it is, then", r.outcome == "followup" and "what does zxq mean" in r.say.lower(),
          r.say)
    r = w.hear("the exchange queue", w.t[0] + 10)
    check("and takes the answer", r.outcome == "answered" and w.kb.terms["zxq"]["definition"] == "the exchange queue")

    w = wrig(lk.for_wonder)
    w.observe("ZXQ again"); w.observe("ZXQ is slow")
    ask(w)
    r = w.hear("right, exactly", w.t[0] + 5)
    check("other ways of saying yes", r.outcome == "answered" and w.kb.terms["zxq"]["definition"] == "Zero Extra Queue")

    nolk = Lookup(LookupConfig(files=False), episodic=Epi("nothing here"))
    w = wrig(nolk.for_wonder)
    w.observe("ZXQ again"); w.observe("ZXQ is slow")
    q = ask(w)
    check("nothing found: the plain question", q and w.said == "What does ZXQ stand for?", str(w.said))
    check("and it was looked up only once", nolk.stats["looked"] == 1)
    ask(w)
    check("(not again on the next turn)", nolk.stats["looked"] == 1)

    def crash(wd):
        raise RuntimeError("bad source")
    w = wrig(crash)
    w.observe("ZXQ again"); w.observe("ZXQ is slow")
    q = ask(w)
    check("a failing lookup still asks the plain question", q and q.question == "What does ZXQ stand for?")

    looked = []

    def eager(wd):
        looked.append(wd.kind)
        return {"guess": "anything", "question": "I think it's anything. Is that right?"}
    w = wrig(eager)
    w.observe("I told Omar about it")
    q = ask(w)
    check("people are never looked up", q and q.question == "Who is Omar to you?" and looked == [], str(looked))

    w = wrig(lk.for_wonder)
    w.observe("ZXQ again"); w.observe("ZXQ is slow")
    q = ask(w)
    q.asked_count = 1
    check("re-asked soon, the guess keeps its capital I",
          w.render(q, w.t[0] + 60).startswith("Still curious: I think ZXQ"), w.render(q, w.t[0] + 60))
    check("re-asked later, it is introduced instead",
          w.render(q, w.t[0] + 86400).startswith("Earlier you said ZXQ. I think ZXQ"), w.render(q, w.t[0] + 86400))

    d = folder({"src/brain.py": '"""The central loop."""\n'})
    flk = Lookup(LookupConfig(memory=False, roots=(d,)))
    w = wrig(flk.for_wonder)
    w.raise_wonder("activity", "brain.py", "What is brain.py for?", "activity:brain py",
                   data={"term": "brain.py", "how": "activity"}, now=w.t[0])
    q = ask(w)
    check("an activity file: ASH read it first", q and "it says it's The central loop" in (w.said or ""),
          str(w.said))
    r = w.hear("yep", w.t[0] + 5)
    check("and a yes keeps it", r.outcome == "answered" and w.kb.terms["brain py"]["definition"] == "The central loop",
          str(w.kb.terms))


def rt_rig(lookup_on, lookup_cfg=None):
    clock = [T0]
    out = []
    ash = tw.Ash()
    ash.episodic_memory = Epi("ZXQ stands for Zero Extra Queue")
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    wc = {"min_gap_s": 0}
    if lookup_cfg is not None:
        wc["lookup"] = lookup_cfg
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             wonder=True, wonder_cfg=wc, lookup=lookup_on,
                             wonder_path=os.path.join(d, "w.json"), knowledge_path=os.path.join(d, "k.json")),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]

    class H:
        pass
    h = H()
    h.rt, h.ash, h.out, h.idle, h.clock = rt, ash, out, idle, clock

    def step(dt, idle_s, ticks=2):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    def say(text, source="console", addressed=True):
        rt.submit(tw.SensorEvent(source=source, modality=tw.Modality.TEXT, text=text, addressed=addressed,
                                 salience=0.9, urgency=0.6))
        rt._tick()

    def free():
        for _ in range(15):
            step(10, 0.0, ticks=1)
        step(15, 25.0)

    h.step, h.say, h.free = step, say, free
    h.said = lambda: [o[0] for o in out]
    return h


def l5_runtime():
    print("\n=== L5. The runtime ===")
    h = rt_rig(True)
    h.say("can you move the jobs to ZXQ")
    h.say("ZXQ is slow today")
    h.free()
    check("ASH asks with its guess", any(s.endswith("I think ZXQ means Zero Extra Queue. Is that right?")
                                         for s in h.said()), str(h.said()))
    h.step(5, 5.0)
    h.say("yes", source="microphone", addressed=False)
    check("and a yes stores it", h.rt.wonder.kb.terms.get("zxq", {}).get("definition") == "Zero Extra Queue",
          str(h.said()))
    check("status carries lookup", h.rt.status()["wondering"].get("lookup", {}).get("found_memory") == 1,
          str(h.rt.status()["wondering"].get("lookup")))

    h = rt_rig(False)
    h.say("can you move the jobs to ZXQ")
    h.say("ZXQ is slow today")
    h.free()
    check("a bare runtime does not look anything up", "What does ZXQ stand for?" in h.said(), str(h.said()))

    h = rt_rig(True, {"enabled": False})
    h.say("can you move the jobs to ZXQ")
    h.say("ZXQ is slow today")
    h.free()
    check("lookup.enabled false in daemon.json turns it off", "What does ZXQ stand for?" in h.said(), str(h.said()))


def l6_fuzz():
    print("\n=== L6. Fuzz ===")
    rng = random.Random(3)
    alphabet = list("ZXQzxq ()[]*:-.\n\"'abcdefgh ") + ["stands for", " means ", " is a ", "(ZXQ)", "Zero", "Extra",
                                                            "Queue", "é", "\x00"]
    bad = 0
    for _ in range(3000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        term = rng.choice(["ZXQ", "zxq", "Orion", "a.b", "(", "*", "", "Zero Extra"])
        try:
            for d, c in definitions_in(s, term):
                if not (0 < c <= 1) or len(d) > 140:
                    bad += 1
            self_description(rng.choice(["a.py", "b.md", "c.sh", "d"]), s)
        except Exception as e:
            bad += 1
            print("   crash:", repr(s), repr(term), e)
    check("3000 random texts: no crash, bounded output", bad == 0, str(bad))


if __name__ == "__main__":
    for fn in (l1_definitions, l2_self, l3_sources, l4_wondering, l5_runtime, l6_fuzz):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
