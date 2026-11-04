"""
tst_activity.py -- what the user does, kept like a conversation.

Everything runs on a fake clock, with fake entity and episodic memory, so a
week of working takes milliseconds.

  A1  Stretches: a glance is nothing, a stretch is a sentence, away ends it
  A2  The conversation-like record: phrasing, links, the per-hour cap
  A3  Connecting: an entity, what was said, a sitting, recurrence -- and a
      one-off connected to nothing is let go
  A4  Asking: only connected + unexplained, only once seen again, never twice,
      never about what the user already explained
  A5  Answers: stored like every other answer, never read as one later
  A6  Privacy: withheld windows teach nothing, keywords not text, off means off
  A7  Persistence: save, restart, corrupt file, caps
  A8  The runtime end to end, and the control command

Run:  PYTHONIOENCODING=utf-8 python tst_activity.py
"""

import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

from src.daemon import (AmbientConfig, AmbientRuntime, AttentionGate, Journal,  # noqa: E402
                        PrivacyPolicy)
from src.daemon.activity import (ActivityConfig, ActivityMemory, keywords, norm)  # noqa: E402
from src.daemon.wondering import (Environment, Knowledge, Wonderer,  # noqa: E402
                                  WonderingConfig)
from src.senses.base import Modality, PeripheryRegistry, SensorEvent  # noqa: E402

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


T0 = time.mktime((2026, 3, 10, 9, 0, 0, 0, 0, -1))
MIN, HOUR, DAY = 60.0, 3600.0, 86400.0


# ---- fakes --------------------------------------------------------------
class FakeEntities:
    """Enough of an EntityManager: ingest, and a store that lists them."""

    def __init__(self, known=()):
        self.items = []
        for spec in known:
            name, etype = spec[0], spec[1]
            source = spec[2] if len(spec) > 2 else "chat"
            self.items.append(NS(entity_id=f"id-{norm(name)}", entity_type=etype,
                                 primary_identifiers={"name": norm(name)},
                                 attributes={"canonical_name": name, "source": source}))
        self.store = NS(all=lambda: list(self.items))
        self.ingested = []

    def ingest(self, d):
        self.ingested.append(d)
        e = NS(entity_id=f"new-{len(self.items)}", entity_type=d["entity_type"],
               primary_identifiers=d["primary_identifiers"], attributes=d["attributes"])
        self.items.append(e)
        return e


class FakeEpisodic:
    def __init__(self, said=()):
        self.added = []
        eps = [NS(summary=s, event_type="interaction", timestamp=T0 - HOUR) for s in said]
        self.store = NS(all=lambda: eps + [NS(summary=a["summary"], event_type=a["event_type"],
                                              timestamp=T0) for a in self.added])

    def add_episode(self, summary, event_type, related_entities=None, importance=0.5):
        self.added.append({"summary": summary, "event_type": event_type,
                           "related": list(related_entities or []), "importance": importance})
        return NS(episode_id=str(len(self.added)))


def win(subject, project="", process="Code.exe", **meta):
    return SensorEvent(source="active_window", modality=Modality.TEXT,
                       text=f"switched to {subject}",
                       meta={"subject": subject, "project": project, "process": process, **meta})


def reading(text, title="", **meta):
    return SensorEvent(source="screenshot", modality=Modality.TEXT, text="read the screen",
                       meta={"on_screen": text, "title": title, **meta})


def rig(known=(), said=(), wonder=True, **cfg):
    t = [T0]
    ent, epi = FakeEntities(known), FakeEpisodic(said)
    w = kb = None
    if wonder:
        kb = Knowledge()
        w = Wonderer(WonderingConfig.from_dict({"min_gap_s": 0.0}), kb, clock=lambda: t[0])
    kw = dict(min_stretch_s=180.0, episode_min_s=300.0)
    kw.update(cfg)
    a = ActivityMemory(ActivityConfig.from_dict(kw), entities=ent, episodic=epi, wonder=w,
                       knowledge=kb, user="Sam", clock=lambda: t[0])
    a.t, a.ent, a.epi, a.w, a.kb = t, ent, epi, w, kb

    def spend(subject, minutes, project="", process="Code.exe", then=0, **meta):
        """Be in a window for `minutes`, then (optionally) idle `then` minutes."""
        a.observe(win(subject, project, process, **meta), t[0])
        t[0] += minutes * MIN
        a.tick(t[0], idle_s=0.0)
        if then:
            t[0] += then * MIN
    a.spend = spend

    def close():
        a.tick(t[0] + 1, idle_s=10 ** 6)
    a.close = close
    return a


def pending(a):
    return [r for r in a.w.pending()]


# ============================================================================
print("\n=== A1. Stretches ===")
a = rig()
a.spend("brain.py", 1)
a.observe(win("system2.py"), a.t[0])
check("a minute in a window is a glance, not an activity", a.stats["stretches"] == 0)
a.spend("brain.py", 6, "A.S.H")
a.observe(win("other.py"), a.t[0])
check("six minutes is a stretch", a.stats["stretches"] == 1, str(a.stats))

a = rig()
a.spend("brain.py", 10, "A.S.H")
a.t[0] += 20 * MIN
a.tick(a.t[0], idle_s=20 * MIN)
check("going away ends the stretch where it last had the user",
      a.stats["stretches"] == 1 and a.topics["subject:brain py"].seconds == 600.0,
      str(a.topics.get("subject:brain py")))

a = rig()
a.observe(win("brain.py", "A.S.H"), a.t[0])
a.t[0] += 3 * HOUR
for _ in range(7):
    a.tick(a.t[0], idle_s=0.0)
    a.t[0] += 0
check("an unbroken stretch is recorded in pieces, not held for hours", a.stats["stretches"] >= 1,
      str(a.stats))

for label, ev in [("a blocked window", win("1Password - Vault", blocked=True)),
                  ("a window with no subject (withheld, or a bare app title)", win("")),
                  ("ASH's own window", win("A.S.H")),
                  ("a generic title", win("New Tab")),
                  ("a title too short to mean anything", win("a"))]:
    a = rig()
    a.observe(ev, a.t[0])
    a.t[0] += 20 * MIN
    a.tick(a.t[0], idle_s=0.0)
    a.close()
    check(f"{label} teaches nothing", not a.topics and a.stats["stretches"] == 0, str(a.topics))

a = rig()
a.observe(SensorEvent(source="microphone", modality=Modality.TEXT, text="hello", meta={}), T0)
a.observe(SensorEvent(source="system", modality=Modality.TEXT, text="cpu", meta={}), T0)
check("events from other sensors are ignored", a.stats["ignored"] == 0 and not a.topics)

a = rig(enabled=False)
a.spend("brain.py", 20, "A.S.H")
a.close()
check("switched off, it does nothing at all", not a.topics and not a.epi.added)

# ============================================================================
print("\n=== A2. The conversation-like record ===")
a = rig()
a.spend("brain.py", 12, "A.S.H", process="Code.exe")
a.observe(win("other.py"), a.t[0])
check("a real stretch is one sentence in episodic memory", len(a.epi.added) == 1, str(a.epi.added))
ep = a.epi.added[0]
check("...phrased as something the user did, by name",
      ep["summary"] == "Sam spent 12 minutes on brain.py in A.S.H (VS Code)", ep["summary"])
check("...as an activity, not an interaction", ep["event_type"] == "activity")

a = rig()
a.spend("brain.py", 4, "A.S.H")
a.observe(win("x.py"), a.t[0])
check("a stretch shorter than the sentence threshold is a topic but not an episode",
      not a.epi.added and a.stats["stretches"] == 1)

a = rig()
a.spend("notes.txt", 1, process="notepad.exe")
a.spend("a-long-doc", 5, process="notepad.exe")
a.observe(win("x.py"), a.t[0])
check("with no project the sentence names only what it has",
      any(e["summary"] == "Sam spent 5 minutes on a-long-doc (Notepad)" for e in a.epi.added),
      str([e["summary"] for e in a.epi.added]))

a = rig(max_episodes_per_hour=3)
for i in range(8):
    a.spend(f"file{i}.py", 6, "Proj")
a.close()
check("episodes are capped per hour, so a busy day is not a flood", len(a.epi.added) == 3,
      str(len(a.epi.added)))
a.t[0] += 2 * HOUR
a.spend("later.py", 6, "Proj")
a.close()
check("...and the allowance comes back", len(a.epi.added) == 4, str(len(a.epi.added)))

a = rig(remember=False)
a.spend("brain.py", 12, "A.S.H")
a.observe(win("x.py"), a.t[0])
check("remember=false: no episodes, topics still kept", not a.epi.added and a.stats["stretches"] == 1)

# ============================================================================
print("\n=== A3. Connecting ===")
a = rig(known=[("A.S.H", "SPEAKER_SELF"), ("Neuromorphic Device", "topic")])
a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
t = a.topics["project:a s h"]
check("a project that IS an entity ASH already has is recognised as that entity",
      t.remembered and any(l["type"] == "named" and l["to"] == "A.S.H" for l in t.links),
      str(t.links))
check("...and becomes an entity of its own, marked as made from activity",
      any(d["attributes"].get("source") == "activity" and d["entity_type"] == "project"
          for d in a.ent.ingested), str(a.ent.ingested))
check("...and the episode is linked to what it connects to",
      "id-a s h" in a.epi.added[0]["related"], str(a.epi.added[0]))

a = rig(known=[("Galadriel", "person"), ("Open", "proper"), ("Mon", "date"), ("Sam", "SPEAKER_USER"),
               ("2026", "cardinal"), ("Opera", "org")],
        said=["Sam: I was talking to Galadriel about Opera and the open question on Sam 2026"])
a.spend("Notes about Galadriel and Open and Sam 2026 on Opera", 6, "Whatever")
a.observe(win("x.py"), a.t[0])
t = a.topics["subject:notes about galadriel and open and sam 2026 on opera"]
check("an interface word, a browser, a date, a number and the user itself are not entities worth linking",
      [l["to"] for l in t.links if l["type"] == "entity"] == ["Galadriel"], str(t.links))

# Entity memory also holds names minted from ASH's own replies. Treating those
# as already known made ASH ask "I only know the name, what is it?" about a
# document title it had named itself.
a = rig(known=[("Quantum Garden", "org", "ash")], said=["Sam: nothing relevant was said here"])
a.spend("Quantum Garden Plans", 6)
a.observe(win("x.py"), a.t[0])
check("an entity only ASH has ever said is not something the user already knows",
      not any(l["type"] in ("entity", "named")
              for l in a.topics["subject:quantum garden plans"].links),
      str(a.topics["subject:quantum garden plans"].links))
a = rig(known=[("Quantum Garden", "org", "chat")],
        said=["Sam: I joined the Quantum Garden project yesterday"])
a.spend("Quantum Garden Plans", 6)
a.observe(win("x.py"), a.t[0])
check("...while one the user said is", any(l["type"] == "entity"
                                           for l in a.topics["subject:quantum garden plans"].links))
a = rig(known=[("Quantum Garden", "org", "chat")], said=["Sam: hello there"])
a.spend("Quantum Garden Plans", 6)
a.observe(win("x.py"), a.t[0])
check("...and a 'chat' entity that no turn of the user's names is not either",
      not any(l["type"] in ("entity", "named")
              for l in a.topics["subject:quantum garden plans"].links))

a = rig(said=["Sam: I have been reading about neuromorphic hardware all week",
              "A.S.H: Here is a general approach to neuromorphic hardware"])
a.spend("Neuromorphic Hardware Survey - arXiv", 6)
a.observe(win("x.py"), a.t[0])
t = next(iter(a.topics.values()))
check("a topic that shares a distinctive word with what the user said is connected to it",
      t.remembered and any(l["type"] == "conversation" for l in t.links), str(t.links))
a = rig(said=["A.S.H: Here is a general approach to quantum thermodynamics"])
a.spend("Quantum Thermodynamics Lecture", 6)
a.observe(win("x.py"), a.t[0])
check("...but only the user's own words count, not what ASH said",
      not any(l["type"] == "conversation" for l in next(iter(a.topics.values())).links))

a = rig()
a.spend("A Page About Gemini", 6, "Google Gemini", process="opera.exe")
a.spend("Another Page", 6, "Google Gemini", process="opera.exe")
a.observe(win("x.py"), a.t[0])
check("in a browser the second title field is a website, not a project: it is never a topic",
      "project:google gemini" not in a.topics and not pending(a), str(list(a.topics)))
check("...nor does it appear in the sentence",
      all("Google Gemini" not in e["summary"] for e in a.epi.added),
      str([e["summary"] for e in a.epi.added]))

a = rig()
for _ in range(3):
    a.spend("Thesis Draft", 4, then=1)
    a.spend("Task Manager", 4, then=1)
check("an operating-system window is not something the user is working on",
      "subject:task manager" not in a.topics and not any(t.together for t in a.topics.values()),
      str(list(a.topics)))

a = rig()
a.spend("Random Page Nobody Cares About", 6)
a.observe(win("x.py"), a.t[0])
t = a.topics["subject:random page nobody cares about"]
check("a one-off connected to nothing is let go: no links, no entity",
      not t.links and not t.remembered and not a.ent.ingested, str(t.links))
check("...though it still earned its sentence in the record", len(a.epi.added) == 1)

a = rig()
for i in range(3):
    a.spend("Design Doc", 4, then=1)
    a.spend("Budget", 4, then=1)
t = a.topics["subject:design doc"]
check("topics that keep turning up in the same sitting are connected to each other",
      t.together.get("subject:budget", 0) >= 2 and any(l["type"] == "with" for l in t.links),
      str(t.together))

a = rig()
for i in range(4):
    a.spend("brain.py", 4, "A.S.H", then=1)
    a.spend("system2.py", 4, "A.S.H", then=1)
check("...but a project is not connected to its own files, nor one file to another in it",
      not any(t.together for t in a.topics.values()),
      str({k: t.together for k, t in a.topics.items() if t.together}))

a = rig()
for d in range(3):
    a.spend("Thesis Draft", 4, then=0)
    a.observe(win("x.py"), a.t[0])
    a.t[0] += DAY
t = a.topics["subject:thesis draft"]
check("something that keeps coming back over days recurs", any(l["type"] == "recurs" for l in t.links)
      and t.remembered, str(t.links))

# ============================================================================
print("\n=== A4. Asking ===")
a = rig(known=[("A.S.H", "SPEAKER_SELF")])
a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
check("seen once, it is not asked about yet", not pending(a), str(pending(a)))
a.spend("system2.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
ws = pending(a)
check("seen again and connected: one question is raised",
      len(ws) == 1 and ws[0]["kind"] == "activity", str(ws))
check("...that says why it is asking: it knows the name and nothing else",
      "A.S.H" in ws[0]["question"] and "only know the name" in ws[0]["question"], ws[0]["question"])
a.spend("planner.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
check("...and not again for the same thing", len(pending(a)) == 1)

a = rig(known=[("A.S.H", "SPEAKER_SELF")])
a.kb.store_term("A.S.H", "my assistant project")
for _ in range(3):
    a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
check("something the user has already explained is never asked about", not pending(a))

a = rig(known=[("A.S.H", "SPEAKER_SELF")], ask=False)
for _ in range(3):
    a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
check("ask=false: connected topics are saved but nothing is asked",
      not pending(a) and a.topics["project:a s h"].remembered)

a = rig()
for _ in range(4):
    a.spend("Some Page", 6, then=0)
a.observe(win("x.py"), a.t[0])
check("recurrence alone does not make a page worth a question", not pending(a), str(pending(a)))
a = rig()
for _ in range(4):
    a.spend("main.py", 6, "Orbit")
a.observe(win("x.py"), a.t[0])
check("...but a project that keeps coming back is", any(w["subject"] == "Orbit" for w in pending(a)),
      str(pending(a)))

a = rig(said=["Sam: thinking about neuromorphic hardware today"])
for _ in range(2):
    a.spend("Neuromorphic Hardware Survey", 6)
a.observe(win("x.py"), a.t[0])
q = [w for w in pending(a) if "Neuromorphic" in w["subject"]]
check("a topic connected to what the user said is asked about in those terms",
      q and "talked about" in q[0]["question"] and "neuromorphic" in q[0]["question"], str(pending(a)))

a = rig()
for _ in range(3):
    a.spend("Design Doc", 4, then=1)
    a.spend("Budget", 4, then=1)
a.observe(win("x.py"), a.t[0])
check("topics seen together are asked about as a pair",
      any("switching between" in w["question"] for w in pending(a)), str([w["question"] for w in pending(a)]))

a = rig(known=[("A.S.H", "SPEAKER_SELF")], wonder=False)
for _ in range(3):
    a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
check("with no wonderer it still remembers and connects, it just cannot ask",
      a.topics["project:a s h"].remembered)

# ============================================================================
print("\n=== A5. Answers ===")
a = rig(known=[("A.S.H", "SPEAKER_SELF")])
for _ in range(2):
    a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
w = a.w.next_question(Environment(idle_s=60.0, free_moment=True), a.t[0])
check("at a free moment the question is the next thing asked", w is not None and w.kind == "activity")
a.w.mark_asked(w.id, a.t[0])
check("the question reads as ASH noticing, not as quoting the user",
      "Earlier you said" not in a.w.render(w, a.t[0] + 3600), a.w.render(w, a.t[0] + 3600))
rep = a.w.hear("it's my personal assistant project", a.t[0] + 5, addressed=True)
check("the user's reply is stored like any other answer",
      a.kb.term_known("A.S.H") and "assistant" in a.kb.terms["a s h"]["definition"],
      str(a.kb.terms))
check("...and the wonder is closed for good", not pending(a))
for _ in range(2):
    a.spend("planner.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
check("...so it is never asked again", not pending(a))

a = rig(known=[("A.S.H", "SPEAKER_SELF")])
for _ in range(2):
    a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
w = a.w.next_question(Environment(idle_s=60.0, free_moment=True), a.t[0])
a.w.mark_asked(w.id, a.t[0])
a.w.hear("let me think", a.t[0] + 2, addressed=True)
rep = a.w.hear("open brain.py please", a.t[0] + 4 * HOUR, addressed=True)
check("a later sentence that merely names it is not taken for the answer",
      not a.kb.term_known("A.S.H"), str(a.kb.terms))

a = rig(known=[("A.S.H", "SPEAKER_SELF")])
for _ in range(2):
    a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
w = a.w.next_question(Environment(idle_s=60.0, free_moment=True), a.t[0])
a.w.mark_asked(w.id, a.t[0])
rep = a.w.hear("hm", a.t[0] + 5, addressed=True)
check("an unclear reply earns a follow-up in the right words",
      rep.say and "what is A.S.H" in rep.say, str(rep.say))

# ============================================================================
print("\n=== A6. Privacy ===")
a = rig()
secret = "my password is hunter2 and the account number is 4417 1234 5678 9010"
a.spend("Quarterly Planning", 6, "Corp")
a.observe(reading(secret + " budget forecast forecast forecast", "Quarterly Planning - Corp"), a.t[0])
a.observe(win("x.py"), a.t[0])
blob = json.dumps({k: vars(t) for k, t in a.topics.items()})
check("only keywords are kept from what was read, never the text: no digits, no sentence",
      "hunter2" not in blob and "4417" not in blob and "account number" not in blob
      and "my password is" not in blob and "forecast" in blob, blob[:200])

a = rig(keep_terms=False)
a.spend("Quarterly Planning", 6, "Corp")
a.observe(reading("budget forecast forecast", "Quarterly Planning"), a.t[0])
a.observe(win("x.py"), a.t[0])
check("keep_terms=false keeps no keywords at all",
      not any(t.terms for t in a.topics.values()))

a = rig()
a.observe(win("Quarterly Planning", "Corp"), a.t[0])
a.observe(reading("something unrelated entirely", "Some Other Window"), a.t[0] + 60)
a.t[0] += 6 * MIN
a.observe(win("x.py"), a.t[0])
check("a reading taken from a different window is not attributed to this one",
      not a.topics["subject:quarterly planning"].terms, str(a.topics["subject:quarterly planning"].terms))

check("keywords are distinctive words, most repeated first",
      keywords("energy energy function function function the and neuromorphic", 3)[:2]
      == ["function", "energy"], str(keywords("energy energy function function function the and neuromorphic", 3)))
check("...and short or common words are not keywords", keywords("the and is a an of to") == [])

d = tempfile.mkdtemp()
p = os.path.join(d, "activity", "topics.json")
t = [T0]
a = ActivityMemory(ActivityConfig(), path=p, entities=FakeEntities(), episodic=FakeEpisodic(),
                   user="Sam", clock=lambda: t[0])
a.observe(win("Quarterly Planning", "Corp"), t[0])
t[0] += 6 * MIN
a.observe(win("x.py"), t[0])
a.save(force=True)
check("the topic file is written under its own directory, created as needed", os.path.exists(p))

# ============================================================================
print("\n=== A7. Persistence and bounds ===")
a2 = ActivityMemory(ActivityConfig(), path=p, clock=lambda: t[0])
check("a restart remembers what was learned",
      "subject:quarterly planning" in a2.topics and a2.topics["subject:quarterly planning"].stretches == 1)
open(p, "w").write("{ this is not json")
a3 = ActivityMemory(ActivityConfig(), path=p, clock=lambda: t[0])
check("a corrupt file starts empty and is set aside, not fatal",
      not a3.topics and os.path.exists(p + ".bad"))

a = rig(max_topics=10)
for i in range(40):
    a.spend(f"Doc number {i}", 4, "P")
a.observe(win("done.py"), a.t[0])
check("the topic list is bounded", len(a.topics) <= 10, str(len(a.topics)))

a = rig(known=[("A.S.H", "SPEAKER_SELF")], max_topics=3)
a.spend("brain.py", 6, "A.S.H")
for i in range(10):
    a.spend(f"Doc number {i}", 4, "P")
a.observe(win("done.py"), a.t[0])
check("...and what was connected outlives what was not", "project:a s h" in a.topics)

a = rig(known=[("A.S.H", "SPEAKER_SELF")])
a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
a.forget()
check("forget clears it", not a.topics and "0 topics" in a.report())
a = rig(known=[("A.S.H", "SPEAKER_SELF")])
a.spend("brain.py", 6, "A.S.H")
a.observe(win("x.py"), a.t[0])
r = a.report()
check("the report says what is connected and why", "A.S.H" in r and "named" in r, r)

# ============================================================================
print("\n=== A8. The runtime, end to end ===")


class FakeBrain:
    user = "Sam"

    def __init__(self):
        self.homeostasis = NS(drives=NS(curiosity=0.5, urgency=0.3, fatigue=0.2, caution=0.3,
                                        social=0.5, effort_budget=0.6))

    def observe(self, **kw):
        return NS(percepts={"novelty": 0.3})

    def status(self):
        return {"cycles": 0}


class FakeAsh:
    def __init__(self, ent, epi):
        self.brain = FakeBrain()
        self.entity_memory, self.episodic_memory = ent, epi

    def think(self, q, image=None, context=""):
        return NS(text="ok", pathway=NS(value="slow"))


class FakeIdle:
    name = "idle"
    value = 0.0

    def due(self, now):
        return False

    def seconds_idle(self):
        return self.value

    def status(self):
        return {"name": "idle"}

    def close(self):
        pass


d = tempfile.mkdtemp()
clock = [T0]
ent, epi = FakeEntities([("A.S.H", "SPEAKER_SELF")]), FakeEpisodic()
per = PeripheryRegistry()
idle = FakeIdle()
per.sensors["idle"] = idle
rt = AmbientRuntime(
    ash=FakeAsh(ent, epi), periphery=per, journal=Journal(tempfile.mkdtemp()),
    gate=AttentionGate(), privacy=PrivacyPolicy(),
    config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                         wonder=True, wonder_cfg={"min_gap_s": 0.0},
                         wonder_path=os.path.join(d, "w.json"),
                         knowledge_path=os.path.join(d, "k.json"),
                         activity=True, activity_cfg={"min_stretch_s": 180.0},
                         activity_path=os.path.join(d, "activity", "topics.json")),
    response_sink=lambda text, meta: None)
rt.clock = lambda: clock[0]
check("the runtime builds it when asked", rt.activity is not None and rt.wonder is not None)
check("...and shares the wonderer's knowledge, so what was told is not asked",
      rt.activity.kb is rt.wonder.kb)

for _ in range(2):
    rt.submit(win("brain.py", "A.S.H"))
    rt._tick()
    clock[0] += 6 * MIN
    idle.value = 0.0
    rt._tick()
rt.submit(win("elsewhere.py"))
rt._tick()
check("window events through the loop become stretches", rt.activity.stats["stretches"] >= 2,
      str(rt.activity.stats))
check("...a sentence each in episodic memory", len(epi.added) >= 2, str(epi.added))
check("...and a question about the connected project", any(w["kind"] == "activity"
                                                           for w in rt.wonder.pending()),
      str(rt.wonder.pending()))
check("the status shows it", rt.status()["activity"]["connected"] >= 1, str(rt.status().get("activity")))
rt.stop()
check("stopping writes the file", os.path.exists(os.path.join(d, "activity", "topics.json")))

rt2 = AmbientRuntime(
    ash=FakeAsh(FakeEntities(), FakeEpisodic()), periphery=PeripheryRegistry(),
    journal=Journal(tempfile.mkdtemp()), gate=AttentionGate(), privacy=PrivacyPolicy(),
    config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9),
    response_sink=lambda text, meta: None)
check("a bare runtime has none of it, as it always did", rt2.activity is None
      and rt2.status()["activity"] is None)

from src.daemon.service import DEFAULT_CONFIG, ControlServer  # noqa: E402

check("it is in the default config, on, and visible there",
      DEFAULT_CONFIG["activity"]["enabled"] is True and DEFAULT_CONFIG["activity"]["ask"] is True)
cs = ControlServer.__new__(ControlServer)
cs.runtime = rt
out = cs.dispatch("activity")
check("the control command reports it", out.get("ok") and "topics" in out.get("report", ""), str(out)[:120])
cs.runtime = rt2
out = cs.dispatch("activity")
check("...and says so when it is off", out.get("ok") is False and "not enabled" in out.get("error", ""))

print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
sys.exit(0 if OK else 1)
