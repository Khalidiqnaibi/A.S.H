"""
tst_research.py -- "look into X and tell me later".

  R1  Reading a request, and what is not one
  R2  The job: acknowledged, done off the loop, a report that says where each
      part came from, and plainly when nothing was found; no duplicates; a cap
  R3  Telling: at a free moment (beating a joke), or in the evening briefing,
      once; a restart redoes an interrupted job
  R4  The runtime end to end

Run:  PYTHONIOENCODING=utf-8 python tests/tst_research.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon.freemoment import FreeMoments, FreeMomentsConfig, Joke  # noqa: E402
from src.daemon.lookup import Lookup, LookupConfig  # noqa: E402
from src.daemon.research import Research, ResearchActivity, ResearchConfig, parse_request  # noqa: E402
from src.daemon.wondering import Environment  # noqa: E402

check = tw.check
T0 = tw.T0
FREE = dict(idle_s=60.0, free_moment=True)


class Ep:
    def __init__(self, s):
        self.summary = s


class Epi:
    def __init__(self, *s):
        self.eps = [Ep(x) for x in s]

    def retrieve(self, q, top_k=5):
        return self.eps[:top_k]


def folder(files):
    d = tempfile.mkdtemp()
    for rel, text in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
    return d


def rs(lookup=None, episodic=None, **cfg):
    t = [T0]
    r = Research(ResearchConfig.from_dict(cfg), lookup=lookup, episodic=episodic, clock=lambda: t[0],
                 background=False)
    r.t = t
    return r


def r1_reading():
    print("\n=== R1. Reading a request ===")
    for text, want in (("look into Kubernetes", "Kubernetes"), ("research the Raft algorithm", "Raft algorithm"),
                       ("can you look into ZXQ for me later", "ZXQ"),
                       ("find out about Rust lifetimes when you can", "Rust lifetimes"),
                       ("hey ash, read up on WebGPU and tell me later", "WebGPU"),
                       ("dig into the deploy failure", "deploy failure")):
        check(f"{text!r}", parse_request(text) == want, repr(parse_request(text)))
    for text in ("look into it", "what is Kubernetes", "I need to research something", "look at this",
                 "research", "", None, "look into " + "x " * 80):
        check(f"not one: {str(text)[:30]!r}", parse_request(text) is None, repr(parse_request(text)))


def r2_job():
    print("\n=== R2. The job ===")
    d = folder({"notes/plan.md": "We will move the queue to Kubernetes next month.\n",
                "notes/todo.txt": "kubernetes: check the helm chart\n", "notes/other.txt": "nothing\n"})
    lk = Lookup(LookupConfig(memory=False, roots=(d,), web=True),
                fetch=lambda u, t: {"type": "standard", "extract": "Kubernetes is an open-source container "
                                                                   "orchestration system. More."})
    r = rs(lookup=lk, episodic=Epi("User said Kubernetes is hard", "unrelated"))
    say = r.hear("look into Kubernetes")
    check("acknowledged", say == "Okay, I'll look into Kubernetes and tell you what I find.")
    j = r.ready()
    check("done, with a report", j is not None and j.found, str(j))
    rep = j.report
    check("what the web says, credited", "Wikipedia says Kubernetes is an open-source container orchestration system."
          in rep, rep)
    check("where it comes up in the user's files", "It comes up in 2 of your files" in rep and "plan.md" in rep, rep)
    check("how often it was mentioned", "You've mentioned it 1 time before." in rep, rep)
    check("a to-do line is not quoted as a definition", "helm chart" not in rep, rep)
    check("not twice", r.hear("research Kubernetes") == "I'm already looking into Kubernetes.")

    r = rs(lookup=Lookup(LookupConfig(memory=False, files=False, web=False)))
    r.hear("look into ZXQ")
    check("nothing found: said plainly", r.ready().report == "I looked into ZXQ but found nothing beyond what "
                                                             "you've told me." and not r.ready().found)
    r = rs(max_jobs=1)
    r.hear("look into A thing")
    check("capped", "a lot to look into" in r.hear("look into another thing"))
    check("off: not taken", rs(enabled=False).hear("look into Kubernetes") is None)

    class Boom(Lookup):
        def from_web(self, *a, **k):
            raise RuntimeError("down")
    r = rs(lookup=Boom())
    r.hear("look into ZXQ")
    check("a failing source is not fatal", r.ready() is not None and "found nothing" in r.ready().report)

    r = Research(ResearchConfig(), lookup=Lookup(LookupConfig(memory=False, files=False)), clock=lambda: T0)
    r.hear("look into ZXQ")
    for _ in range(100):
        if r.ready(T0) is not None:
            break
        time.sleep(0.02)
    check("done on its own thread", r.ready(T0) is not None)


def r3_telling():
    print("\n=== R3. Telling ===")
    lk = Lookup(LookupConfig(memory=False, files=False, web=True),
                fetch=lambda u, t: {"type": "standard", "extract": "Raft is a consensus algorithm. More."})
    r = rs(lookup=lk)
    m = FreeMoments(FreeMomentsConfig(min_gap_s=0, max_per_hour=9), clock=lambda: r.t[0])
    m.register(ResearchActivity(r))
    m.register(Joke())
    check("nothing ready: a joke", m.pick(Environment(**FREE), T0).activity == "joke")
    r.hear("look into Raft")
    c = m.pick(Environment(**FREE), T0)
    check("ready: the report beats a joke", c.activity == "research" and c.text.startswith(
        "You asked me to look into something. About Raft: Wikipedia says Raft is a consensus algorithm."), c.text)
    m.mark_spoken("research", T0)
    check("told once", r.ready() is None and r.stats["told"] == 1)
    r.hear("look into Paxos")
    check("not in the morning briefing", r.briefing_part("morning", T0) is None)
    check("in the evening one instead", r.briefing_part("evening", T0).startswith("About Paxos"))
    check("and then not again", r.briefing_part("evening", T0) is None)
    r.hear("look into Zab")
    check("an untold report is let go after a few days", r.ready(T0 + 4 * 86400) is None)

    p = os.path.join(tempfile.mkdtemp(), "r.json")
    a = Research(ResearchConfig(), lookup=lk, path=p, clock=lambda: T0, background=False)
    a.jobs["x"] = __import__("src.daemon.research", fromlist=["Job"]).Job("x", "Raft", T0, status="working")
    a.save()
    b = Research(ResearchConfig(), lookup=lk, path=p, clock=lambda: T0, background=False)
    check("an interrupted job is queued again after a restart", b.jobs["x"].status == "queued")


def r4_runtime():
    print("\n=== R4. The runtime ===")
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out = []
    ash = tw.Ash()
    ash.episodic_memory = Epi("ZXQ stands for Zone eXchange Queue")
    per = PeripheryRegistry()
    idle = tw.FakeIdle()
    per.sensors["idle"] = idle
    d = tempfile.mkdtemp()
    rt = AmbientRuntime(
        ash=ash, periphery=per, journal=Journal(tempfile.mkdtemp()),
        gate=AttentionGate(max_offers_per_hour=20), privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, flow_min_burst_s=120.0,
                             moments=True, moments_cfg={"min_gap_s": 0, "mood_check": False},
                             research=True, research_path=os.path.join(d, "r.json"),
                             briefings=True, briefings_path=os.path.join(d, "b.json"),
                             wonder_cfg={"lookup": {"files": False, "web": False}}),
        response_sink=lambda t, m: out.append((t, m)))
    rt.clock = lambda: clock[0]
    rt.research.clock = lambda: clock[0]

    def step(dt, idle_s, ticks=1):
        clock[0] += dt
        idle.value = idle_s
        for _ in range(ticks):
            rt._tick()

    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="look into ZXQ for me", addressed=True,
                          salience=0.9, urgency=0.6))
    rt._tick()
    said = [o[0] for o in out]
    check("acknowledged without the model", said == ["Okay, I'll look into ZXQ and tell you what I find."]
          and ash.calls == [], str(said))
    for _ in range(100):
        if rt.research.ready(clock[0]) is not None:
            break
        time.sleep(0.02)
    for _ in range(15):
        step(10, 0.0)
    step(15, 25.0, ticks=2)
    said = [o[0] for o in out]
    check("told at the free moment", any(s.startswith("You asked me to look into something. About ZXQ: From what "
                                                      "you've told me: ZXQ means Zone eXchange Queue.") for s in said),
          str(said))
    check("status carries research", rt.status()["research"]["told"] == 1)
    rt.submit(SensorEvent(source="console", modality=Modality.TEXT, text="research Paxos", addressed=True,
                          salience=0.9, urgency=0.6))
    rt._tick()
    for _ in range(100):
        if rt.research.ready(clock[0]) is not None:
            break
        time.sleep(0.02)
    check("a report not yet told goes into the evening briefing",
          "I looked into Paxos" in rt.briefings.compose("evening", clock[0]),
          str([(j.topic, j.status, j.report[:40]) for j in rt.research.jobs.values()]) + str([o[0] for o in out][-2:]))


if __name__ == "__main__":
    for fn in (r1_reading, r2_job, r3_telling, r4_runtime):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
