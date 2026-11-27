"""tst_rules27.py -- 27 Nov lines that went to the model (40-160 s each) and are rules now: "what the latest update", "what happened while I was away",
"what did I do yesterday / last session", "latest commits in your repo", "that is not true at all" / "all of what you said was made up",
"you did nothing", "no that was good", "sleep for an hour", "what questions are you wondering about". Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_rules27.py"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import away, honest, quiet, selfmodel, working  # noqa: E402
from src.daemon.activity import Topic  # noqa: E402

check = tw.check


def main():
    print("\n=== Y1. Patterns ===")
    for t in ("what the latest update", "what's your latest update", "what's the latest update for you", "what is the latest update on you"):
        check(f"{t!r} is the update question", selfmodel.parse(t) == "whatsnew", str(selfmodel.parse(t)))
    for t in ("What happened, what while I was away?", "what happened while I was away"):
        check(f"{t!r} is the away digest", away.asks_digest(t))
    for t in ("that is not true at all", "all of what you said was made up", "but it was wrong", "you lied"):
        check(f"{t!r} is a flat denial", honest.is_dispute(t) if hasattr(honest, "is_dispute") else honest._DISPUTE_MORE.match(t) is not None)
    check("sleep for an hour is quiet", quiet.parse("sleep for an hour", 1_800_000_000.0) is not None)
    check("'go to sleep' alone is not a timed quiet", quiet.parse("go to sleep", 1_800_000_000.0) is None)

    print("\n=== Y2. In the runtime: no model ===")
    for line in ("what's your latest update", "that is not true at all", "all of what you said was made up", "you did nothing all the time", "no that was good",
                 "what questions you are wondering about or unresolved on your end"):
        rt = tm.rig()
        rt.say("what's using my memory")                  # something was just said: a dispute is about it
        used, said = rt.say(line)
        check(f"{line!r} -> by rule", not used and said, f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("sleep for an hour")
    check("'sleep for an hour' -> quiet for an hour", not used and rt.gate.muted and said and "quiet" in said[-1].lower(), f"{used} {said}")

    print("\n=== Y3. What did I do yesterday / last session ===")
    rt = tm.rig()
    now = time.time()
    y = time.strftime("%Y-%m-%d", time.localtime(now - 86400))
    d3 = time.strftime("%Y-%m-%d", time.localtime(now - 3 * 86400))
    rt.activity.topics["project:branch"] = Topic(key="project:branch", name="Branch", kind="project", by_day={y: 5400.0})
    rt.activity.topics["project:orbit"] = Topic(key="project:orbit", name="Orbit", kind="project", by_day={d3: 3600.0})
    rt.wonder.kb.store_term("Branch", "im working on my first novel")
    used, said = rt.say("what did I do yesterday")
    check("yesterday, with what was taught about it, no model", not used and said[-1] == "Yesterday: Branch, your first novel (1 hour 30 minutes).", f"{used} {said}")
    rt.activity.topics["project:branch"].by_day = {d3: 100.0}
    used, said = rt.say("what did I do last session")
    check("last session: the latest earlier day with enough in it", not used and "Orbit (1 hour)" in said[-1], f"{used} {said}")
    used, said = rt.say("what did I do yesterday")
    check("nothing long yesterday: said so", said[-1].startswith("I didn't track anything"), str(said))

    print("\n=== Y4. The real commits ===")
    rt = tm.rig()
    used, said = rt.say("what are the latest commits in your repo")
    check("read from git, no model", not used and said and (said[-1].startswith("The latest commits:") or "can't read the git" in said[-1]), f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
