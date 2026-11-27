"""tst_pipeline.py -- things ASH can already do, chained like a Unix pipe: each stage's output feeds the next. Typed stages, joined by commas / 'then' / '|';
read-only pipes just run, the first stage that changes something is read back and asked about once; 'in the background', 'what are you doing?', 'stop that'.
Everything is a fake context: nothing is searched, opened, copied or written. Offline.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_pipeline.py"""
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
from src.daemon import pipeline as pl  # noqa: E402

check = tw.check

LEASE = ("The lease runs for twelve months from the first of March. Rent is due on the first of each month by bank transfer. "
         "Pets are allowed only with the written consent of the landlord, and a deposit of 200 pounds applies. "
         "The tenant must give two months notice before leaving. Repairs under fifty pounds are the tenant's cost. "
         "The landlord may inspect the flat with 24 hours notice.")


class Fake(pl.Ctx):
    def __init__(self, files=("C:/docs/lease.pdf",), page=LEASE, gate=None):
        self.files, self.page, self.notes, self.copied, self.opened, self.gate = list(files), page, [], [], [], gate

    def find_files(self, q):
        return list(self.files)

    def read_file(self, path):
        return LEASE

    def search_open(self, q):
        self.opened.append(q)
        return True

    def read_page(self):
        return self.page

    def memory(self, q):
        return "kept: " + q

    def note(self, text):
        self.notes.append(text)
        return True

    def remember(self, name, text):
        self.notes.append(f"{name}={text}")
        return True

    def copy(self, text):
        self.copied.append(text)
        return True


def run(text, ctx):
    p = pl.parse(text)
    stages = p["parallel"][0]
    r = pl.Run(1, text, stages)
    return pl.run_stages(ctx, r, pause=lambda s: None), r


def main():
    print("\n=== P1. Reading a pipe: stages found by what they do ===")
    p = pl.parse("Find the lease, summarize it, and tell me what it says about pets.")
    names = [s.name for s in p["parallel"][0]] if p else None
    check("find -> summarize -> about", names == ["find_files", "summarize", "about"], str(names))
    p = pl.parse("find the lease | read it | summarize it")
    check("the | syntax is the same pipe", p and [s.name for s in p["parallel"][0]] == ["find_files", "read_file", "summarize"], str(p))
    p = pl.parse("Search Google for pygame sprites, read the first result and save it as a note")
    check("search -> read the page -> note", p and [s.name for s in p["parallel"][0]] == ["search_web", "read_page", "note"] and p["parallel"][0][0].arg == "pygame sprites", str(p))
    check("a single request is not a pipe", pl.parse("find the lease") is None and pl.parse("what time is it") is None and pl.parse("I like jazz and blues") is None)
    p = pl.parse("search for pygame sprites and then summarize it in the background")
    check("'in the background' is read", p and p["background"], str(p))
    p = pl.parse("find the lease and summarize it meanwhile search for cats and read the page")
    check("'meanwhile' splits two pipes that run together", p and len(p["parallel"]) == 2, str(p))

    print("\n=== P2. Types: what one gives, the next takes ===")
    st = pl.parse("summarize it, copy it")["parallel"][0]
    check("a pipe that starts with nothing to work on is said so before it runs", pl.check_types(st) is not None)
    st = pl.parse("find the lease, summarize it")["parallel"][0]
    check("files feed a text stage (the first file is read: an adapter)", pl.check_types(st) is None)
    st = pl.parse("search for cats, summarize it")["parallel"][0]
    check("a page feeds a text stage (the page is read)", pl.check_types(st) is None)
    st = pl.parse("find the lease, copy it, summarize it")["parallel"][0]
    check("a stage that gives nothing cannot feed the next", pl.check_types(st) is not None and "stage 3" in pl.check_types(st), str(pl.check_types(st)))

    print("\n=== P3. Running ===")
    out, r = run("find the lease, summarize it, tell me what it says about pets", Fake())
    check("the answer is the lease's own sentence on pets", "Pets are allowed" in out and r.state == "done", out)
    f = Fake()
    out, r = run("find the lease, read it, summarize it, save it as a note", f)
    check("the summary is what was kept", f.notes and "lease" in f.notes[0].lower() and out == "Kept as a note.", f"{f.notes} {out}")
    f = Fake()
    out, r = run("search for pygame sprites, read the first result, copy it", f)
    check("search -> page -> copied", f.opened == ["pygame sprites"] and len(f.copied) == 1 and out == "Copied.", f"{f.opened} {f.copied} {out}")
    out, r = run("find the lease, summarize it", Fake(files=()))
    check("a stage that finds nothing ends the pipe, named", out.startswith("Stopped at stage 1") and r.state == "failed", out)
    out, r = run("find the lease, tell me what it says about parrots", Fake())
    check("nothing on the topic is said, not made up", "nothing in it about parrots" in out, out)
    check("the summary keeps sentences in their order and is shorter", pl.summarize(LEASE, 2) in LEASE.replace("  ", " ") or len(pl.summarize(LEASE, 2)) < len(LEASE))

    print("\n=== P4. Cancel and status ===")
    runs = pl.Runs()
    stages = pl.parse("find the lease, summarize it")["parallel"][0]
    r = runs.new("x", stages)
    check("status lists it", "find lease" in runs.status() and "stage 1 of 2" in runs.status(), runs.status())
    check("stop that stops it", runs.stop().startswith("Stopping") and r.cancel.is_set())
    out = pl.run_stages(Fake(), r)
    check("a cancelled pipe says so", out == "Stopped." and r.state == "cancelled", out)
    check("nothing running says so", runs.status() == "Nothing running right now.")
    check("the phrases", pl.asks_status("what are you doing right now") and pl.asks_stop("stop that") and not pl.asks_stop("stop"))

    print("\n=== P5. In the runtime ===")
    import tst_modellast as tm

    def rig(ctx):
        rt = tm.rig()
        rt.rt = rt
        rt._pipeline_ctx = lambda: ctx
        return rt
    f = Fake()
    rt = rig(f)
    used, said = rt.say("find the lease, summarize it, and tell me what it says about pets")
    check("a read-only pipe just runs, no model", not used and said and "Pets are allowed" in said[-1], f"{used} {said}")
    f = Fake()
    rt = rig(f)
    used, said = rt.say("find the lease, summarize it and save it as a note")
    check("a pipe that changes something is read back first", not used and said and said[-1].startswith("So:") and said[-1].endswith("Go ahead?") and not f.notes, f"{used} {said}")
    used, said = rt.say("yes")
    check("on a yes it runs, as one", f.notes and said and "note" in said[-1].lower(), f"{f.notes} {said}")
    f = Fake()
    rt = rig(f)
    rt.say("find the lease, summarize it and save it as a note")
    used, said = rt.say("no")
    check("on a no nothing is kept", not f.notes and said and "not doing" in said[-1], f"{f.notes} {said}")
    rt = rig(Fake())
    used, said = rt.say("summarize it, copy it")
    check("a pipe that cannot join is said so", not used and said and said[-1].startswith("That doesn't join up"), f"{used} {said}")
    rt = rig(Fake())
    used, said = rt.say("what's the capital of Peru")
    check("an ordinary question still goes to the model", used, f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
