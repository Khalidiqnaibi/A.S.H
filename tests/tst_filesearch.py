"""
tst_filesearch.py -- your own files, found on this computer by rule.

  F1  the ways people ask (and what is not a file search)
  F2  dates: today, yesterday, last week, this month, recently
  F3  finding by name, by text, by kind, by when; newest breaks ties
  F4  by meaning, when the encoder is given ("contract" finds "agreement")
  F5  "what did the lease say about pets?" -- the sentence, read back, no model
  F6  the index: skips hidden and program folders, keeps what did not change, saved and loaded
  F7  "open it" -- only right after, through the executor
  F8  in the runtime: no model call; only for lines meant for ASH

Run:  PYTHONIOENCODING=utf-8 python tests/tst_filesearch.py
"""

import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402
from src.daemon.filesearch import FileSearch, kind_of, when_range, words  # noqa: E402

check = tw.check
NOW = datetime(2026, 11, 12, 15, 0).timestamp()          # a Thursday
DAY = 86400


def make_tree():
    root = tempfile.mkdtemp(prefix="ash-files-")
    docs = os.path.join(root, "Documents")
    os.makedirs(os.path.join(docs, "Work"))
    os.makedirs(os.path.join(docs, ".secret"))
    os.makedirs(os.path.join(docs, "node_modules", "pkg"))

    def put(rel, text, age_days):
        p = os.path.join(docs, rel)
        if p.endswith(".docx"):
            import docx
            d = docx.Document()
            for para in text.split("\n"):
                d.add_paragraph(para)
            d.save(p)
        else:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
        t = NOW - age_days * DAY
        os.utime(p, (t, t))
        return p
    put("Work/client_contract_v2.docx", "Service agreement between ACME and Khalid.\nPayment is due within 30 days.", 8)
    put("Work/meeting_notes.md", "Standup notes. The deadline for the report moved to Friday.", 1)
    put("resume_2026.txt", "Khalid -- software engineer. Experience: robotics, assistants.", 40)
    put("lease.txt", "The tenant pays rent on the 1st. Pets are allowed with a deposit of 200. Smoking is not allowed.", 100)
    put("tax_return_2025.txt", "Tax return for 2025. Refund expected.", 200)
    put("holiday.jpg", "not really a picture", 3)
    put(".secret/passwords.txt", "do not index me", 1)
    put("node_modules/pkg/contract.txt", "a library file named contract", 1)
    return root, docs


def fs_for(docs, embed=None, path=None):
    f = FileSearch(folders=[docs], index_path=path, embed=embed, clock=lambda: NOW)
    f.build()
    return f


def f1_asking():
    print("\n=== F1. The ways people ask ===")
    for said, kind in [("find the contract I was editing last week", "find"), ("Ash, where's my resume?", "find"),
                       ("search my files for invoice", "find"), ("find my tax return pdf", "find"),
                       ("where did I save the meeting notes", "find"), ("what did the lease say about pets?", "about"),
                       ("look for the report I was working on yesterday", "find")]:
        p = FileSearch.parse(said)
        check(f"{said!r} -> {kind}", p is not None and p[0] == kind, str(p))
    for said in ("where's my phone", "find a good restaurant", "where is Omar", "what did Omar say about the trip",
                 "look for my keys", "where did I park"):
        check(f"not a file search: {said!r}", FileSearch.parse(said) is None, str(FileSearch.parse(said)))


def f2_dates():
    print("\n=== F2. Dates ===")
    d = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    r = when_range("last week", NOW)
    check("last week: Monday to Monday", (d(r[0]), d(r[1])) == ("2026-11-02", "2026-11-09"), str((d(r[0]), d(r[1]))))
    r = when_range("yesterday", NOW)
    check("yesterday", (d(r[0]), d(r[1])) == ("2026-11-11", "2026-11-12"))
    check("today, this week, this month, recently all read", all(when_range(w, NOW) for w in ("today", "this week", "this month", "recently")))
    check("no time said: no filter", when_range("the contract", NOW) is None)
    check("kinds", kind_of("my tax return pdf") == "pdf" and kind_of("the slides") == "slides" and kind_of("the contract") is None)
    check("words: plurals and filler off", words("find the contracts I was editing") == ["contract"], str(words("find the contracts I was editing")))


def f3_finding():
    print("\n=== F3. Finding ===")
    root, docs = make_tree()
    f = fs_for(docs)
    names = sorted(e.name for e in f.entries)
    check("hidden and program folders are skipped", "passwords.txt" not in names and "contract.txt" not in names, str(names))
    out = f.find("the contract I was editing last week", NOW)
    check("by name and week", out.startswith("I found client_contract_v2.docx in") and "Want me to open it?" in out, out)
    out = f.find("the resume", NOW)
    check("by name", "resume_2026.txt" in out, out)
    out = f.find("the file about the report deadline", NOW)
    check("by text", "meeting_notes.md" in out, out)
    out = f.find("the contract I was editing yesterday", NOW)
    check("a time that excludes it: not found, says where it looked", out.startswith("I couldn't find") and "Documents" in out, out)
    out = f.find("that picture from this week", NOW)
    check("by kind and time only", "holiday.jpg" in out, out)
    check("when it changed is said in words", "changed on Wednesday" in f.find("the contract", NOW) or "changed on " in f.find("the contract", NOW))


def f4_meaning():
    print("\n=== F4. By meaning ===")
    root, docs = make_tree()
    syn = {"contract": 0, "agreement": 0, "deal": 0, "rent": 1, "lease": 1, "tenant": 1, "cv": 2, "resume": 2, "engineer": 2}

    def embed(texts):
        out = []
        for t in texts:
            v = [0.0] * 4
            for w in words(t):
                v[syn.get(w, 3)] += 1.0 if w in syn else 0.05     # other words barely count
            out.append(v)
        return out
    f = fs_for(docs, embed=embed)
    out = f.find("the agreement with the client", NOW)
    check("'agreement' finds the contract", "client_contract_v2.docx" in out, out)
    out = f.find("my cv", NOW)
    check("'cv' finds the resume", "resume_2026.txt" in out, out)
    check("without an encoder 'cv' finds nothing", fs_for(docs).find("my cv", NOW).startswith("I couldn't find"))


def f5_about():
    print("\n=== F5. What did it say about ===")
    root, docs = make_tree()
    f = fs_for(docs)
    out = f.about("lease", "pets", NOW)
    check("the sentence about pets, read back", out == "In lease.txt: “Pets are allowed with a deposit of 200.” Want me to open it?", out)
    out = f.about("contract", "payment", NOW)
    check("from a Word file", "Payment is due within 30 days." in out, out)
    out = f.about("lease", "parking", NOW)
    check("nothing about it: says so", "doesn't seem to say anything about parking" in out, out)
    check("no such document", f.about("mortgage letter", "rate", NOW).startswith("I couldn't find"))


def f6_index():
    print("\n=== F6. The index ===")
    root, docs = make_tree()
    path = os.path.join(root, "files.json")
    f = fs_for(docs, path=path)
    check("saved", os.path.exists(path))
    again = FileSearch(folders=[docs], index_path=path, clock=lambda: NOW)
    check("loaded without rebuilding", len(again.entries) == len(f.entries) and again.find("the resume", NOW).startswith("I found"))
    calls = []
    f2 = FileSearch(folders=[docs], index_path=path, embed=lambda ts: calls.append(ts) or [[1.0, 0.0]] * len(ts), clock=lambda: NOW)
    f2.build()
    check("unchanged files keep what was read (nothing re-read or re-embedded)", calls == [], f"{len(calls)} embeds")
    f.refresh(NOW, background=False)
    check("refresh at most hourly", f.built_at == NOW)
    capped = FileSearch(folders=[docs], clock=lambda: NOW, max_files=2)
    check("a cap on how many files", capped.build() == 2)


def f7_open():
    print("\n=== F7. Open it ===")
    root, docs = make_tree()
    f = fs_for(docs)
    opened = []
    f.find("the resume", NOW)
    check("'open it' right after opens it", f.reply("open it", lambda p: opened.append(p) or True, NOW) == "Opening resume_2026.txt."
          and opened and opened[0].endswith("resume_2026.txt"))
    check("not again", f.reply("yes", opened.append, NOW) is None and len(opened) == 1)
    f.find("the resume", NOW)
    check("too late", f.reply("yes", opened.append, NOW + 600) is None and len(opened) == 1)
    f.find("the resume", NOW)
    check("anything else is not an answer", f.reply("what's the weather", opened.append, NOW) is None)


def f8_runtime():
    print("\n=== F8. In the runtime ===")
    import tst_effects as te
    from tst_desktop import FakeEx
    root, docs = make_tree()
    rt = te.rig(filesearch=True, filesearch_folders=[docs], desktop=True)
    rt.filesearch.clock = lambda: NOW
    rt.filesearch.build()
    fake = FakeEx()
    rt.desktop.ex = fake
    n = len(rt.ash.calls)
    rt.say("Ash, find the contract I was editing")
    check("found, no model", "client_contract_v2.docx" in rt.out[-1] and len(rt.ash.calls) == n, rt.out[-1])
    rt.say("open it")
    check("'open it' opens that file (fake executor), not the last thing named", fake.opened and fake.opened[-1].endswith("client_contract_v2.docx"),
          str(fake.opened))
    k = len(rt.out)
    rt.say("find the contract", addressed=False)
    check("overheard: not searched", not any("client_contract" in o for o in rt.out[k:]), str(rt.out[k:]))
    n = len(rt.ash.calls)
    rt.say("Ash, what did the lease say about pets")
    check("read back by rule", "Pets are allowed" in rt.out[-1] and len(rt.ash.calls) == n, rt.out[-1])


def f9_embedding_cost():
    print("\n=== F9. The first index does not embed what has nothing to say ===")
    root = tempfile.mkdtemp()
    for i in range(70):
        open(os.path.join(root, f"IMG_{2000 + i}.jpg"), "wb").write(b"JFIF-binary-bytes")
    for i in range(5):
        open(os.path.join(root, f"trip plan {i}.txt"), "w").write("flights and hotels")
    open(os.path.join(root, "annual budget summary.xlsx"), "wb").write(b"PK")
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[1.0, 0.0]] * len(texts)
    f = FileSearch(folders=[root], index_path=None, embed=embed, clock=lambda: NOW)
    f.build()
    n = sum(len(c) for c in calls)
    check("70 photos called IMG_nnnn are not embedded; the text files and the named spreadsheet are", n == 6, f"{n} embedded")
    check("embedded in batches, not one call per file", len(calls) <= 2 and max(len(c) for c in calls) > 1, str([len(c) for c in calls]))
    names = {e.name: e.vec for e in f.entries}
    check("the ones with something to say have a vector", names["trip plan 0.txt"] is not None and names["annual budget summary.xlsx"] is not None)
    check("the rest have none, and are still found by name", names["IMG_2001.jpg"] is None and "IMG_2001.jpg" in f.find("IMG_2001", NOW))
    boom = FileSearch(folders=[root], index_path=None, embed=lambda ts: 1 / 0, clock=lambda: NOW)
    check("an encoder that fails leaves the index usable", boom.build() == 76 and all(e.vec is None for e in boom.entries))


if __name__ == "__main__":
    for fn in (f1_asking, f2_dates, f3_finding, f4_meaning, f5_about, f6_index, f7_open, f8_runtime, f9_embedding_cost):
        fn()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
