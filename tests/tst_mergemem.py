"""tst_mergemem.py -- two machines' memory files merge by identity: nothing lost, forgotten stays forgotten.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_mergemem.py"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mergemem as M  # noqa: E402

OK = True


def check(n, c, d=""):
    global OK
    print(("  ok   " if c else "  FAIL ") + n + ("" if c else f"  {d}"))
    OK = OK and bool(c)


def ep(i, t, **k):
    return {"episode_id": i, "timestamp": t, "summary": f"s{i}", "importance": 0.5, "access_count": 0, **k}


base = [ep("a", 1), ep("b", 2), ep("c", 3)]
ours = [ep("a", 1, access_count=3), ep("b", 2), ep("c", 3), ep("d", 5)]          # this PC: used a, added d
theirs = [ep("a", 1, access_count=1), ep("c", 3, importance=0.9), ep("e", 4)]     # that PC: dropped b (forgot), changed c, added e
merged = M.merge_jsonl(base, ours, theirs)
m = {r["episode_id"]: r for r in merged}
check("both sides' new episodes are kept", {"d", "e"} <= set(m), str(sorted(m)))
check("an episode one side forgot stays forgotten", "b" not in m, str(sorted(m)))
check("counts take the larger", m["a"]["access_count"] == 3)
check("a field only one side changed is taken", m["c"]["importance"] == 0.9)
check("in time order", [r["episode_id"] for r in merged] == ["a", "c", "e", "d"])
check("changed on one side, deleted on the other: kept (the change is newer news)",
      "x" in {r["episode_id"] for r in M.merge_jsonl([ep("x", 1)], [ep("x", 1, importance=0.8)], [])})

eb = {"1": {"name": "omar", "tags": ["a"], "seen": 2}, "2": {"name": "zed"}}
eo = {"1": {"name": "omar", "tags": ["a", "b"], "seen": 5}, "2": {"name": "zed"}, "3": {"name": "new-here"}}
et = {"1": {"name": "omar", "tags": ["a", "c"], "seen": 4}, "4": {"name": "new-there"}}
r = M.merge_dicts(eb, eo, et)
check("entities: both new ones kept, the one removed there stays removed", set(r) == {"1", "3", "4"}, str(sorted(r)))
check("lists joined, counts larger", r["1"]["tags"] == ["a", "b", "c"] and r["1"]["seen"] == 5, str(r["1"]))

d = tempfile.mkdtemp()


def w(n, rows):
    p = os.path.join(d, n)
    with open(p, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(x) + "\n" for x in rows))
    return p


pb, po, pt = w("b", base), w("o", ours), w("t", theirs)
rc = M.run(pb, po, pt, "episodic_memory.jsonl")
got = [json.loads(line)["episode_id"] for line in open(po, encoding="utf-8") if line.strip()]
check("the command-line form writes the result over 'ours'", rc == 0 and got == ["a", "c", "e", "d"], str(got))
bad = os.path.join(d, "bad.json")
with open(bad, "w") as f:
    f.write("{not json")
check("an unreadable file is left to git (non-zero)", M.run(bad, bad, bad, "entities.json") == 1)

repo = tempfile.mkdtemp()


def g(*a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)


g("init", "-q")
g("config", "user.email", "t@t")
g("config", "user.name", "t")
M.install(repo)
with open(os.path.join(repo, ".gitattributes"), "w") as f:
    f.write("episodic_memory.jsonl merge=ashmem\n")


def put(rows):
    with open(os.path.join(repo, "episodic_memory.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(x) + "\n" for x in rows))


put(base)
g("add", "-A")
g("commit", "-qm", "base")
main = g("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
g("checkout", "-qb", "other")
put(theirs)
g("commit", "-qam", "other")
g("checkout", "-q", main)
put(ours)
g("commit", "-qam", "ours")
res = g("merge", "other", "-m", "m")
ids = [json.loads(line)["episode_id"] for line in open(os.path.join(repo, "episodic_memory.jsonl"), encoding="utf-8") if line.strip()]
check("a real git merge goes through the driver, no conflict", res.returncode == 0 and ids == ["a", "c", "e", "d"],
      res.stdout[-200:] + res.stderr[-200:] + str(ids))
print("\n" + ("ALL PASS" if OK else "FAILED"))
sys.exit(0 if OK else 1)
