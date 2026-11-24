"""
mergemem.py -- a git merge driver for ASH's memory files, so two PCs that both learn never overwrite each other.

`entities.json` (a dict of entities) and `episodic_memory.jsonl` (one episode per line) are rewritten whole by every run; a plain git merge of two
machines' copies conflicts on nearly every line. This merges them by identity instead:

  * nothing either side added is lost (union by key / episode_id);
  * a record both sides changed is merged field by field (lists joined, counts take the larger, other values take the side that changed);
  * a record one side DELETED (a 'forget that') stays deleted when the other side did not change it -- a scrubbed memory is not brought back.

Install once per clone:  python tools/mergemem.py --install     (sets merge.ashmem.* in .git/config; .gitattributes already names it)
Git calls:  python tools/mergemem.py %O %A %B %P     (base, ours = result file, theirs, path)
"""
import json
import os
import subprocess
import sys
from typing import Any, List, Optional

_MISSING = object()


def _ordered(a: dict, b: dict) -> List[str]:
    return list(a.keys()) + [k for k in b.keys() if k not in a]


def _merge_value(base: Any, ours: Any, theirs: Any) -> Any:
    if ours == theirs:
        return ours
    if ours == base:
        return theirs
    if theirs == base:
        return ours
    if isinstance(ours, dict) and isinstance(theirs, dict):
        return merge_dicts(base if isinstance(base, dict) else {}, ours, theirs)
    if isinstance(ours, list) and isinstance(theirs, list):
        out = list(ours)
        for x in theirs:
            if x not in out:
                out.append(x)
        return out
    if isinstance(ours, (int, float)) and isinstance(theirs, (int, float)) and not isinstance(ours, bool) and not isinstance(theirs, bool):
        return max(ours, theirs)
    return ours                      # both changed a plain value: this machine's wins


def _merge_key(base: dict, ours: dict, theirs: dict, k: str) -> Any:
    o, t, b = ours.get(k, _MISSING), theirs.get(k, _MISSING), base.get(k, _MISSING)
    if o is _MISSING and t is _MISSING:
        return _MISSING
    if o is _MISSING:                # gone here: deleted (when there it is as it was) or never here (new there / changed there)
        return _MISSING if (b is not _MISSING and t == b) else t
    if t is _MISSING:
        return _MISSING if (b is not _MISSING and o == b) else o
    return _merge_value(None if b is _MISSING else b, o, t)


def merge_dicts(base: Optional[dict], ours: dict, theirs: dict) -> dict:
    """entities.json: {id: record} (and any nested dict)."""
    base = base or {}
    out = {}
    for k in _ordered(ours, theirs):
        v = _merge_key(base, ours, theirs, k)
        if v is not _MISSING:
            out[k] = v
    return out


def merge_jsonl(base: List[dict], ours: List[dict], theirs: List[dict], key: str = "episode_id") -> List[dict]:
    """episodic_memory.jsonl: a record per line, keyed by episode_id, in time order."""
    def index(rows):
        return {r.get(key): r for r in rows if isinstance(r, dict) and r.get(key)}
    merged = merge_dicts(index(base), index(ours), index(theirs))
    return sorted(merged.values(), key=lambda r: (float(r.get("timestamp", 0) or 0), str(r.get(key))))


def _read(path: str, jsonl: bool):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return [] if jsonl else {}
    if not text.strip():
        return [] if jsonl else {}
    if jsonl:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        return rows
    return json.loads(text)


def run(base_p: str, ours_p: str, theirs_p: str, name: str) -> int:
    jsonl = name.lower().endswith(".jsonl")
    try:
        base, ours, theirs = _read(base_p, jsonl), _read(ours_p, jsonl), _read(theirs_p, jsonl)
        if jsonl:
            text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merge_jsonl(base, ours, theirs))
        else:
            text = json.dumps(merge_dicts(base, ours, theirs), indent=2, ensure_ascii=False)
        with open(ours_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return 0
    except Exception as e:           # unreadable input: leave the conflict to git rather than write a guess
        sys.stderr.write(f"mergemem: {e}\n")
        return 1


def install(repo: str = ".") -> None:
    py = sys.executable.replace("\\", "/")
    me = os.path.abspath(__file__).replace("\\", "/")
    for k, v in (("merge.ashmem.name", "ASH memory merge by identity"), ("merge.ashmem.driver", f'"{py}" "{me}" %O %A %B %P')):
        subprocess.run(["git", "config", k, v], cwd=repo, check=True)
    print("merge driver 'ashmem' installed for this clone")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--install":
        install()
        sys.exit(0)
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    sys.exit(run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
