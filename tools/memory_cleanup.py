"""
tools/memory_cleanup.py

Clear out what ASH's own free-form replies left in memory.

ASH used to save its own answers as episodes. Anything the model made up
("Khalid's CV: Phone +212 XXX-XXXXXX, Location: Morocco") became a memory,
was found again months later and said back as fact. That no longer happens
(see MemoryEngine.commit_turn); this removes what is already there:

  * episodes whose text is ASH's own reply ("A.S.H: ...") and that had no
    tool behind them -- they are marked [tool] when there was one
  * entities that are not things at all: headings from a document that was
    on screen ("Contents", "Summary", "Technical Skills", "Phone"), words that
    only start a sentence, and made-up placeholder addresses and numbers (an
    address or number you really gave ASH is kept)

By default it only SHOWS what it would remove. With --apply it first copies
episodic_memory.jsonl and entities.json to *.bak-<time>, then rewrites them.
Run it with ASH stopped: it is a running daemon's files.

    python tools/memory_cleanup.py                 # look
    python tools/memory_cleanup.py --apply         # back up, then clean
    python tools/memory_cleanup.py --dir D:\\A.S.H  # another install
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

HEADINGS = {
    "contents", "summary", "contact", "phone", "email", "location", "education", "experience", "projects", "skills",
    "technical skills", "background", "profile", "objective", "references", "languages", "certifications",
    "introduction", "overview", "conclusion", "abstract", "references", "table of contents", "work experience",
}
SENTENCE_STARTERS = {
    "let", "given", "answering", "task", "system", "entertainment", "smile", "turn", "nope", "good", "today",
    "here", "now", "sure", "okay", "yes", "no", "thanks", "hello", "please", "also", "again", "just", "maybe",
}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d|\+\d+\s*X{3,}")


def _name(e: Dict[str, Any]) -> str:
    a = e.get("attributes") or {}
    return str(a.get("canonical_name") or (e.get("primary_identifiers") or {}).get("name") or "").strip()


def junk_entity(e: Dict[str, Any]) -> str:
    """Why this entity should go ('' if it should stay)."""
    n = _name(e)
    low = re.sub(r"\s+", " ", n.lower()).strip(" :-")
    if not low:
        return "no name"
    if "\n" in n:
        return "text with a line break in it"
    # An address or number you really gave ASH is a fact worth keeping; only
    # the placeholders a model makes up are removed.
    if _EMAIL.search(n) and re.search(r"@(?:email|example|domain|mail|yourmail|test)\.(?:com|org|net)\b", low):
        return "a made-up e-mail address"
    if _PHONE.search(n) and re.search(r"x{3,}|0{5,}|1234567", low):
        return "a made-up phone number"
    if low in HEADINGS:
        return "a document heading"
    if low in SENTENCE_STARTERS:
        return "a word that only starts a sentence"
    return ""


def own_reply(ep: Dict[str, Any]) -> bool:
    s = str(ep.get("summary") or "")
    return s.startswith("A.S.H:") and "[tool]" not in s[:200]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"_raw": line})
    return out


def plan(directory: str) -> Dict[str, Any]:
    ep_path = os.path.join(directory, "episodic_memory.jsonl")
    en_path = os.path.join(directory, "entities.json")
    eps = read_jsonl(ep_path) if os.path.exists(ep_path) else []
    drop_eps = [e for e in eps if "_raw" not in e and own_reply(e)]
    ents: List[Dict[str, Any]] = []
    shape = None
    if os.path.exists(en_path):
        with open(en_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            ents, shape = raw, "list"
        elif isinstance(raw, dict) and isinstance(raw.get("entities"), list):
            ents, shape = raw["entities"], "dict"
        elif isinstance(raw, dict):
            ents, shape = list(raw.values()), "values"
    drop_ents = [(e, junk_entity(e)) for e in ents if isinstance(e, dict) and junk_entity(e)]
    return {"ep_path": ep_path, "en_path": en_path, "episodes": eps, "drop_episodes": drop_eps,
            "entities": ents, "drop_entities": drop_ents, "shape": shape}


def apply(p: Dict[str, Any]) -> Tuple[str, str]:
    """Back up, then rewrite. Returns the two backup paths."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    b1 = p["ep_path"] + ".bak-" + stamp
    b2 = p["en_path"] + ".bak-" + stamp
    if os.path.exists(p["ep_path"]):
        shutil.copy2(p["ep_path"], b1)
    if os.path.exists(p["en_path"]):
        shutil.copy2(p["en_path"], b2)
    drop_ids = {id(e) for e in p["drop_episodes"]}
    kept = [e for e in p["episodes"] if id(e) not in drop_ids]
    tmp = p["ep_path"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for e in kept:
            fh.write(e["_raw"] + "\n" if "_raw" in e else json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, p["ep_path"])
    if p["shape"]:
        bad = {id(e) for e, _ in p["drop_entities"]}
        keep_e = [e for e in p["entities"] if id(e) not in bad]
        gone_ids = {str(e.get("entity_id")) for e, _ in p["drop_entities"]}
        with open(p["en_path"], "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if p["shape"] == "list":
            raw = keep_e
        elif p["shape"] == "dict":
            raw["entities"] = keep_e
        else:
            raw = {k: v for k, v in raw.items() if not (isinstance(v, dict) and str(v.get("entity_id")) in gone_ids)}
        tmp = p["en_path"] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, p["en_path"])
    return b1, b2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Remove what ASH's own replies left in memory.")
    ap.add_argument("--dir", default=os.getcwd(), help="the ASH folder (default: here)")
    ap.add_argument("--apply", action="store_true", help="do it (after making backups)")
    a = ap.parse_args(argv)
    p = plan(a.dir)
    print(f"{len(p['episodes'])} episodes, {len(p['entities'])} entities in {a.dir}")
    print(f"\nEpisodes that are ASH's own words, with no tool behind them: {len(p['drop_episodes'])}")
    for e in p["drop_episodes"][:8]:
        print("   " + time.strftime("%m-%d %H:%M", time.localtime(float(e.get("timestamp") or 0))) + "  "
              + str(e.get("summary", ""))[:90].replace("\n", " "))
    if len(p["drop_episodes"]) > 8:
        print(f"   ... and {len(p['drop_episodes']) - 8} more")
    print(f"\nEntities that are not things: {len(p['drop_entities'])}")
    for e, why in p["drop_entities"][:15]:
        print(f"   {_name(e)[:40]!r:44} {why}")
    if not a.apply:
        print("\nNothing was changed. Run again with --apply (with ASH stopped) to back up and clean.")
        return 0
    b1, b2 = apply(p)
    print(f"\nDone. Backups: {b1}  {b2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
