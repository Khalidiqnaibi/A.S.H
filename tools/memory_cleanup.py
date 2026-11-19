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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEADINGS = {
    "contents", "summary", "contact", "phone", "email", "location", "education", "experience", "projects", "skills",
    "technical skills", "background", "profile", "objective", "references", "languages", "certifications",
    "introduction", "overview", "conclusion", "abstract", "references", "table of contents", "work experience",
}
SENTENCE_STARTERS = {
    "let", "given", "answering", "task", "system", "entertainment", "smile", "turn", "nope", "good", "today",
    "here", "now", "sure", "okay", "yes", "no", "thanks", "hello", "please", "also", "again", "just", "maybe",
}
# Words that were read as names but are readings of the machine, parts of speech or phrases from a model's answer.
JUNK_WORDS = {
    "cpu_percent", "operation", "words", "sentiment", "users", "desktop", "based", "tomorrow", "august", "runner", "ram", "episodic",
    "top recommendation", "also worth considering", "semantic trap", "the semantic trap", "backend", "developer", "software engineer",
    "ai software engineer", "planfollower",
}
# Never dropped for being seen once: the user, ASH, and the places and people said about them.
LONELY_DAYS = 3.0
ALWAYS_KEEP = {"a.s.h", "ash", "khalid", "immortal", "immortal0ne", "khalid iqnaibi", "khalid afif iqnaibi", "hebron", "palestine"}
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
    if re.sub(r"[^a-z0-9]", "", low) == "ash":
        return "ASH's own name"
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


def lonely_entity(e: Dict[str, Any], mentions: int, protect: set) -> str:
    """Why an entity with nothing known about it but its name should go: it was seen in fewer than two sentences (the rule that
    memory_router now applies before it makes one), and it is not someone ASH was told about."""
    a = e.get("attributes") or {}
    if set(a) - {"canonical_name", "last_seen_context", "source", "updated_at"} or e.get("secondary_identifiers"):
        return ""
    low = re.sub(r"\s+", " ", _name(e).lower()).strip(" :-")
    if low in ALWAYS_KEEP or low in protect:
        return ""
    if low in JUNK_WORDS:
        return "a reading of the machine or a phrase, not a thing"
    if mentions < 2:
        # Only what the last few days put there: an older one may be a project or a place that was said once and is still true.
        updated = a.get("updated_at")
        try:
            fresh = updated is not None and (time.time() - float(updated)) <= LONELY_DAYS * 86400.0
        except (TypeError, ValueError):
            fresh = False
        if fresh:
            return "seen once in the last few days, with nothing known about it but its name"
    return ""


_DIRECTIVE_TASTE = re.compile(r"(?:^|\s)(?:keep that in mind|keep this in mind|remember that|remember this|don'?t forget|note that|"
                              r"you know|please)\s*$", re.I)
_FILLER_DEF = re.compile(r"^(?:(?:yeah|yes|yep|oh|well|um+|uh+|so)[,\s]+(?:(?:man|bro|dude|mate|boy)\s+)?(?:is\s+)?(?=\w)|(?:man|bro|dude|mate|boy)\s+is\s+)", re.I)


def plan_knowledge(directory: str) -> Dict[str, Any]:
    """What a wrong answer or a mis-heard sentence left in state/wonder: tastes that are the request to keep one ("shows keep that
    in mind"), a definition that starts with the filler the recogniser lost the comma after ("man is a friend"), and a question closed
    as answered whose answer was never kept (a father's name "answered" with "You")."""
    wdir = os.path.join(directory, "state", "wonder")
    kpath, wpath = os.path.join(wdir, "knowledge.json"), os.path.join(wdir, "wonders.json")
    out: Dict[str, Any] = {"kpath": kpath, "wpath": wpath, "k": None, "w": None, "changes": []}
    if os.path.exists(kpath):
        with open(kpath, "r", encoding="utf-8") as fh:
            k = json.load(fh)
        for uk, u in (k.get("tastes") or {}).items():
            for ik in list((u.get("items") or {})):
                if _DIRECTIVE_TASTE.search(ik):
                    out["changes"].append(f"taste {ik!r} is a request to keep something, not a liking")
                    del u["items"][ik]
        for tk, t in (k.get("terms") or {}).items():
            d = str(t.get("definition") or "")
            nd = _FILLER_DEF.sub("", d)
            if nd != d:
                out["changes"].append(f"definition of {t.get('term', tk)!r}: {d!r} -> {nd!r}")
                t["definition"] = nd
        out["k"] = k
    if os.path.exists(wpath) and out["k"] is not None:
        with open(wpath, "r", encoding="utf-8") as fh:
            w = json.load(fh)
        roles = set((out["k"].get("roles") or {}))
        for key in list((w.get("closed") or {})):
            if key.startswith("role:") and (w["closed"][key] or {}).get("s") == "answered" and key[5:] not in roles:
                out["changes"].append(f"question {key!r} was closed as answered but nobody is on record for it: asked again")
                del w["closed"][key]
                for x in (list(w["wonders"].values()) if isinstance(w.get("wonders"), dict) else (w.get("wonders") or [])):
                    if x.get("key") == key and x.get("status") == "answered":
                        x["status"], x["not_before"], x["closed_at"] = "pending", 0.0, 0.0
        out["w"] = w
    return out


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


_BARE_REMEMBER = re.compile(r"^\s*(?:ash[,\s]+)?(?:please\s+)?remember\s+(?:that|this|it)?\s*[.!]*\s*$", re.I)
_REMEMBER_FACT = re.compile(r"\bremember\s+(?:that|this)\b", re.I)
_BEHAVIOUR = re.compile(r"\b(?:always|never|must|from now on|going forward|as a rule|as a policy|don'?t ever|do not ever)\b", re.I)


def plan_core(directory: str) -> Dict[str, Any]:
    """Core memory entries that a user's "remember that" turned into HARD constraints: one with nothing in it goes, one that is a
    fact about the user stays but is no longer hard (so it is not put in every prompt)."""
    path = os.path.join(directory, "core_memory.json")
    out: Dict[str, Any] = {"path": path, "data": None, "drop": [], "soften": []}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out["data"] = data
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        if not isinstance(v, dict) or v.get("category") not in ("constraint", "standard") or not v.get("hard"):
            continue
        t = str(v.get("text") or "")
        if _BARE_REMEMBER.match(t):
            out["drop"].append(k)
        elif _REMEMBER_FACT.search(t) and not _BEHAVIOUR.search(t) and not str(k).startswith("manual_"):
            out["soften"].append(k)
    return out


def plan(directory: str) -> Dict[str, Any]:
    ep_path = os.path.join(directory, "episodic_memory.jsonl")
    en_path = os.path.join(directory, "entities.json")
    eps = read_jsonl(ep_path) if os.path.exists(ep_path) else []
    from src.senses.mic import is_loop
    drop_eps = [e for e in eps if "_raw" not in e and (own_reply(e) or is_loop(str(e.get("summary", "")).partition(":")[2]))]
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
    counts: Dict[str, int] = {}
    for ep in eps:
        for r in ep.get("related_entities") or []:
            counts[str(r)] = counts.get(str(r), 0) + 1
    protect = set()
    kn = os.path.join(directory, "state", "wonder", "knowledge.json")
    try:
        with open(kn, "r", encoding="utf-8") as fh:
            k = json.load(fh)
        for key, v in (k.get("people") or {}).items():
            protect.add(str(key))
            protect.add(str(v.get("name", "")).lower())
            protect.update(str(b) for b in (v.get("between") or {}))
        protect.update(str(n).lower() for n in (k.get("roles") or {}).values())
        protect.update(str((v or {}).get("name", "")).lower() for v in (k.get("tastes") or {}).values())
    except Exception:
        pass
    drop_ents = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        why = junk_entity(e) or lonely_entity(e, counts.get(str(e.get("entity_id")), 0), protect)
        if why:
            drop_ents.append((e, why))
    return {"core": plan_core(directory), "knowledge": plan_knowledge(directory), "ep_path": ep_path, "en_path": en_path, "episodes": eps, "drop_episodes": drop_eps,
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
    gone = {str(e.get("entity_id")) for e, _ in p["drop_entities"]}
    for e in kept:                                           # an episode must not point at an entity that is gone
        if "_raw" not in e and e.get("related_entities"):
            e["related_entities"] = [r for r in e["related_entities"] if str(r) not in gone]
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
    core = p.get("core") or {}
    if core.get("data") is not None and (core["drop"] or core["soften"]):
        shutil.copy2(core["path"], core["path"] + ".bak-" + stamp)
        for k in core["drop"]:
            core["data"].pop(k, None)
        for k in core["soften"]:
            if k in core["data"]:
                core["data"][k]["hard"] = False
        tmp = core["path"] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(core["data"], fh, indent=2, ensure_ascii=False)
        os.replace(tmp, core["path"])
    kn = p.get("knowledge") or {}
    if kn.get("changes"):
        for path, data in ((kn["kpath"], kn["k"]), (kn["wpath"], kn["w"])):
            if data is not None and os.path.exists(path):
                shutil.copy2(path, path + ".bak-" + stamp)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=1, ensure_ascii=False)
                os.replace(tmp, path)
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
    co = p["core"]
    print(f"\nCore memory: {len(co['drop'])} empty 'remember that' rule(s) to remove, {len(co['soften'])} fact(s) to stop forcing into every prompt")
    for k in co["drop"] + co["soften"]:
        print("   " + k)
    kn = p["knowledge"]
    print(f"\nWhat wrong answers left in what ASH knows: {len(kn['changes'])}")
    for c in kn["changes"]:
        print("   " + c[:150])
    if not a.apply:
        print("\nNothing was changed. Run again with --apply (with ASH stopped) to back up and clean.")
        return 0
    b1, b2 = apply(p)
    print(f"\nDone. Backups: {b1}  {b2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
