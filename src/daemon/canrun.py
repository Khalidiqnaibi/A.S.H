"""
src/daemon/canrun.py

"Can this PC run Cyberpunk?" -- what the game asks for, looked up online, set against what this computer has.

28 Nov: "Can this little stupid old device run GTA 6?" took 52-55 s on the language model, which has no idea what the machine is or what a game needs. machine.py read the specs and
said it couldn't check the requirements. This does check them -- but only when ASH is online (offline it says so and gives the specs, never a guess):

    1. the Steam store's own search finds the game and its page lists "Minimum" and "Recommended" (memory, graphics card, storage)
    2. the computer's memory and graphics card come from machine.specs()
    3. memory is a number against a number; the graphics card is ranked against the one the game names with a small table of common cards
       (roughly -- said as "roughly"); a card the table doesn't know is said, not guessed

Nothing but the game's name is sent (to store.steampowered.com), and only when asked. A game not on Steam is said to be unknown, not made up.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

# Rough graphics scores (a PassMark-like scale, for ranking only). Matched by the model number in the name.
_GPU: List[Tuple[str, int]] = [
    (r"gt\s*710", 400), (r"gt\s*730", 700), (r"gt\s*1030", 1700), (r"gtx\s*650(?!\s*ti)", 1500), (r"gtx\s*650\s*ti", 2000), (r"gtx\s*660(?!\s*ti)", 3000), (r"gtx\s*750\s*ti", 3900),
    (r"gtx\s*750(?!\s*ti)", 3200), (r"gtx\s*760", 3800), (r"gtx\s*770", 4500), (r"gtx\s*780", 5500), (r"gtx\s*950", 4700), (r"gtx\s*960", 5700), (r"gtx\s*970", 8800), (r"gtx\s*980\s*ti", 11000),
    (r"gtx\s*980", 10000), (r"gtx\s*1050\s*ti", 6300), (r"gtx\s*1050", 5300), (r"gtx\s*1630", 5600), (r"gtx\s*1650\s*super", 9100), (r"gtx\s*1650", 7800), (r"gtx\s*1660\s*super", 12700),
    (r"gtx\s*1660\s*ti", 12500), (r"gtx\s*1660", 11500), (r"gtx\s*1060", 10000), (r"gtx\s*1070\s*ti", 14500), (r"gtx\s*1070", 14000), (r"gtx\s*1080\s*ti", 18500), (r"gtx\s*1080", 15500),
    (r"rtx\s*2060\s*super", 14500), (r"rtx\s*2060", 13500), (r"rtx\s*2070\s*super", 17500), (r"rtx\s*2070", 15500), (r"rtx\s*2080\s*ti", 21500), (r"rtx\s*2080", 19000),
    (r"rtx\s*3050", 12500), (r"rtx\s*3060\s*ti", 20000), (r"rtx\s*3060", 17000), (r"rtx\s*3070", 22000), (r"rtx\s*3080", 25000), (r"rtx\s*3090", 27000),
    (r"rtx\s*4060\s*ti", 22500), (r"rtx\s*4060", 19700), (r"rtx\s*4070", 26500), (r"rtx\s*4080", 34500), (r"rtx\s*4090", 38500),
    (r"rx\s*460", 5800), (r"rx\s*470", 8300), (r"rx\s*480", 9700), (r"rx\s*570", 8800), (r"rx\s*580", 9800), (r"rx\s*590", 10300), (r"rx\s*5500", 9500), (r"rx\s*5600", 12500),
    (r"rx\s*5700", 14500), (r"rx\s*6500", 8800), (r"rx\s*6600", 15000), (r"rx\s*6700", 17500), (r"rx\s*6800", 21500), (r"rx\s*7600", 17500), (r"rx\s*7700", 22500),
    (r"r9\s*270", 4400), (r"r9\s*280", 5600), (r"r9\s*290", 7600), (r"r9\s*380", 5500), (r"r9\s*390", 8300), (r"r7\s*260", 3300), (r"r7\s*370", 4700),
    (r"arc\s*a380", 5000), (r"arc\s*a580", 14000), (r"arc\s*a750", 15500), (r"arc\s*a770", 16500),
    (r"uhd\s*(?:graphics\s*)?6\d\d", 1200), (r"hd\s*graphics\s*[3-6]\d\d\d", 700), (r"iris\s*xe", 2000), (r"vega\s*8", 1800),
]
_GPU_RX = [(re.compile(p, re.I), v) for p, v in _GPU]


def gpu_score(name: str) -> Optional[int]:
    """A rough score for a graphics card named in text ('GeForce GTX 1060 6GB'), or None when the table doesn't know it."""
    n = re.sub(r"\s+", " ", (name or "").replace("-", " "))
    for rx, v in _GPU_RX:
        if rx.search(n):
            return v
    return None


def _plain(t: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t or ""))).strip()


def _field(block: str, label: str) -> str:
    m = re.search(r"<strong>\s*" + label + r"\s*:?\s*</strong>\s*:?\s*(.*?)(?:<br\s*/?>|</li>|$)", block or "", re.I | re.S)
    return _plain(m.group(1)) if m else ""


def read_requirements(block: str) -> Dict[str, Any]:
    """{"ram_gb", "gpu": [alternatives], "storage", "cpu"} from one Steam requirements block (minimum or recommended)."""
    ram = _field(block, "Memory")
    gb = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB)", ram, re.I)
    ram_gb = (float(gb.group(1)) / (1024.0 if gb.group(2).upper() == "MB" else 1.0)) if gb else None
    gfx = _field(block, "Graphics")
    alts = [a.strip(" ,.") for a in re.split(r"\s+or\s+|\||/|,\s*(?=[A-Za-z]{2,}\s*\d)", gfx) if a.strip(" ,.")]
    return {"ram_gb": ram_gb, "gpu": alts, "storage": _field(block, "Storage"), "cpu": _field(block, "Processor"), "gfx_text": gfx}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


def pick_game(name: str, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The base game among the search hits: the best name match, a shorter name first (not 'Phantom Liberty', 'REDmod', a bundle)."""
    want = _norm(name)
    best, got = 0.0, None
    for it in items[:8]:
        have = _norm(str(it.get("name", "")))
        r = 1.0 if have == want else SequenceMatcher(None, want, have).ratio()
        if want and want in have:
            r = max(r, 0.9 - 0.01 * (len(have) - len(want)))
        if re.search(r"\b(?:dlc|soundtrack|bundle|pack|demo|expansion|beta|editor|server)\b", have):
            r -= 0.3
        if r > best:
            best, got = r, it
    return got if best >= 0.55 else None


def _get_json(url: str, timeout: float = 8.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ASH-local-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:          # noqa: S310 -- fixed https host
        return json.loads(r.read(2_000_000).decode("utf-8", "ignore"))


def lookup(name: str, get: Callable[[str], Any] = _get_json) -> Optional[Dict[str, Any]]:
    """{"name", "minimum": {...}, "recommended": {...}} from the Steam store, or None if it isn't there. Raises on a network failure."""
    q = urllib.parse.quote(name)
    hits = (get(f"https://store.steampowered.com/api/storesearch/?term={q}&l=english&cc=us") or {}).get("items") or []
    game = pick_game(name, hits)
    if game is None:
        return None
    aid = game["id"]
    d = (get(f"https://store.steampowered.com/api/appdetails?appids={aid}&l=english&cc=us") or {}).get(str(aid), {})
    pc = ((d.get("data") or {}).get("pc_requirements")) or {}
    if not isinstance(pc, dict) or not pc.get("minimum"):
        return None
    return {"name": game.get("name", name), "minimum": read_requirements(pc.get("minimum", "")), "recommended": read_requirements(pc.get("recommended", "")) if pc.get("recommended") else None}


def _have_gpu(spec: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    best_name, best = "", None
    for g in (spec.get("gpus") or []):
        if re.search(r"basic\s+display|remote|virtual|parsec|meta", g, re.I):
            continue
        sc = gpu_score(g)
        if best is None or (sc is not None and sc > best) or not best_name:
            best_name, best = g, sc if sc is not None else best
    return best_name, best


def judge(game: Dict[str, Any], spec: Dict[str, Any]) -> str:
    """What to say: the minimum and recommended memory and graphics card, against this computer's. 'Roughly' for the card; unknown cards are said, not guessed."""
    ram_have = float(spec.get("ram") or 0) / (1024 ** 3)
    gname, gscore = _have_gpu(spec)
    gname_short = re.sub(r"\(R\)|\(TM\)|NVIDIA |AMD |Intel\(R\) ", "", gname).strip()
    name = game["name"]
    lines: List[str] = []
    verdict = "ok"
    mn, rc = game["minimum"], game.get("recommended")

    def vendor(t: str) -> str:
        t = t.lower()
        return "nvidia" if re.search(r"geforce|gtx|rtx|\bgt\s*\d|nvidia", t) else "amd" if re.search(r"radeon|\brx\b|\br[579]\b|amd", t) else "intel" if re.search(r"arc|uhd|iris|intel", t) else ""
    mine = vendor(gname)

    def card(req: Dict[str, Any]) -> Tuple[Optional[int], str]:
        """The alternative the game lists for the same maker as your card (it lists one per maker), else the first it lists that the table knows."""
        scores = [(gpu_score(a), a) for a in req.get("gpu", [])]
        known = [(s, a) for s, a in scores if s is not None]
        same = [(s, a) for s, a in known if mine and vendor(a) == mine]
        if same:
            return same[0]
        if known:
            return known[0]
        return None, (req.get("gfx_text") or "a graphics card")

    need_ram = mn.get("ram_gb")
    if need_ram is not None:
        if ram_have + 0.5 < need_ram:
            verdict = "no"
            lines.append(f"memory: it needs {need_ram:g} GB, you have {ram_have:.0f} GB -- short")
        else:
            lines.append(f"memory: {need_ram:g} GB needed, you have {ram_have:.0f} GB -- fine")
    need_s, need_card = card(mn)
    if need_s is None:
        lines.append(f"graphics: it asks for {need_card[:60]} -- I can't rank that against your {gname_short or 'card'}")
        if verdict == "ok":
            verdict = "unsure"
    elif gscore is None:
        lines.append(f"graphics: it asks for {need_card} -- I can't rank your {gname_short or 'graphics card'}")
        if verdict == "ok":
            verdict = "unsure"
    elif gscore < need_s * 0.9:
        verdict = "no"
        lines.append(f"graphics: it asks for at least {need_card}, you have {gname_short} -- roughly below that")
    else:
        lines.append(f"graphics: {need_card} is the minimum, and your {gname_short} is roughly at or above it")
        if rc:
            rs, rcard = card(rc)
            if rs is not None and gscore < rs * 0.9 and verdict == "ok":
                verdict = "low"
    if rc and rc.get("ram_gb") and ram_have + 0.5 < rc["ram_gb"] and verdict == "ok":
        verdict = "low"
    head = {"no": f"Probably not: {name} asks for more than this computer has.",
            "low": f"{name} should start, on low settings -- you're over the minimum but under the recommended.",
            "ok": f"{name} should run" + (" well: you're at or above the recommended." if rc else ": you meet the minimum."),
            "unsure": f"I can't say for sure about {name}."}[verdict]
    return head + " " + "; ".join(lines) + ". (From the Steam page; the card comparison is rough.)"
