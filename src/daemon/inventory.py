"""
src/daemon/inventory.py

What is installed on this computer: apps and games -- so "what games do I have", "do I have Blender", "open Cyberpunk" and "open Minecraft" are answered from
the machine, not from a model's guess.

Read once a day (and on "rescan my apps") from places Windows and the launchers already keep, on this computer only:
    the Start menu         the shortcuts in both Start Menu "Programs" folders (apps)
    Steam                  steamapps/appmanifest_*.acf in every library folder (games; opened with steam://rungameid/<id>)
    Epic Games             ProgramData/Epic/EpicGamesLauncher/Data/Manifests/*.item (games)
Nothing is sent anywhere. Kept in state/apps.json.

    "what games do I have"            the games, by name
    "what apps do I have installed"   a short list, and how many
    "do I have Blender" / "is Steam installed"
    "open Cyberpunk" / "launch Minecraft"   only for a name the desktop rules don't already know, matched closely (never a guess)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.inventory")

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|now)[,\s]+)*(?:please\s+)?"
_END = r"\s*[.!?]*\s*$"
_WHAT = re.compile(_LEAD + r"(?:what|which|list)\s+(?:(?:are\s+)?(?:the\s+)?)?(?P<k>games|apps|applications|programs|software)\s+(?:do\s+i\s+have|have\s+i\s+got|are\s+(?:installed|on\s+(?:this|my)\s+(?:computer|pc|laptop|machine))|"
                   r"i\s+(?:have|got)(?:\s+installed)?)(?:\s+(?:installed|on\s+(?:this|my)\s+(?:computer|pc|laptop|machine)))?" + _END, re.I)
_WHAT2 = re.compile(_LEAD + r"(?:list|show\s+me|tell\s+me)\s+(?:all\s+)?(?:my\s+|the\s+)?(?:installed\s+)?(?P<k>games|apps|applications|programs)(?:\s+(?:installed|i\s+have))?" + _END, re.I)
_HAVE = re.compile(_LEAD + r"(?:do\s+i\s+have|have\s+i\s+got|is)\s+(?P<n>[\w][\w '\-.:&]{1,40}?)\s+(?:installed|on\s+(?:this|my)\s+(?:computer|pc|laptop|machine))(?:\s+installed)?" + _END, re.I)
_HAVE2 = re.compile(_LEAD + r"do\s+i\s+have\s+(?P<n>[\w][\w '\-.:&]{1,40}?)" + _END, re.I)
_OPEN = re.compile(_LEAD + r"(?:open|launch|start|run|play)\s+(?:the\s+game\s+|the\s+app\s+)?(?P<n>[\w][\w '\-.:&]{1,40}?)" + _END, re.I)
_SWITCH = re.compile(_LEAD + r"(?:switch|go|jump|change)\s+(?:back\s+)?to\s+(?:the\s+)?(?P<n>[\w][\w '\-.:&]{1,40}?)(?:\s+(?:app|window|desktop))?" + _END, re.I)
_RESCAN = re.compile(_LEAD + r"(?:re-?scan|refresh|update)\s+(?:my\s+|the\s+)?(?:installed\s+)?(?:apps|games|programs|applications|app\s+list)" + _END, re.I)
_NOISE = re.compile(r"\b(?:uninstall|readme|help|documentation|manual|license|release notes|website|web site|visit|support)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower().replace("&", " and ")).strip()


def _scan_start_menu() -> List[Dict[str, str]]:
    out = []
    roots = [os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), r"Microsoft\Windows\Start Menu\Programs"),
             os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs")]
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*.lnk"), recursive=True):
            name = os.path.splitext(os.path.basename(path))[0]
            if _NOISE.search(name) or not name.strip():
                continue
            out.append({"name": name, "kind": "app", "open": path})
    return out


def _steam_libraries() -> List[str]:
    libs = []
    for base in (os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Steam"),
                 os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Steam")):
        vdf = os.path.join(base, "steamapps", "libraryfolders.vdf")
        if os.path.isdir(os.path.join(base, "steamapps")):
            libs.append(os.path.join(base, "steamapps"))
        try:
            with open(vdf, encoding="utf-8", errors="replace") as fh:
                for m in re.finditer(r'"path"\s+"([^"]+)"', fh.read()):
                    p = os.path.join(m.group(1).replace("\\\\", "\\"), "steamapps")
                    if os.path.isdir(p) and p not in libs:
                        libs.append(p)
        except OSError:
            pass
    return libs


def _scan_steam() -> List[Dict[str, str]]:
    out = []
    for lib in _steam_libraries():
        for path in glob.glob(os.path.join(lib, "appmanifest_*.acf")):
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                appid = re.search(r'"appid"\s+"(\d+)"', text)
                name = re.search(r'"name"\s+"([^"]+)"', text)
                if appid and name and not re.search(r"steamworks|redistributable|proton|steam linux runtime", name.group(1), re.I):
                    out.append({"name": name.group(1), "kind": "game", "open": f"steam://rungameid/{appid.group(1)}"})
            except OSError:
                continue
    return out


def _scan_epic() -> List[Dict[str, str]]:
    out = []
    d = os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), r"Epic\EpicGamesLauncher\Data\Manifests")
    for path in glob.glob(os.path.join(d, "*.item")):
        try:
            with open(path, encoding="utf-8") as fh:
                j = json.load(fh)
            name = j.get("DisplayName")
            if name:
                out.append({"name": str(name), "kind": "game", "open": f"com.epicgames.launcher://apps/{j.get('AppName', '')}?action=launch&silent=true"})
        except (OSError, ValueError):
            continue
    return out


def _scan_store_apps() -> List[Dict[str, str]]:
    """Apps installed from the Microsoft Store (Claude, WhatsApp, ...): the Start menu's own list. They have no .lnk file in the Start Menu folders --
    28 Nov: 'switch to claude' went to the model for 68 s although Claude is installed (as 'Claude'), because nothing listed it."""
    import subprocess
    import sys
    if not sys.platform.startswith("win"):
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-StartApps | ForEach-Object { $_.Name + '|' + $_.AppID }"],
                             capture_output=True, text=True, timeout=25).stdout
    except Exception:
        return []
    rows = []
    for ln in out.splitlines():
        if "|" not in ln:
            continue
        name, appid = ln.split("|", 1)
        name, appid = name.strip(), appid.strip()
        if name and appid and "!" in appid and not _NOISE.search(name):
            rows.append({"name": name, "kind": "app", "open": "shell:AppsFolder\\" + appid})
    return rows


def scan_all() -> List[Dict[str, str]]:
    seen, rows = set(), []
    for fn in (_scan_steam, _scan_epic, _scan_store_apps, _scan_start_menu):            # a game found by its launcher wins over its Start menu shortcut
        try:
            for r in fn():
                k = _norm(r["name"])
                if k and k not in seen:
                    seen.add(k)
                    rows.append(r)
        except Exception:
            logger.exception("Scanning %s failed", fn.__name__)
    return rows


class Inventory:
    def __init__(self, path: Optional[str] = None, scan: Callable[[], List[Dict[str, str]]] = scan_all,
                 opener: Optional[Callable[[str], bool]] = None, clock: Callable[[], float] = time.time, max_age_s: float = 86400.0):
        self.path = path
        self._scan = scan
        self._open = opener or self._startfile
        self.clock = clock
        self.max_age_s = max_age_s
        self.items: List[Dict[str, str]] = []
        self.at = 0.0
        self._load()

    @staticmethod
    def _startfile(target: str) -> bool:
        try:
            os.startfile(target)
            return True
        except Exception:
            logger.exception("Could not open %s", target)
            return False

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.items, self.at = list(d.get("items") or []), float(d.get("at") or 0.0)
        except (OSError, ValueError):
            pass

    def refresh(self, force: bool = False, now: Optional[float] = None):
        now = self.clock() if now is None else now
        if not force and self.items and now - self.at < self.max_age_s:
            return
        self.items, self.at = self._scan(), now
        if self.path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as fh:
                    json.dump({"at": now, "items": self.items}, fh, ensure_ascii=False)
            except OSError:
                pass

    def find(self, name: str, cutoff: float = 0.82) -> Optional[Dict[str, str]]:
        want = _norm(name)
        if not want:
            return None
        best, got = 0.0, None
        for it in self.items:
            have = _norm(it["name"])
            r = 1.0 if have == want else 0.93 if (want in have.split() or have in want.split()) and min(len(want), len(have)) >= 4 else \
                0.9 if want in have and len(want) >= 5 else SequenceMatcher(None, want, have).ratio()
            if r > best:
                best, got = r, it
        return got if best >= cutoff else None

    def hear(self, text: str, known_app: Callable[[str], bool] = lambda n: False, now: Optional[float] = None,
             switch: Optional[Callable[[Dict[str, str]], Optional[bool]]] = None) -> Optional[str]:
        """What to say back for an installed-apps question or an open request for something only the inventory knows; None if it is not one."""
        t = (text or "").strip()
        if not t or len(t) > 100:
            return None
        if _RESCAN.match(t):
            self.refresh(force=True, now=now)
            g = sum(1 for i in self.items if i["kind"] == "game")
            return f"Scanned: {len(self.items) - g} apps and {g} games."
        m = _WHAT.match(t) or _WHAT2.match(t)
        if m:
            self.refresh(now=now)
            k = m.group("k").lower()
            games = k == "games"
            rows = sorted((i["name"] for i in self.items if (i["kind"] == "game") == games), key=str.lower)
            if not rows:
                return "I couldn't find any games installed through Steam or Epic." if games else "I couldn't read the installed apps list."
            shown = rows[:12]
            more = f", and {len(rows) - len(shown)} more" if len(rows) > len(shown) else ""
            return f"{len(rows)} {'games' if games else 'apps'}: " + ", ".join(shown) + more + "."
        m = _HAVE.match(t) or _HAVE2.match(t)
        if m:
            name = m.group("n").strip()
            if re.match(r"^(?:any|a|an|the|some|my|your|time|enough|anything|something|everything|it|that|this|to)\b", name, re.I) or len(name.split()) > 4:
                return None
            self.refresh(now=now)
            hit = self.find(name)
            return f"Yes -- {hit['name']} is installed." if hit else f"I can't find {name} installed."
        m = _SWITCH.match(t)
        if m and switch is not None:
            name = m.group("n").strip()
            if known_app(name) or re.match(r"^(?:the|a|an|my|it|that|this|some|something|up|me|files?|notes?|memory|tab|window|next|previous|back|desktop)\b", name, re.I) or len(name.split()) > 4:
                return None
            self.refresh(now=now)
            hit = self.find(name, cutoff=0.8)
            if hit is None:
                return None
            got = switch(hit)
            if got is None:
                return f"{hit['name']} isn't open. Say “open {hit['name']}” and I will."
            return f"Switching to {hit['name']}." if got else f"{hit['name']} is open but Windows wouldn't bring it forward -- try alt-tab."
        m = _OPEN.match(t)
        if m:
            name = m.group("n").strip()
            if known_app(name) or re.match(r"^(?:the|a|an|my|it|that|this|some|something|up|me|files?|notes?|memory)\b", name, re.I) or len(name.split()) > 4:
                return None                      # the desktop rules (or something else) have it
            self.refresh(now=now)
            hit = self.find(name, cutoff=0.88)
            if hit is None:
                return None
            if self._open(hit["open"]):
                return f"Opening {hit['name']}."
            return f"I couldn't open {hit['name']}."
        return None

    def status(self) -> Dict[str, Any]:
        return {"apps": sum(1 for i in self.items if i["kind"] == "app"), "games": sum(1 for i in self.items if i["kind"] == "game"), "scanned": bool(self.at)}
