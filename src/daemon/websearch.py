"""
src/daemon/websearch.py

"Look up cat pics", "search for the best ramen near me", "search YouTube for lo-fi", "google segue": opens the default browser on the search page. A URL,
and nothing else -- ASH does not read the page, fetch anything or send anything itself; the person's own browser does the search, in front of them.

It was answered "I can't browse the web" (honest.py) or by the model offering to discuss. Opening a search is something it CAN do, and the nearest thing
to "look it up": the user sees the results, in their own browser, under their own privacy settings.

Left alone: anything that is about ASH's own stores ("search my files", "look up my notes", "search memory", "look it up in Wikipedia" is the facts rule's), a
query that is empty, and anything longer than a short line.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional, Tuple

_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|so|and|please|can\s+you|could\s+you|just)[,\s]+)*"
_ENGINES = {
    "youtube": "https://www.youtube.com/results?search_query=",
    "google": "https://www.google.com/search?q=",
    "images": "https://www.google.com/search?tbm=isch&q=",
    "maps": "https://www.google.com/maps/search/",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search=",
    "duckduckgo": "https://duckduckgo.com/?q=",
    "bing": "https://www.bing.com/search?q=",
}
_NAMES = {"youtube": "YouTube", "google": "Google", "images": "Google Images", "maps": "Google Maps", "wikipedia": "Wikipedia", "duckduckgo": "DuckDuckGo", "bing": "Bing"}
_ON = r"(?:(?:on|in|at|using|with)\s+(?P<e>youtube|google|wikipedia|maps|google\s+maps|bing|duckduckgo|the\s+web|the\s+internet|the\s+net|online)\s+)?"
_SEARCH = re.compile(_LEAD + r"(?:search\s+(?P<e2>youtube|google|wikipedia|maps|bing|duckduckgo|the\s+web|the\s+internet|online)\s+for\s+(?P<q2>.+?)|"
                     r"(?:look\s+up|search(?:\s+for)?|google)\s+" + _ON + r"(?:for\s+)?(?P<q>.+?)|"
                     r"(?:show|find|get)\s+me\s+(?P<pics>pictures|photos|images|pics)\s+of\s+(?P<q3>.+?))"
                     r"(?:\s+(?:on|in|at)\s+(?P<e3>youtube|google|wikipedia|maps|bing|duckduckgo|the\s+web|the\s+internet|online))?(?:\s+please)?\s*[.!?]*\s*$", re.I)
# What these are about is ASH's own stores or the machine, not the web: left to the rules that know them.
_NOT_WEB = re.compile(r"^(?:(?:my|our|the)\s+)?(?:files?|documents?|folders?|notes?|memory|memories|computer|pc|laptop|disk|drive|downloads?|desktop|"
                      r"email|emails|calendar|reminders?|messages?|contacts?|history|screen|clipboard|logs?|camera|photos?\s+on\s+my)\b|^(?:it|this|that|them|him|her|something|anything|"
                      r"everything|nothing)$|^(?:my|your)\b", re.I)


def parse(text: str, default: str = "google") -> Optional[Tuple[str, str]]:
    """(engine, query) for a search to open, or None."""
    t = (text or "").strip()
    if not t or len(t) > 100:
        return None
    m = _SEARCH.match(t)
    if not m:
        return None
    q = (m.group("q") or m.group("q2") or m.group("q3") or "").strip(" .,!?\"'")
    e = (m.group("e") or m.group("e2") or m.group("e3") or "").strip().lower()
    if not q or len(q.split()) > 12 or _NOT_WEB.match(q):
        return None
    if m.group("pics"):
        e = e or "images"
    if re.search(r"\b(?:pics|pictures|photos|images)\b\s*$", q, re.I) and not e:
        e = "images"                                         # "look up cat pics": the images, not the page about them
    if e in ("the web", "the internet", "the net", "online", ""):
        e = default if default in _ENGINES else "google"
    if e == "google maps":
        e = "maps"
    if e == "youtube" or e in _ENGINES:
        pass
    else:
        e = default if default in _ENGINES else "google"
    return e, q


def url_for(engine: str, query: str) -> str:
    base = _ENGINES.get(engine, _ENGINES["google"])
    return base + urllib.parse.quote(query if engine != "maps" else query, safe="")


def say(engine: str, query: str) -> str:
    return f"Searching {_NAMES.get(engine, 'Google')} for {query}."
