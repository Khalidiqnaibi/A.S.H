"""
src/daemon/learnweb.py

Learning something ASH has never seen, the way the user taught it to:

    1. break the problem into concepts            "learn how to make a thumbnail in photoshop" -> make a thumbnail, photoshop
    2. for each concept, check the user's own history first    what they have written, done or been asked about it -- if they have done it, that is the source
    3. otherwise google how it is usually used     one short snippet, kept with where it came from
    4. save each concept                           what was learned (learned.json) and a skill that goes to the same search, so "do <concept>" works
    5. ask "is that right?" -- a no forgets what was just saved

Offline is normal: with no connection a concept is QUEUED, said so, and done the next time the network is there ("what are you learning?" lists the queue). Nothing here needs a model.
The network, the history and the skills are injected, so tests use fakes and nothing is searched or saved for real.

    "learn how to X from google" / "go and learn X" / "teach yourself X"        learn
    "what have you learned from google" / "what are you learning"              the store and the queue
    "how do you learn from google"                                              the method, in its steps
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_CONCEPTS = 5

METHOD = ["break the problem into concepts", "check what you have done yourself about each one", "google how each is usually used (when I'm online; otherwise I queue it)",
          "save each concept, and a skill that goes to the same search", "ask you if it's right -- a no forgets it"]

_LEAD = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|ok(?:ay)?|so|and)[,\s]+)*(?:(?:can|could)\s+you\s+)?(?:please\s+)?"
                   # 28 Nov: "From Google, learn how to open Discord" (said three times) went to the model: only "learn how to X from google" was read.
                   r"(?:(?:from|on|using|through|with)\s+(?:google|the\s+(?:web|internet)|online)[,\s]+)?"
                   r"(?:(?:go\s+(?:and\s+)?)?learn(?:\s+from\s+(?:google|the\s+(?:web|internet)|online))?(?:\s+(?:about|how\s+to|to))?|teach\s+yourself(?:\s+(?:about|how\s+to|to))?|figure\s+out\s+how\s+to|find\s+out\s+how\s+to|"
                   r"google\s+how\s+to|research\s+how\s+to)\s+(?P<p>.{3,160}?)(?:\s+(?:from|on|using|through|with)\s+(?:google|the\s+(?:web|internet)|online)|\s+by\s+(?:googling|searching)(?:\s+it)?)?\W*$", re.I)
_SPLIT = re.compile(r"\s*(?:,|;|\band\s+then\b|\bthen\b|\band\b|\bwith\b|\busing\b|\bin\b|\bon\b|\bfor\b)\s+", re.I)
_STATUS = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:what\s+(?:have\s+you|did\s+you)\s+learn(?:ed|t)?(?:\s+from\s+(?:google|the\s+web|online))?|what\s+are\s+you\s+learning|what'?s\s+(?:in\s+)?(?:your|the)\s+learning\s+queue)\W*$", re.I)
_METHOD = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:how\s+do\s+you\s+learn(?:\s+(?:things|stuff|something\s+new))?(?:\s+from\s+(?:google|the\s+web|online))?|how\s+(?:do|would)\s+you\s+learn\s+something\s+(?:new|you\s+don'?t\s+know))\W*$", re.I)
_STOP = {"it", "this", "that", "the", "a", "an", "my", "your", "to", "how", "do", "some", "something", "things", "stuff"}


def parse(text: str) -> Optional[str]:
    m = _LEAD.match((text or "").strip())
    return m.group("p").strip() if m else None


def asks_status(text: str) -> bool:
    return bool(_STATUS.match((text or "").strip()))


def asks_method(text: str) -> bool:
    return bool(_METHOD.match((text or "").strip()))


def concepts(problem: str) -> List[str]:
    """The parts of a problem, in order, each a short phrase; 'make a thumbnail in photoshop' -> ['make a thumbnail', 'photoshop']."""
    p = re.sub(r"^(?:how\s+to|to)\s+", "", (problem or "").strip(" .?!"), flags=re.I)
    out: List[str] = []
    for piece in _SPLIT.split(p):
        piece = re.sub(r"^(?:the|a|an|my|some)\s+", "", piece.strip(" .,'\""), flags=re.I)
        if len(piece) >= 3 and piece.lower() not in _STOP and piece.lower() not in [o.lower() for o in out]:
            out.append(piece)
    return out[:MAX_CONCEPTS]


class Store:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.path, self.clock = path, clock
        self.items: Dict[str, Dict[str, Any]] = {}
        self.queue: List[str] = []
        if path and os.path.exists(path):
            try:
                d = json.load(open(path, encoding="utf-8"))
                self.items, self.queue = d.get("items", {}), d.get("queue", [])
            except Exception:
                pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            json.dump({"items": self.items, "queue": self.queue}, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def status(self) -> str:
        bits = []
        if self.items:
            bits.append("I've learned " + ", ".join(list(self.items)[-6:]) + (f" (and {len(self.items) - 6} more)" if len(self.items) > 6 else ""))
        if self.queue:
            bits.append("waiting for a connection: " + ", ".join(self.queue[:6]))
        return ("; ".join(bits) + ".") if bits else "Nothing learned from the web yet -- say “learn how to ...”."


def snippet_from_html(html: str, n: int = 2) -> str:
    """The first n result snippets of a DuckDuckGo html page, as plain text."""
    out = []
    for m in re.finditer(r'class="result__snippet"[^>]*>(.*?)</a>', html or "", re.S | re.I):
        t = re.sub(r"<[^>]+>", "", m.group(1))
        t = re.sub(r"\s+", " ", t.replace("&amp;", "&").replace("&#x27;", "'").replace("&quot;", '"')).strip()
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return " ".join(out)


class NoConnection(Exception):
    """The network itself is not there (as against a search that came back empty or was refused)."""


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36", "Accept-Language": "en"}


def _plain(t: str) -> str:
    import html as _h
    t = re.sub(r"<[^>]+>", "", t or "")
    return re.sub(r"\s+", " ", _h.unescape(t)).strip()


def snippet_from_bing(page: str, n: int = 2) -> str:
    """The first n result snippets of a Bing results page, as plain text (the leading 'May 13, 2015 ·' date is dropped)."""
    out = []
    for li in re.findall(r'<li class="b_algo".*?</li>', page or "", re.S):
        m = re.search(r'<p class="b_lineclamp\d*[^"]*"[^>]*>(.*?)</p>', li, re.S)
        t = re.sub(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\W+", "", _plain(m.group(1))) if m else ""
        if len(t) > 30:
            out.append(t)
        if len(out) >= n:
            break
    return " ".join(out)


def snippet_from_wikipedia(js: str, n: int = 2) -> str:
    out = []
    try:
        for r in (json.loads(js).get("query", {}).get("search") or [])[:n]:
            out.append(_plain(str(r.get("title", "")) + ": " + str(r.get("snippet", ""))))
    except ValueError:
        return ""
    return " ".join(out)


def _get(url: str, timeout: float) -> Tuple[int, str]:
    """(status code, body) through net.fetch -- NoConnection only when the network is really down; a refusal (a 4xx, a bot check) is a code and no body."""
    from . import net
    r = net.fetch(url, "learning how something is done", timeout, headers={k: v for k, v in _UA.items() if k != "User-Agent"} | {"User-Agent": _UA["User-Agent"]})
    if r.status == "down":
        raise NoConnection(r.host)
    return (r.code or 200) if r.status in ("ok", "empty") else (r.code or 403), r.text


def is_online(timeout: float = 2.0) -> bool:
    from . import net
    return net.online()


def web_fetch(q: str, timeout: float = 6.0) -> str:
    """One search, snippets only, from the first of three places that gives any: DuckDuckGo, Bing, Wikipedia. Raises NoConnection only when the network is
    not there. 28 Nov: DuckDuckGo answered 202 'anomaly' (a bot check) to every search, no snippets came back, and ASH said 'no connection' while online."""
    import urllib.parse
    qq = urllib.parse.quote_plus(q + " how to")
    status, page = _get("https://html.duckduckgo.com/html/?q=" + qq, timeout)
    got = snippet_from_html(page) if status == 200 else ""
    if not got:
        status, page = _get("https://www.bing.com/search?setlang=en-US&cc=US&mkt=en-US&ensearch=1&q=" + qq, timeout)
        got = snippet_from_bing(page) if status == 200 else ""
    if not got:
        status, page = _get("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=2&srsearch=" + urllib.parse.quote_plus(q), timeout)
        got = snippet_from_wikipedia(page) if status == 200 else ""
    return got


def learn(problem: str, store: Store, fetch: Optional[Callable[[str], str]], history: Callable[[str], List[str]], teach: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
    """Run the method. Returns {"concepts": [...], "from_history": [...], "from_web": [...], "queued": [...], "saved": [...]} -- what was done with each."""
    cs = concepts(problem)
    out: Dict[str, Any] = {"concepts": cs, "from_history": [], "from_web": [], "queued": [], "saved": [], "empty": []}
    for c in cs:
        mine = history(c) or []
        if mine:
            _save(store, c, mine[0][:240], "your own history", teach)
            out["from_history"].append(c)
            out["saved"].append(c)
            continue
        got = ""
        offline = fetch is None
        if fetch is not None:
            try:
                got = (fetch(c) or "").strip()
            except NoConnection:
                offline = True
            except Exception:
                offline = not is_online() if fetch is web_fetch else True
        if got:
            _save(store, c, got[:300], "google", teach)
            out["from_web"].append(c)
            out["saved"].append(c)
        elif offline:
            if c not in store.queue:
                store.queue.append(c)
            out["queued"].append(c)
        else:
            out["empty"].append(c)             # the network answered and the search had nothing usable: not 'offline', and not worth queueing
    store.save()
    return out


def _save(store: Store, concept: str, summary: str, source: str, teach) -> None:
    store.items[concept] = {"summary": summary, "source": source, "at": store.clock()}
    if concept in store.queue:
        store.queue.remove(concept)
    if teach is not None:
        try:
            teach({"phrase": concept, "slots": [], "steps": [{"do": "search", "engine": "google", "q": concept}, {"do": "click_first_result"}, {"do": "read_title"}]})
        except Exception:
            pass


def drain(store: Store, fetch: Callable[[str], str], teach=None) -> List[str]:
    """When the network is back: the queued concepts, done."""
    done = []
    for c in list(store.queue):
        try:
            got = (fetch(c) or "").strip()
        except NoConnection:
            break
        except Exception:
            continue
        if got:
            _save(store, c, got[:300], "google", teach)
            done.append(c)
    store.save()
    return done


def forget(store: Store, names: List[str]) -> int:
    n = 0
    for c in names:
        if store.items.pop(c, None) is not None:
            n += 1
        if c in store.queue:
            store.queue.remove(c)
    store.save()
    return n


def say(out: Dict[str, Any]) -> str:
    if not out["concepts"]:
        return "I couldn't find a thing to learn in that."
    bits = []
    if out["from_history"]:
        bits.append("from what you've done yourself: " + ", ".join(out["from_history"]))
    if out["from_web"]:
        bits.append("from the web: " + ", ".join(out["from_web"]))
    if out["queued"]:
        bits.append("no connection for " + ", ".join(out["queued"]) + " -- queued for when I'm online")
    if out.get("empty"):
        bits.append("I reached the web but the search had nothing usable for " + ", ".join(out["empty"]))
    head = "I broke it into " + ", ".join(out["concepts"]) + ". "
    if not out["saved"]:
        return head + bits[0][0].upper() + bits[0][1:] + "."
    return head + "Learned " + "; ".join(bits) + ". Is that right?"
