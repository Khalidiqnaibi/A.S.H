"""
src/daemon/browsing.py

What the ledger says about the user, said back by rule -- no model.

The ledger (ledger.py) holds stretches of attention. This turns them into
things a person asks, in plain words, with the evidence in the answer:

    "what have I been browsing this week?"      time per category
    "how much time did I spend on youtube?"     one site or category, a period
    "what sites do I use the most?"             top sites
    "what am I into lately?"                    interests: recent, repeated, decaying
    "when do I focus best?"                     the hours most focused time falls in
    "what do you know about my browsing?"       what is kept, and how much
    "that site is work" / "tag this page as learning"   teach what is in front
    "stop watching my tabs" / "resume"          pause and resume
    "forget my browsing" / "forget everything about github.com" / "forget my social browsing"

All of it is counting and arithmetic. An answer says how many days it rests on,
and says plainly when there is not enough to say anything. "Interests" fade: a
site's weight halves every 14 days, and it has to turn up on 3 separate days.
Nothing here describes a person's mind -- it says when and where time went.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

DAY = 86400.0
FOCUS = frozenset({"code", "docs", "learning", "work", "reading"})
HALF_LIFE_DAYS = 14.0
MIN_DAYS = 3                      # a pattern needs this many separate days
MIN_FOCUS_S = 2 * 3600            # ...and this much focused time

_A = r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:so\s+)?(?:please\s+|can you\s+|could you\s+)?"
_Z = r"\s*[.!?]*\s*$"
_CATS = ("work|code|coding|docs|learning|studying|study|ai|search|video|videos|music|social|news|shopping|games|gaming|"
         "reading|travel|reference|other")
_SYN = {"coding": "code", "studying": "learning", "study": "learning", "videos": "video", "gaming": "games"}
_SITE = r"(?P<site>[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.[a-z]{2,})"

_RX: List[Tuple[str, "re.Pattern"]] = [(k, re.compile(p, re.I)) for k, p in (
    ("pause", _A + r"(?:stop|quit|pause)\s+(?:watching|tracking|recording|logging|reading)\s+(?:my\s+)?(?:tabs?|browser|browsing|websites?)"
              r"|" + _A + r"pause\s+(?:the\s+)?(?:tab|browser|browsing)\s+(?:tracking|watching|ledger)" + _Z),
    ("resume", _A + r"(?:resume|start|continue|restart|begin)\s+(?:watching|tracking|recording|logging)\s+(?:my\s+)?(?:tabs?|browser|browsing|websites?)"
               r"|" + _A + r"resume\s+(?:the\s+)?(?:tab|browser|browsing)\s+(?:tracking|watching)" + _Z),
    ("forget_all", _A + r"(?:forget|delete|erase|clear|wipe)\s+(?:all\s+|everything\s+(?:about\s+)?)?(?:of\s+)?my\s+(?:browsing|tab|tabs|web|website)"
                   r"(?:\s+(?:history|data|record|records|ledger|habits))?" + _Z),
    ("forget_tags", _A + r"(?:forget|clear|reset)\s+(?:all\s+)?my\s+(?:site\s+tags|tags)" + _Z),
    ("forget_site", _A + r"(?:forget|delete|erase|remove)\s+(?:everything|anything|all)\s+(?:you\s+(?:know|have)\s+)?(?:about|on|for)\s+" + _SITE + _Z),
    ("forget_cat", _A + r"(?:forget|delete|erase|clear)\s+(?:my\s+|all\s+my\s+)?(?P<cat>" + _CATS + r")\s+(?:browsing|sites|visits|history)" + _Z),
    ("tag", _A + r"(?:(?:tag|mark|set|file|call)\s+(?:this|that|the current)\s+(?:site|page|tab|website)\s+as\s+(?:a\s+|being\s+)?(?:for\s+)?"
              r"(?P<cat>" + _CATS + r")(?:\s+site)?"
              r"|(?:this|that)\s+(?:site|page|tab|website)\s+is\s+(?:a\s+|for\s+|about\s+)?(?P<cat2>" + _CATS + r")(?:\s+site)?"
              r"|(?:this|that)\s+is\s+a\s+(?P<cat3>" + _CATS + r")\s+(?:site|page|tab|website))" + _Z),
    ("status", _A + r"(?:what\s+do\s+you\s+(?:know|keep|remember)\s+about\s+my\s+(?:browsing|tabs|web\s+(?:use|habits)|websites)"
               r"|are\s+you\s+(?:watching|tracking|recording)\s+(?:my\s+)?(?:tabs|browsing|browser)"
               r"|is\s+(?:tab|browser|browsing)\s+(?:tracking|watching)\s+on)" + _Z),
    ("focus", _A + r"(?:when\s+(?:do|am)\s+i\s+(?:focus|work|concentrate|study|code)\s+(?:the\s+)?best"
              r"|when\s+am\s+i\s+(?:the\s+)?most\s+(?:productive|focused)"
              r"|what(?:'s|\s+is)\s+my\s+(?:best|focus)\s+(?:time|hours|pattern)(?:\s+(?:to|of the day)\b.*)?"
              r"|how\s+long\s+(?:do\s+i\s+)?(?:can\s+i\s+)?(?:stay\s+)?focus(?:ed)?)" + _Z),
    ("interests", _A + r"(?:what\s+am\s+i\s+(?:into|interested\s+in|looking\s+at|reading\s+about)(?:\s+(?:lately|recently|these\s+days|now))?"
                  r"|what\s+(?:are\s+)?my\s+(?:browsing\s+)?interests(?:\s+(?:lately|recently))?"
                  r"|what\s+do\s+i\s+browse\s+(?:the\s+)?most)" + _Z),
    ("top", _A + r"(?:what|which)\s+(?:sites|websites|pages)\s+do\s+i\s+(?:visit|use|open|go\s+to|spend\s+(?:the\s+)?(?:most\s+)?time\s+on)"
               r"(?:\s+(?:the\s+)?most)?(?:\s+(?:today|this\s+week|this\s+month|lately|recently))?"
               r"|" + _A + r"(?:what\s+(?:are\s+)?)?my\s+(?:most\s+visited|top)\s+(?:sites|websites)(?:\s+(?:today|this\s+week|this\s+month))?" + _Z),
    ("time_on", _A + r"how\s+(?:much|long)\s+(?:time\s+)?(?:did|have)\s+i\s+(?:spend|spent|been)\s+(?:on|in|at)\s+(?:the\s+)?"
                r"(?:(?P<site>[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.[a-z]{2,})|(?P<cat>" + _CATS + r")(?:\s+(?:sites|stuff|things))?|(?P<name>[a-z][a-z0-9]{2,20}))"
                r"(?:\s+(?P<per>today|this\s+week|this\s+month|lately|recently))?" + _Z),
    ("summary", _A + r"(?:what\s+(?:have\s+i\s+been|did\s+i)\s+(?:browsing|doing\s+(?:online|on\s+the\s+(?:web|computer|browser))|looking\s+at|use|spend\s+(?:my\s+)?time\s+on)"
                r"(?:\s+(?P<per>today|this\s+week|this\s+month|lately|recently))?"
                r"|where\s+did\s+my\s+(?:browsing\s+)?time\s+go(?:\s+(?P<per2>today|this\s+week|this\s+month|lately|recently))?"
                r"|how\s+(?:much\s+)?(?:have\s+i\s+been|did\s+i)\s+(?:browse|browsing|spend\s+online)(?:\s+(?P<per3>today|this\s+week|this\s+month))?)" + _Z),
)]

_NAMES = {"youtube": "youtube.com", "github": "github.com", "reddit": "reddit.com", "twitter": "x.com", "facebook": "facebook.com",
          "netflix": "netflix.com", "instagram": "instagram.com", "amazon": "amazon.com", "wikipedia": "wikipedia.org",
          "stackoverflow": "stackoverflow.com", "linkedin": "linkedin.com", "twitch": "twitch.tv", "chatgpt": "chatgpt.com"}


def parse(text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """(kind, details) if this is about browsing, else None. Used for answering and for gating a guest."""
    t = (text or "").strip()
    if not t or len(t) > 140:
        return None
    for kind, rx in _RX:
        m = rx.match(t)
        if m:
            d = {k: v for k, v in m.groupdict().items() if v}
            cat = d.pop("cat", "") or d.pop("cat2", "") or d.pop("cat3", "")
            if cat:
                d["cat"] = _SYN.get(cat.lower(), cat.lower())
            per = d.pop("per", "") or d.pop("per2", "") or d.pop("per3", "")
            if per:
                d["per"] = re.sub(r"\s+", " ", per.lower())
            name = d.pop("name", "")
            if kind == "time_on" and name and not d.get("site") and not d.get("cat"):
                if name.lower() not in _NAMES:
                    return None                  # "how long did I spend on homework": not a site we can name
                d["site"] = _NAMES[name.lower()]
            return kind, d
    return None


def dur(secs: float) -> str:
    s = int(round(secs))
    if s < 60:
        return "under a minute"
    m = s // 60
    h, m = divmod(m, 60)
    if h and m:
        return f"{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"
    if h:
        return f"{h} hour{'s' if h != 1 else ''}"
    return f"{m} minute{'s' if m != 1 else ''}"


def _lst(xs: List[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _midnight(now: float) -> float:
    t = time.localtime(now)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def _since(per: str, now: float) -> Tuple[float, str]:
    if per == "today":
        return _midnight(now), "today"
    if per == "this month":
        return now - 30 * DAY, "this month"
    return now - 7 * DAY, "this week" if per in ("", "this week") else "lately"


class Browsing:
    """Answers about the ledger, and the controls over it. `sensor()` returns the BrowserTabSensor or None."""

    def __init__(self, sensor: Callable[[], Any], clock: Callable[[], float] = time.time):
        self.sensor = sensor
        self.clock = clock

    # -- plumbing -----------------------------------------------------------------
    def _s(self):
        try:
            return self.sensor()
        except Exception:
            return None

    def _on(self) -> bool:
        s = self._s()
        return s is not None and getattr(s, "ledger", None) is not None

    # -- the profile (pure arithmetic over episodes) ------------------------------
    def interests(self, now: Optional[float] = None, limit: int = 5) -> List[Tuple[str, str, float, int]]:
        """(site, category, weighted seconds, days seen): recent, repeated, fading with a 14 day half-life."""
        s = self._s()
        now = self.clock() if now is None else now
        w: Dict[Tuple[str, str], float] = defaultdict(float)
        days: Dict[Tuple[str, str], set] = defaultdict(set)
        for e in s.ledger.episodes(now - 30 * DAY, now + 1, limit=200000):
            if e["category"] in ("search", "other", "unknown"):
                continue
            k = (e["site"], e["category"])
            age = max(0.0, (now - e["start"]) / DAY)
            w[k] += (e["end"] - e["start"]) * 0.5 ** (age / HALF_LIFE_DAYS)
            days[k].add(time.strftime("%Y-%m-%d", time.localtime(e["start"])))
        rows = [(k[0], k[1], v, len(days[k])) for k, v in w.items() if len(days[k]) >= MIN_DAYS]
        rows.sort(key=lambda r: -r[2])
        return rows[:limit]

    def focus_profile(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """When focused time falls, and the longest unbroken stretch, over the last 30 days. None if too little."""
        s = self._s()
        now = self.clock() if now is None else now
        eps = [e for e in s.ledger.episodes(now - 30 * DAY, now + 1, limit=200000) if e["category"] in FOCUS]
        if not eps:
            return None
        hours = [0.0] * 24
        days = set()
        total = 0.0
        for e in eps:
            a, b = e["start"], e["end"]
            total += b - a
            days.add(time.strftime("%Y-%m-%d", time.localtime(a)))
            while a < b:                                           # split across the hours it spans
                t = time.localtime(a)
                nxt = min(b, a + (3600 - (t.tm_min * 60 + t.tm_sec)))
                hours[t.tm_hour] += nxt - a
                a = nxt
        if len(days) < MIN_DAYS or total < MIN_FOCUS_S:
            return {"enough": False, "days": len(days), "total": total}
        best = max(range(24), key=lambda h: hours[h] + hours[(h + 1) % 24])
        stretch, longest, last_end = 0.0, 0.0, None
        for e in sorted(eps, key=lambda e: e["start"]):
            if last_end is not None and e["start"] - last_end <= 300:
                stretch += e["end"] - e["start"]
            else:
                stretch = e["end"] - e["start"]
            last_end = e["end"]
            longest = max(longest, stretch)
        return {"enough": True, "days": len(days), "total": total, "from": best, "to": (best + 2) % 24,
                "share": (hours[best] + hours[(best + 1) % 24]) / total, "longest": longest}

    # -- answering -----------------------------------------------------------------
    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        p = parse(text)
        if p is None:
            return None
        kind, d = p
        now = self.clock() if now is None else now
        s = self._s()
        if s is None or getattr(s, "ledger", None) is None:
            if kind == "tag":
                return None                                  # "this page is work" is ordinary talk when tabs are not watched
            if kind in ("pause", "resume", "status"):
                return ("I'm not watching your tabs. It's off by default; turn on sensors.browser_tabs in daemon.json "
                        "and load the ash-tabs extension, and I'll start.")
            return ("I'm not keeping track of your browsing -- it's off. Turn on sensors.browser_tabs in daemon.json "
                    "and load the ash-tabs extension if you want that.")
        led = s.ledger
        if kind == "pause":
            s.pause()
            return "Okay, I've stopped watching your tabs. Say \"resume watching my tabs\" when you want me to start again."
        if kind == "resume":
            s.resume()
            return "Okay, I'm watching your tabs again."
        if kind == "status":
            c = led.counts()
            n_days = len({time.strftime("%Y-%m-%d", time.localtime(e["start"])) for e in led.episodes(now - 60 * DAY, now + 1, limit=200000)})
            state = "paused" if s.paused else "watching"
            return (f"I'm {state}. I keep the site, its category and how long it was in front -- not pages, addresses, "
                    f"titles or anything from private windows, banks, mail or health sites. Right now that's {c['episodes']} "
                    f"stretches over {n_days} days, plus older totals; the detail expires after {int(led.retention_days)} days. "
                    f"Say \"forget my browsing\" to delete it.")
        if kind == "forget_all":
            n = led.forget_all()
            return f"Done -- I've deleted everything I kept about your browsing ({n} records)." if n else "I didn't have anything about your browsing."
        if kind == "forget_tags":
            s.cls.forget_all()
            return "I've forgotten the sites you tagged and what I'd learned from them."
        if kind == "forget_site":
            site = d["site"].lower()
            from .pagekind import host_of, site_of
            n = led.forget(site=site_of(host_of(site)))
            s.cls.untag(site)
            return f"Forgotten: everything about {site}." if n else f"I don't have anything about {site}."
        if kind == "forget_cat":
            n = led.forget(category=d["cat"])
            return f"Forgotten: your {d['cat']} browsing." if n else f"I don't have any {d['cat']} browsing."
        if kind == "tag":
            cat = d.get("cat", "")
            site = s.teach_current(cat)
            if site:
                return f"Okay, {site} is {cat} from now on."
            return "I don't see a page in front right now to tag." if s.current() is None else "I can't tag that one."
        if s.paused:
            return "I'm not watching your tabs at the moment, so I can't say. Say \"resume watching my tabs\" to start again."
        if kind == "summary":
            since, label = _since(d.get("per", ""), now)
            rows = [(c, sec) for c, sec, _n in led.totals(since, now + 1) if c not in ("unknown",)]
            tot = sum(sec for _c, sec in rows)
            if tot < 60:
                return f"I haven't seen much browsing {label} -- under a minute."
            top = [f"{c} {dur(sec)}" for c, sec in rows[:4]]
            return f"{label.capitalize()} you spent {dur(tot)} in the browser: {_lst(top)}."
        if kind == "time_on":
            since, label = _since(d.get("per", ""), now)
            if d.get("site"):
                from .pagekind import host_of, site_of
                site = site_of(host_of(d["site"]))
                rows = led.totals(since, now + 1, site=site)
                sec = sum(r[1] for r in rows)
                what = d["site"]
            else:
                rows = led.totals(since, now + 1, category=d["cat"])
                sec = sum(r[1] for r in rows)
                what = f"{d['cat']} sites"
            if sec < 60:
                return f"I haven't seen you on {what} {label}."
            return f"{label.capitalize()} you spent {dur(sec)} on {what}."
        if kind == "top":
            since, label = _since(d.get("per", ""), now) if d.get("per") else (now - 7 * DAY, "this week")
            rows = led.totals(since, now + 1, by="site")[:5]
            if not rows or sum(r[1] for r in rows) < 60:
                return "I don't have enough browsing yet to say."
            return f"{label.capitalize()}, mostly " + _lst([f"{n} ({dur(sec)})" for n, sec, _e in rows[:5]]) + "."
        if kind == "interests":
            rows = self.interests(now)
            if not rows:
                return ("Nothing stands out yet -- I need a site to turn up on at least three separate days before I'd "
                        "call it an interest.")
            return "Lately you keep coming back to " + _lst([f"{site} ({cat})" for site, cat, _w, _d in rows]) + "."
        if kind == "focus":
            f = self.focus_profile(now)
            if f is None or not f.get("enough"):
                have = f["days"] if f else 0
                return (f"I don't have enough yet -- I'd need focused browsing (code, docs, learning, work, reading) on at "
                        f"least {MIN_DAYS} days and a couple of hours in all; so far it's {have} day{'s' if have != 1 else ''}.")
            return (f"Most of your focused browsing falls between {f['from']:02d}:00 and {f['to']:02d}:00 "
                    f"({int(round(f['share'] * 100))}% of it, over {f['days']} days), and your longest unbroken stretch "
                    f"lately was {dur(f['longest'])}.")
        return None
