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

import os
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .numwords import digits

DAY = 86400.0
FOCUS = frozenset({"code", "docs", "learning", "work", "reading"})
LEISURE = ("video", "social", "games", "shopping", "news")
HALF_LIFE_DAYS = 14.0
MIN_DAYS = 3                      # a pattern needs this many separate days
MIN_FOCUS_S = 2 * 3600            # ...and this much focused time

_A = r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:so\s+)?(?:please\s+|can you\s+|could you\s+)?"
_Z = r"\s*[.!?]*\s*$"
_CATS = ("work|code|coding|docs|learning|studying|study|ai|search|video|videos|music|social|news|shopping|games|gaming|"
         "reading|travel|reference|other")
_SYN = {"coding": "code", "studying": "learning", "study": "learning", "videos": "video", "gaming": "games"}
_SITE = r"(?P<site>[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.[a-z]{2,})"

_FOR_HOW_LONG = (r"(?:\s+for\s+(?:the\s+next\s+)?|\s+the\s+next\s+)(?:(?:(?:an?|one)\s+)?(?P<unit1>hour|minute|min)\b|(?P<n>\d+(?:\.\d+)?)\s*"
                 r"(?P<unit>hours?|hrs?|minutes?|mins?))")

_RX: List[Tuple[str, "re.Pattern"]] = [(k, re.compile(p, re.I)) for k, p in (
    # "don't record the next hour": a pause that ends by itself. Before "pause", whose pattern would take it as a stop.
    ("pause_for", _A + r"(?:go\s+private|don'?t\s+(?:record|track|watch|log)(?:\s+(?:my\s+)?(?:tabs|browsing|browser|websites?))?|"
                  r"stop\s+(?:recording|tracking|watching|logging)(?:\s+(?:my\s+)?(?:tabs|browsing|browser|websites?))?|"
                  r"pause\s+(?:the\s+)?(?:tab|browser|browsing)\s+(?:tracking|watching))" + _FOR_HOW_LONG + _Z),
    ("pause", _A +r"(?:stop|quit|pause)\s+(?:watching|tracking|recording|logging|reading)\s+(?:my\s+)?(?:tabs?|browser|browsing|websites?)"
              r"|" + _A + r"pause\s+(?:the\s+)?(?:tab|browser|browsing)\s+(?:tracking|watching|ledger)" + _Z),
    ("resume", _A + r"(?:resume|start|continue|restart|begin)\s+(?:watching|tracking|recording|logging)\s+(?:my\s+)?(?:tabs?|browser|browsing|websites?)"
               r"|" + _A + r"resume\s+(?:the\s+)?(?:tab|browser|browsing)\s+(?:tracking|watching)" + _Z),
    # "never track youtube" / "ignore reddit.com" / "start tracking youtube again" / "what are you not tracking?"
    ("ignore_site", _A + r"(?:(?:never|don'?t|do\s+not|stop)\s+(?:track|tracking|record|recording|log|logging|watch|watching)|ignore|exclude)\s+"
                    r"(?:" + _SITE + r"|(?P<name>[a-z][a-z0-9]{2,20}))" + _Z),
    ("unignore_site", _A + r"(?:(?:start|resume|begin)\s+(?:tracking|recording|logging|watching)|include)\s+"
                      r"(?:" + _SITE + r"|(?P<name>[a-z][a-z0-9]{2,20}))(?:\s+again)?" + _Z),
    ("export", _A + r"(?:export|download|save|dump)\s+(?:a\s+copy\s+of\s+)?(?:all\s+of\s+)?my\s+(?:browsing|tab|tabs|web)\s*(?:data|history|ledger|record|records)?(?:\s+to\s+a\s+file)?" + _Z),
    ("everything", _A + r"(?:(?:show|tell|list)\s+me\s+everything\s+you\s+(?:hold|have|keep|know|store)\s+(?:on|about)\s+(?:me|my\s+browsing)"
                        r"|what\s+exactly\s+(?:do\s+you\s+)?(?:hold|have|keep|store)\s+(?:on|about)\s+(?:me|my\s+browsing))" + _Z),
    ("ignored_list", _A + r"what\s+(?:sites\s+|websites\s+)?(?:am\s+i|are\s+you)\s+(?:ignoring|not\s+tracking|excluding)(?:\s+(?:from|in)\s+my\s+browsing)?" + _Z),
    ("forget_all", _A + r"(?:forget|delete|erase|clear|wipe)\s+(?:all\s+|everything\s+(?:about\s+)?)?(?:of\s+)?my\s+(?:browsing|tab|tabs|web|website)"
                   r"(?:\s+(?:history|data|record|records|ledger|habits))?" + _Z),
    ("forget_tags", _A + r"(?:forget|clear|reset)\s+(?:all\s+)?my\s+(?:site\s+tags|tags)" + _Z),
    ("forget_site", _A + r"(?:forget|delete|erase|remove)\s+(?:everything|anything|all)\s+(?:you\s+(?:know|have)\s+)?(?:about|on|for)\s+" + _SITE + _Z),
    ("forget_cat", _A + r"(?:forget|delete|erase|clear)\s+(?:my\s+|all\s+my\s+)?(?P<cat>" + _CATS + r")\s+(?:browsing|sites|visits|history)" + _Z),
    ("tag", _A + r"(?:(?:tag|mark|set|file|call)\s+(?:this|that|the current)\s+(?:site|page|tab|website)\s+as\s+(?:a\s+|being\s+)?(?:for\s+)?"
              r"(?P<cat>" + _CATS + r")(?:\s+site)?"
              r"|(?:this|that)\s+(?:site|page|tab|website)\s+is\s+(?:a\s+|for\s+|about\s+)?(?P<cat2>" + _CATS + r")(?:\s+site)?"
              r"|(?:this|that)\s+is\s+a\s+(?P<cat3>" + _CATS + r")\s+(?:site|page|tab|website))" + _Z),
    ("suggest", _A + r"(?:(?:any|some|got\s+any)\s+(?:suggestions?|tips|advice|thoughts)\s+(?:about|on|for|from)\s+my\s+(?:browsing|time|focus|habits|tabs|screen\s+time|week)"
                r"|how\s+can\s+i\s+(?:use|spend|manage)\s+my\s+(?:time|week)\s+(?:better|more\s+wisely|well)"
                r"|what\s+should\s+i\s+(?:change|do\s+differently)\s+(?:about|with)\s+my\s+(?:browsing|habits|time|week)"
                r"|how\s+am\s+i\s+doing\s+(?:with|on)\s+my\s+(?:time|browsing|focus|week))" + _Z),
    ("habits", _A + r"(?:what\s+are\s+my\s+(?:browsing|web|online)\s+habits|tell\s+me\s+about\s+my\s+(?:browsing|online)\s+habits"
               r"|how\s+do\s+i\s+(?:browse|use\s+(?:the\s+)?(?:browser|web|internet))|what(?:'s|\s+is)\s+my\s+(?:browsing|online)\s+routine)" + _Z),
    # "tag acme.dev as work" / "acme.dev is a work site": teach a site by name (the one in front is "tag this site as ...")
    ("tag_site", _A + r"(?:(?:tag|mark|set|file|call)\s+(?:" + _SITE + r"|(?!(?:this|that|the|my|current)\b)(?P<name>[a-z][a-z0-9]{2,20}))\s+as\s+"
                      r"(?:a\s+|being\s+)?(?:for\s+)?(?P<cat>" + _CATS + r")(?:\s+site)?"
                      r"|" + _SITE.replace("(?P<site>", "(?P<site2>") + r"\s+is\s+(?:a\s+|for\s+)?(?P<cat2>" + _CATS + r")(?:\s+site)?)" + _Z),
    ("unknown_sites", _A + r"(?:what|which)\s+(?:sites|websites)\s+(?:can'?t\s+you\s+place|don'?t\s+you\s+know|are\s+(?:unknown|unplaced|uncategori[sz]ed)|do\s+you\s+not\s+know)"
                           r"(?:\s+(?:yet|so\s+far))?" + _Z + r"|" + _A + r"which\s+sites\s+(?:are\s+you\s+unsure\s+about|need\s+(?:a\s+)?tag(?:s|ging)?)" + _Z),
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
    t = digits((text or "").strip())            # "thirty minutes" -> "30 minutes"
    if not t or len(t) > 140:
        return None
    for kind, rx in _RX:
        m = rx.match(t)
        if m:
            d = {k: v for k, v in m.groupdict().items() if v}
            if kind == "pause_for":
                mins = 60.0 if (d.get("unit1") or "").lower() == "hour" else 1.0 if d.get("unit1") else \
                    float(d["n"]) * (60.0 if d["unit"].lower().startswith("h") else 1.0)
                if not 1.0 <= mins <= 24 * 60:
                    return None
                return kind, {"secs": str(int(mins * 60))}
            cat = d.pop("cat", "") or d.pop("cat2", "") or d.pop("cat3", "")
            if cat:
                d["cat"] = _SYN.get(cat.lower(), cat.lower())
            per = d.pop("per", "") or d.pop("per2", "") or d.pop("per3", "")
            if per:
                d["per"] = re.sub(r"\s+", " ", per.lower())
            name = d.pop("name", "")
            if d.get("site2") and not d.get("site"):
                d["site"] = d.pop("site2")
            if kind in ("time_on", "ignore_site", "unignore_site", "tag_site") and name and not d.get("site") and not d.get("cat"):
                if name.lower() not in _NAMES:
                    return None                  # "how long did I spend on homework": not a site we can name
                d["site"] = _NAMES[name.lower()]
            elif kind == "tag_site" and name and not d.get("site"):
                if name.lower() not in _NAMES:
                    return None
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

    def __init__(self, sensor: Callable[[], Any], clock: Callable[[], float] = time.time, export_dir: Optional[str] = None):
        self.sensor = sensor
        self.clock = clock
        self.export_dir = export_dir

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

    def _totals(self, a: float, b: float) -> Dict[str, float]:
        return {c: sec for c, sec, _n in self.sensor().ledger.totals(a, b) if c != "unknown"}

    def suggest(self, now: float) -> str:
        """Observations about this week against last, each with its numbers; nothing if nothing stands out.
        Rules only. It says what changed and when focus tends to fall -- it does not tell anyone what to do."""
        led = self.sensor().ledger
        this, prev = self._totals(now - 7 * DAY, now + 1), self._totals(now - 14 * DAY, now - 7 * DAY)
        t_tot, p_tot = sum(this.values()), sum(prev.values())
        if t_tot < 2 * 3600:
            return "I don't have enough browsing this week yet to suggest anything."
        out: List[str] = []
        if p_tot >= 2 * 3600:
            for cat in LEISURE:
                a, b = this.get(cat, 0.0), prev.get(cat, 0.0)
                if a - b >= 3600 and a >= 1.5 * b:
                    out.append(f"{cat} is up: {dur(a)} this week against {dur(b)} last week.")
            fa = sum(this.get(c, 0.0) for c in FOCUS)
            fb = sum(prev.get(c, 0.0) for c in FOCUS)
            if fb - fa >= 3600 and fa <= 0.6 * fb:
                out.append(f"Focused browsing is down: {dur(fa)} this week against {dur(fb)} last week.")
            elif fa - fb >= 3600 and fa >= 1.4 * fb:
                out.append(f"Focused browsing is up: {dur(fa)} this week against {dur(fb)} last week.")
            c1, c0 = led.churn_between(now - 7 * DAY, now), led.churn_between(now - 14 * DAY, now - 7 * DAY)
            r1, r0 = c1 / max(t_tot / 3600, 1e-9), c0 / max(p_tot / 3600, 1e-9)
            if r1 >= 15 and r1 >= 1.5 * r0:
                out.append(f"You've been opening and closing tabs within seconds more: about {int(round(r1))} an hour, "
                           f"up from {int(round(r0))}.")
        late = 0.0
        for e in led.episodes(now - 7 * DAY, now + 1, limit=200000):
            a, b = e["start"], e["end"]
            while a < b:
                t = time.localtime(a)
                nxt = min(b, a + (3600 - (t.tm_min * 60 + t.tm_sec)))
                if t.tm_hour < 5:
                    late += nxt - a
                a = nxt
        if late >= 3600 and late / t_tot >= 0.15:
            out.append(f"{dur(late)} of this week's browsing ({int(round(100 * late / t_tot))}%) was between midnight and 5.")
        f = self.focus_profile(now)
        if f and f.get("enough"):
            out.append(f"Your focused browsing clusters between {f['from']:02d}:00 and {f['to']:02d}:00 "
                       f"({f['days']} days of it) -- that's the stretch worth keeping free.")
        if not out:
            return "Nothing stands out -- this week looks much like the last."
        return "Here's what stands out: " + " ".join(out[:4])

    def habits(self, now: float) -> str:
        """A plain description of when and how the user browses, over up to 30 days."""
        led = self.sensor().ledger
        eps = led.episodes(now - 30 * DAY, now + 1, limit=200000)
        days = {time.strftime("%Y-%m-%d", time.localtime(e["start"])) for e in eps}
        total = sum(e["end"] - e["start"] for e in eps)
        if len(days) < MIN_DAYS or total < 2 * 3600:
            return (f"I don't have enough yet to describe your habits -- I need browsing on at least {MIN_DAYS} days "
                    f"and a couple of hours in all; so far it's {len(days)} day{'s' if len(days) != 1 else ''}.")
        parts = {"morning": 0.0, "afternoon": 0.0, "evening": 0.0, "night": 0.0}
        for e in eps:
            a, b = e["start"], e["end"]
            while a < b:
                t = time.localtime(a)
                nxt = min(b, a + (3600 - (t.tm_min * 60 + t.tm_sec)))
                h = t.tm_hour
                parts["morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 17 else "evening" if h >= 17 else "night"] += nxt - a
                a = nxt
        top = max(parts, key=lambda k: parts[k])
        lens = sorted(e["end"] - e["start"] for e in eps)
        median = lens[len(lens) // 2]
        churn = led.churn_between(now - 30 * DAY, now)
        bits = [f"Over {len(days)} days you browse about {dur(total / len(days))} a day, mostly in the {top} "
                f"({int(round(100 * parts[top] / total))}%).",
                f"A typical stretch on one site is {dur(median)}."]
        if total >= 3600 and churn:
            bits.append(f"You open and close about {int(round(churn / (total / 3600)))} tabs an hour within seconds.")
        return " ".join(bits)

    # -- answering -----------------------------------------------------------------
    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        p = parse(text)
        if p is None:
            return None
        kind, d = p
        now = self.clock() if now is None else now
        s = self._s()
        if s is None or getattr(s, "ledger", None) is None:
            if kind in ("tag", "suggest", "habits"):
                return None                                  # ordinary talk ("how can I use my time better") when tabs are not watched
            if kind in ("pause", "resume", "status"):
                return ("I'm not watching your tabs. It's off by default; turn on sensors.browser_tabs in daemon.json "
                        "and load the ash-tabs extension, and I'll start.")
            return ("I'm not keeping track of your browsing -- it's off. Turn on sensors.browser_tabs in daemon.json "
                    "and load the ash-tabs extension if you want that.")
        led = s.ledger
        if kind in ("ignore_site", "unignore_site", "ignored_list"):
            if kind == "ignored_list":
                ig = sorted(s.cls.ignored)
                return ("I'm not ignoring any site, apart from the private kinds I never record (banks, mail, chat, health...)."
                        if not ig else "I'm keeping out: " + _lst(ig) + ". Plus the private kinds I never record (banks, mail, chat, health...).")
            from .pagekind import host_of, site_of
            site = site_of(host_of(d["site"]))
            if kind == "unignore_site":
                return (f"Okay, I'll track {site} again." if s.cls.unignore(site) else f"I wasn't ignoring {site}.")
            if s.cls.is_ignored(site):
                return f"I'm already keeping {site} out."
            s.cls.ignore(site)
            n = led.forget(site=site)
            gone = f" and deleted what I had ({n} records)" if n else ""
            return f"Okay, I won't record anything about {site} from now on{gone}. Say \"start tracking {site} again\" to undo."
        if kind == "pause_for":
            secs = float(d["secs"])
            s.pause(secs)
            return (f"Okay, I won't record your tabs for {dur(secs)} -- I'll start again at "
                    f"{time.strftime('%H:%M', time.localtime(now + secs))}. Say \"resume watching my tabs\" to start sooner.")
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
        if kind == "export":
            folder = self.export_dir or os.path.join(os.environ.get("ASH_STATE_DIR", os.path.join(os.getcwd(), "state")), "exports")
            path = os.path.join(folder, time.strftime("browsing-%Y%m%d-%H%M%S.json", time.localtime(now)))
            try:
                n = led.export(path)
            except Exception:
                return "I couldn't write the export file."
            return (f"Done -- {n} rows saved to {path}. It's plain JSON, site and category and time only, "
                    f"so you can open it in any text editor.")
        if kind == "everything":
            first, last = led.span()
            c = led.counts()
            top = led.top_sites(5)
            when = (f" from {time.strftime('%d %b', time.localtime(first))} to {time.strftime('%d %b', time.localtime(last))}"
                    if first else "")
            bits = [f"Here is all of it: {c['episodes']} recent stretches{when}, and totals for {c['sites']} sites over older days."]
            if top:
                bits.append("Most time: " + _lst([f"{site} ({cat}, {dur(sec)})" for site, cat, sec, _v in top]) + ".")
            bits.append(f"You've told me the category of {s.cls.status()['tags']} site{'s' if s.cls.status()['tags'] != 1 else ''}"
                        f"{'; I keep out ' + _lst(sorted(s.cls.ignored)) if s.cls.ignored else ''}.")
            bits.append("I never keep page addresses, page text, titles, or anything from private windows or banks, mail, chat, "
                        "health and similar sites. Say \"export my browsing data\" for a file, or \"forget my browsing\" to delete it.")
            return " ".join(bits)
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
        if kind == "tag_site":
            from .pagekind import host_of, site_of
            site = site_of(host_of(d["site"]))
            if not s.cls.teach(site, d["cat"]):
                return f"I never record {site}, so there is nothing to tag."
            n = led.recategorize(site, d["cat"])
            moved = f" I've also updated the {n} records I'd kept as unknown." if n else ""
            return f"Okay, {site} is {d['cat']} from now on.{moved}"
        if kind == "unknown_sites":
            rows = [r for r in led.unknown_sites(50) if not s.cls.is_ignored(r[0])][:5]       # never ask about a site kept out
            if not rows:
                return "I've been able to place every site I've seen."
            return ("I can't place " + _lst([f"{site} ({dur(sec)})" for site, sec, _v in rows])
                    + f". Tell me what they are -- for example \"tag {rows[0][0]} as work\".")
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
        if kind == "suggest":
            return self.suggest(now)
        if kind == "habits":
            return self.habits(now)
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
