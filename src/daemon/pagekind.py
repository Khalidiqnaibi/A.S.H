"""
src/daemon/pagekind.py

What kind of page is this -- and is it a page ASH must never record?

No language model, and no network: a table, some keywords, and a tiny Naive
Bayes that learns from the user's own corrections. In order:

    1. sensitive    banks, health, mail, chat, adult, password managers, private
                    network addresses: the page is dropped before anything is
                    kept, not even counted by site
    2. the user     "tag this site as work" beats everything below
    3. the table    a few hundred well-known sites -> a category
    4. keywords     "tutorial", "docs", "cart", "buy" in the title
    5. learned      a Naive Bayes over title words, trained only by the user's
                    tags; speaks only with enough evidence and a clear winner
    6. unknown      said honestly, never guessed

`classify()` returns (category, source), source being one of
sensitive | user | table | keyword | learned | unknown.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from typing import Dict, Iterable, List, Optional, Tuple

CATEGORIES = ("code", "docs", "learning", "work", "ai", "search", "video", "music", "social", "news",
              "shopping", "games", "reading", "travel", "reference", "other", "unknown")


def _grouped(d: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for cat, hosts in d.items():
        for h in hosts.split():
            out[h] = cat
    return out


# Most specific suffix wins ("docs.google.com" before "google.com").
TABLE: Dict[str, str] = _grouped({
    "code": "github.com gitlab.com bitbucket.org stackoverflow.com stackexchange.com superuser.com serverfault.com "
            "npmjs.com pypi.org crates.io pkg.go.dev hub.docker.com codepen.io jsfiddle.net replit.com codesandbox.io "
            "leetcode.com hackerrank.com codeforces.com huggingface.co kaggle.com colab.research.google.com "
            "vercel.com netlify.com heroku.com localhost regex101.com godbolt.org dev.to hashnode.com",
    "docs": "docs.google.com sheets.google.com slides.google.com notion.so notion.site coda.io airtable.com "
            "developer.mozilla.org docs.python.org learn.microsoft.com readthedocs.io readthedocs.org devdocs.io "
            "docs.github.com docs.aws.amazon.com cloud.google.com docs.docker.com kubernetes.io react.dev "
            "nodejs.org rust-lang.org go.dev typescriptlang.org tailwindcss.com wikipedia.org "
            "overleaf.com office.com onedrive.live.com dropbox.com paper.dropbox.com",
    "learning": "coursera.org udemy.com edx.org khanacademy.org udacity.com pluralsight.com skillshare.com "
                "duolingo.com brilliant.org freecodecamp.org codecademy.com datacamp.com ocw.mit.edu "
                "lynda.com masterclass.com",
    "work": "asana.com trello.com atlassian.net jira.com linear.app monday.com figma.com miro.com "
            "zoom.us meet.google.com teams.microsoft.com calendar.google.com clickup.com basecamp.com "
            "salesforce.com hubspot.com zendesk.com intercom.com canva.com linkedin.com",
    "ai": "chatgpt.com openai.com claude.ai anthropic.com gemini.google.com perplexity.ai poe.com "
          "midjourney.com huggingface.co/chat character.ai copilot.microsoft.com",
    "search": "google.com bing.com duckduckgo.com search.brave.com ecosia.org yahoo.com startpage.com yandex.com baidu.com",
    "video": "youtube.com youtu.be netflix.com twitch.tv vimeo.com disneyplus.com hulu.com primevideo.com "
             "crunchyroll.com max.com hbomax.com tv.apple.com dailymotion.com bilibili.com ted.com",
    "music": "spotify.com music.youtube.com soundcloud.com bandcamp.com deezer.com tidal.com music.apple.com "
             "pandora.com last.fm genius.com",
    "social": "twitter.com x.com facebook.com instagram.com reddit.com tiktok.com pinterest.com tumblr.com "
              "threads.net mastodon.social bsky.app snapchat.com quora.com 9gag.com imgur.com",
    "news": "bbc.com bbc.co.uk cnn.com nytimes.com theguardian.com reuters.com apnews.com bloomberg.com wsj.com "
            "washingtonpost.com aljazeera.com news.ycombinator.com techcrunch.com theverge.com wired.com "
            "arstechnica.com engadget.com forbes.com economist.com npr.org medium.com substack.com",
    "shopping": "amazon.com amazon.co.uk ebay.com aliexpress.com etsy.com walmart.com target.com bestbuy.com "
                "ikea.com shein.com temu.com zalando.com newegg.com costco.com noon.com souq.com shopify.com "
                "booking.com airbnb.com",
    "games": "steampowered.com store.steampowered.com epicgames.com itch.io gog.com roblox.com minecraft.net "
             "chess.com lichess.org ign.com gamespot.com twitch.tv/directory leagueoflegends.com",
    "reading": "goodreads.com archive.org gutenberg.org scribd.com wattpad.com royalroad.com kindle.amazon.com "
               "arxiv.org scholar.google.com researchgate.net pubmed.ncbi.nlm.nih.gov semanticscholar.org",
    "travel": "maps.google.com google.com/maps tripadvisor.com expedia.com kayak.com skyscanner.com "
              "uber.com lyft.com waze.com openstreetmap.org",
    "reference": "wikipedia.org wiktionary.org britannica.com dictionary.com merriam-webster.com thesaurus.com "
                 "wolframalpha.com translate.google.com deepl.com imdb.com weather.com timeanddate.com",
})

# Sites that are never recorded. Host suffixes, then words inside a host.
SENSITIVE_HOSTS = frozenset("""
    mail.google.com outlook.live.com outlook.office.com outlook.office365.com mail.yahoo.com proton.me
    protonmail.com mail.proton.me icloud.com tutanota.com tuta.com fastmail.com hey.com
    web.whatsapp.com web.telegram.org discord.com messenger.com signal.org wa.me
    paypal.com venmo.com wise.com revolut.com chase.com wellsfargo.com bankofamerica.com citi.com capitalone.com
    americanexpress.com hsbc.com barclays.com stripe.com dashboard.stripe.com coinbase.com binance.com kraken.com
    robinhood.com fidelity.com schwab.com vanguard.com mint.com
    1password.com bitwarden.com lastpass.com dashlane.com keepass.info vault.bitwarden.com
    webmd.com mayoclinic.org healthline.com nhs.uk cdc.gov nih.gov medlineplus.gov clevelandclinic.org
    mychart.com zocdoc.com betterhelp.com talkspace.com
    accounts.google.com login.microsoftonline.com myaccount.google.com id.apple.com appleid.apple.com
    tinder.com bumble.com hinge.co grindr.com match.com okcupid.com
    gmail.com outlook.com hotmail.com live.com aol.com mail.aol.com zoho.com mail.zoho.com
    slack.com teams.microsoft.com teams.live.com skype.com chat.google.com hangouts.google.com messages.google.com
    voice.google.com element.io matrix.org mattermost.com zulip.com web.whatsapp.com business.whatsapp.com
    drive.google.com photos.google.com contacts.google.com keep.google.com myactivity.google.com takeout.google.com
    onedrive.live.com icloud.com mychart.org
    onlyfans.com fansly.com chaturbate.com xvideos.com xnxx.com redtube.com youporn.com stripchat.com livejasmin.com
    xhamster.com brazzers.com bongacams.com cam4.com myfreecams.com
""".split())
SENSITIVE_WORDS = ("bank", "porn", "xxx", "sex", "adult", "casino", "betting", "gambl", "health", "clinic",
                   "medical", "therap", "pharma", "legal", "lawyer", "attorney", "vpn", "tor", "onion", "mail",
                   "login", "signin", "auth", "account", "wallet", "invoice", "payroll", "tax", "insurance",
                   "hentai", "nsfw", "escort", "erotic", "fetish", "camgirl", "mychart", "patientportal", "dating")
# Short words match whole labels only ("tor" must not hide "history.com").
_WORD_WHOLE = frozenset({"tor", "sex", "tax", "mail", "auth", "vpn", "legal"})
SENSITIVE_TITLE = re.compile(
    r"\b(?:password|passcode|one[- ]time code|verification code|2fa|recovery (?:key|phrase)|seed phrase|"
    r"symptoms?|diagnos\w+|therap(?:y|ist)|depress\w+|anxiety|suicid\w+|self[- ]harm|rehab|std|hiv|"
    r"pregnan\w+|abortion|divorce|custody|lawyer|attorney|bankruptcy|debt|loan application|"
    r"credit card|social security|ssn|porn\w*|nsfw|escort|dating)\b", re.I)

_PRIVATE_NET = re.compile(r"^(?:10\.|127\.|169\.254\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|0\.0\.0\.0|\[?::1\]?$)")
_IP = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

# Title keywords: weak evidence, used only when the table has nothing.
KEYWORDS: List[Tuple[str, "re.Pattern"]] = [(c, re.compile(p, re.I)) for c, p in (
    ("learning", r"\b(?:tutorial|course|lesson|lecture|learn(?:ing)?|how to|beginner|crash course|exam|syllabus)\b"),
    ("code", r"\b(?:github|pull request|commit|api reference|stack ?overflow|sdk|compiler|debug\w*|"
             r"regex|npm|pip install|docker|kubernetes|typescript|python|javascript|rust|golang|"
             r"repository|repo|merge request)\b"),
    ("docs", r"\b(?:documentation|docs|manual|reference|spec(?:ification)?|readme|guide|handbook|cheat ?sheet)\b"),
    ("shopping", r"\b(?:buy|cart|checkout|sale|discount|deal|price|coupon|shop|order)\b"),
    ("news", r"\b(?:breaking|news|headlines|election|report(?:s|ed)?)\b"),
    ("video", r"\b(?:watch|episode|season|trailer|full movie|official video|stream(?:ing)?)\b"),
    ("music", r"\b(?:lyrics|album|playlist|official audio|single|discography)\b"),
    ("games", r"\b(?:walkthrough|gameplay|speedrun|patch notes|mods?)\b"),
    ("travel", r"\b(?:flights?|hotels?|itinerary|visa|things to do)\b"),
    ("reading", r"\b(?:paper|abstract|journal|chapter|novel)\b"),
    ("work", r"\b(?:meeting|invoice|proposal|roadmap|sprint|standup|ticket|jira|backlog)\b"),
)]

_TOKEN = re.compile(r"[a-z][a-z0-9+#]{2,}")
_STOP = frozenset("the and for with you your are this that from not but all can will how what why who "
                  "new com www http https page home sign login".split())


def tokens(title: str) -> List[str]:
    return [t for t in _TOKEN.findall((title or "").lower()) if t not in _STOP]


def host_of(url_or_host: str) -> str:
    """The lower-case host of a URL (or a bare host), without www, port or path."""
    s = (url_or_host or "").strip().lower()
    s = re.sub(r"^[a-z][a-z0-9+.\-]*://", "", s)
    s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    s = s.rsplit("@", 1)[-1]
    if s.startswith("["):
        return s
    s = s.split(":", 1)[0]
    return re.sub(r"^www\d?\.", "", s).strip(".")


_SLD = frozenset("co com org net gov edu ac".split())


def registrable(host: str) -> str:
    """example.co.uk -> example.co.uk, a.b.example.com -> example.com."""
    h = host.split(".")
    if len(h) <= 2:
        return host
    if len(h[-1]) == 2 and h[-2] in _SLD:
        return ".".join(h[-3:])
    return ".".join(h[-2:])


def _suffixes(host: str) -> Iterable[str]:
    parts = host.split(".")
    for i in range(len(parts)):
        yield ".".join(parts[i:])


def is_sensitive_host(host: str) -> bool:
    if not host:
        return True
    if _IP.match(host) or _PRIVATE_NET.match(host):
        return host not in ("127.0.0.1",)             # localhost development is fine, a router page is not
    if "." not in host and host != "localhost":
        return True                                   # an intranet name
    if host.endswith((".local", ".lan", ".internal", ".home", ".corp")):
        return True
    if any(s in SENSITIVE_HOSTS for s in _suffixes(host)):
        return True
    labels = re.split(r"[.\-]", host)
    for w in SENSITIVE_WORDS:
        if w in _WORD_WHOLE:
            if w in labels:
                return True
        elif w in host:
            return True
    return False


def is_sensitive(host: str, title: str = "") -> bool:
    return is_sensitive_host(host) or bool(SENSITIVE_TITLE.search(title or ""))


def site_of(host: str) -> str:
    """The name a site is kept under: the table's own key when it is a known
    sub-site (docs.google.com), else the registrable domain."""
    for s in _suffixes(host):
        if s in TABLE:
            return s
    return registrable(host)


class Classifier:
    """Category for a page, learned only from what the user says."""

    MIN_EXAMPLES = 3
    MIN_SHARE = 0.70

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._lock = threading.RLock()
        self.tags: Dict[str, str] = {}                       # site -> category, said by the user
        self.counts: Dict[str, Dict[str, int]] = {}          # category -> token -> count
        self.docs: Dict[str, int] = {}                       # category -> number of examples
        self._load()

    # -- keeping ----------------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.tags = {str(k): str(v) for k, v in (raw.get("tags") or {}).items()}
            self.counts = {c: {str(t): int(n) for t, n in d.items()} for c, d in (raw.get("counts") or {}).items()}
            self.docs = {str(c): int(n) for c, n in (raw.get("docs") or {}).items()}
        except Exception:
            self.tags, self.counts, self.docs = {}, {}, {}

    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"tags": self.tags, "counts": self.counts, "docs": self.docs}, fh)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # -- teaching ------------------------------------------------------------------
    def teach(self, host: str, category: str, title: str = "") -> bool:
        """The user said this site is `category`. Remembered for the site, and the
        title words are counted so similar pages are guessed later."""
        category = (category or "").strip().lower()
        if category not in CATEGORIES or category in ("unknown",) or is_sensitive_host(host_of(host)):
            return False
        with self._lock:
            self.tags[site_of(host_of(host))] = category
            if title:
                d = self.counts.setdefault(category, {})
                for t in set(tokens(title)):
                    d[t] = d.get(t, 0) + 1
                self.docs[category] = self.docs.get(category, 0) + 1
            self.save()
        return True

    def untag(self, host: str) -> bool:
        with self._lock:
            ok = self.tags.pop(site_of(host_of(host)), None) is not None
            if ok:
                self.save()
            return ok

    def forget_all(self):
        with self._lock:
            self.tags, self.counts, self.docs = {}, {}, {}
            self.save()

    # -- deciding ------------------------------------------------------------------
    def _learned(self, title: str) -> Optional[str]:
        toks = tokens(title)
        with self._lock:
            cats = [c for c, n in self.docs.items() if n >= self.MIN_EXAMPLES]
            if len(cats) < 1 or not toks:
                return None
            total = sum(self.docs.values())
            vocab = {t for c in self.counts.values() for t in c}
            logp: Dict[str, float] = {}
            for c in cats:
                d, n = self.counts.get(c, {}), self.docs[c]
                lp = math.log(n / total)
                for t in toks:
                    lp += math.log((d.get(t, 0) + 1) / (n + len(vocab) + 1))
                logp[c] = lp
        hit = [t for t in toks if any(t in self.counts.get(c, {}) for c in cats)]
        if len(hit) < 2:
            return None                                       # one shared word is not evidence
        m = max(logp.values())
        exp = {c: math.exp(v - m) for c, v in logp.items()}
        z = sum(exp.values())
        best = max(exp, key=exp.get)
        return best if exp[best] / z >= self.MIN_SHARE else None

    def classify(self, host: str, title: str = "") -> Tuple[str, str]:
        h = host_of(host)
        if is_sensitive(h, title):
            return "unknown", "sensitive"
        site = site_of(h)
        with self._lock:
            tag = self.tags.get(site)
        if tag:
            return tag, "user"
        for s in _suffixes(h):
            if s in TABLE:
                return TABLE[s], "table"
        for cat, rx in KEYWORDS:
            if rx.search(title or ""):
                return cat, "keyword"
        learned = self._learned(title)
        if learned:
            return learned, "learned"
        return "unknown", "unknown"

    def status(self) -> Dict[str, int]:
        with self._lock:
            return {"tags": len(self.tags), "examples": sum(self.docs.values())}
