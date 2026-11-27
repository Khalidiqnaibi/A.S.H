"""tst_cutoff.py -- 26 Nov lines that went to the model: "what is gta 6" (33 s; it said the game was not out -- facts refused any topic with a
digit), "What are the biomarkers that tend to..." and "can I be here on..." (Whisper's "..." -- cut off), and "Luxibrize." (one word, nothing).
Fakes only; no network.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_cutoff.py"""
import os
import sys
import tempfile
import urllib.error
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon.facts import Facts, FactsConfig, topic_of  # noqa: E402

check = tw.check
GTA = {"type": "standard", "extract": "Grand Theft Auto VI is an upcoming action-adventure game developed by Rockstar Games. It is due to be released "
       "in 2026."}


class Fetch:
    """Pages by title; Wikipedia's search by query. Anything else is a 404."""
    def __init__(self, pages, search=None):
        self.pages, self.search, self.calls = pages, search or {}, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if "action=opensearch" in url:
            q = urllib.parse.unquote(url.split("search=", 1)[1])
            hit = self.search.get(q.lower())
            return [q, [hit] if hit else [], [], []]
        title = urllib.parse.unquote(url.rsplit("/", 1)[1]).replace("_", " ")
        if title in self.pages:
            return self.pages[title]
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)


def main():
    print("\n=== C1. A name with a number in it is a topic ===")
    for q, want in (("what is gta 6", "gta 6"), ("what is Windows 11", "Windows 11"), ("what's a PS5", "PS5"), ("what is COVID-19", "COVID-19"),
                    ("tell me about python 3.12", "python 3.12")):
        check(f"{q!r} -> {want!r}", topic_of(q) == want, str(topic_of(q)))
    for q in ("what is 15 times 23", "what is 5 km in miles", "what is 1984", "what is 7 + 8", "what is 3 hours", "what is my score 10"):
        check(f"still not a topic: {q!r}", topic_of(q) is None, str(topic_of(q)))

    print("\n=== C2. Found under its acronym, or by Wikipedia's own search ===")
    f = Fetch({"GTA 6": GTA})
    say = Facts(FactsConfig(enabled=True), fetch=f).hear("what is gta 6")
    check("'gta 6' is tried as 'GTA 6'", say and say.startswith("Grand Theft Auto VI is") and say.endswith("(Wikipedia)"), str(say))
    f = Fetch({"Grand Theft Auto VI": GTA}, search={"gta 6": "Grand Theft Auto VI"})
    say = Facts(FactsConfig(enabled=True), fetch=f).hear("what is gta 6")
    check("no page under the words said: Wikipedia's search finds it", say and say.startswith("Grand Theft Auto VI is"), str(say))
    check("...sending only the topic", all(("gta" in u.lower()) or ("Grand" in u) for u in f.calls), str(f.calls))
    f = Fetch({})
    say = Facts(FactsConfig(enabled=True), fetch=f).hear("what is zxqvb")
    check("nothing anywhere: said so", say == "I couldn't find a Wikipedia page for “zxqvb”.", str(say))

    print("\n=== C3. Cut off ===")
    for line, want in (("What are the biomarkers that tend to...", "To what? I lost the end of that."),
                       ("Ash, can I be here on...", "On what? I lost the end of that."),
                       ("Switch to Gemini's…", "I only caught part of that -- say the rest?")):
        rt = tm.rig()
        used, said = rt.say(line, source="microphone")
        check(f"heard {line!r} -> {want!r}", not used and said and said[-1] == want, f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("What are the biomarkers that tend to...", source="console")
    check("typed with '...': not taken as cut (people type that)", not (said and said[-1] == "To what? I lost the end of that."), f"{used} {said}")
    rt = tm.rig()
    rt.say("What are the biomarkers that tend to...", source="microphone")
    used, said = rt.say("What are the biomarkers that tend to...", source="microphone")
    check("said the same way again: it goes on", used, f"{used} {said}")

    print("\n=== C4. One word, taken by nothing ===")
    rt = tm.rig()
    used, said = rt.say("Luxibrize.", source="microphone")
    check("'Luxibrize.' -> 'Sorry?', no model", not used and said == ["Sorry?"], f"{used} {said}")
    used, said = rt.say("Luxibrize.", source="microphone")
    check("...said again, it goes on", used, f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("Wink.", source="microphone")
    check("a one-word command still works", said and said[-1] != "Sorry?", f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("Thanks.", source="microphone")
    check("...and a thanks", said and said[-1] != "Sorry?", f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
