"""
tst_honest.py -- ASH telling the truth about itself, and small fixes from the
session of 2026-11-07.

  H1  "What can you do?" is answered from what is really on
  H2  What ASH can't do (browse the web, send mail, call, buy) is said
      plainly -- and ordinary requests are left alone
  H3  A model answer loses its pasted "Facts:" block, fake tool calls in
      brackets, and any address, number or link that was never given to it
  H4  In the runtime: rule answers with no model, only to lines meant for ASH;
      the sanitiser on what the model says
  H5  Small things: "switch screen capture on", a smile mid-conversation
      without the name, a short "yeah" is kept
  H6  The intent-embedding cache remembers what it was built from
  H7  The memory cleanup tool: shows first, backs up, keeps what is real

Run:  PYTHONIOENCODING=utf-8 python tests/tst_honest.py
"""

import json
import os
import pickle
import sys
import tempfile
import time
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import honest  # noqa: E402

check = tw.check
T0 = tw.T0


def h1_capabilities():
    print("\n=== H1. What can you do ===")
    for t in ("what can you do?", "What can you do for now, Ash?", "what are your capabilities", "help",
              "what can I ask you", "what are you good at", "hey ash what could you do for me"):
        check(f"{t!r}", honest.asks_capabilities(t))
    for t in ("help me write a poem", "what can you do with this file about trees", "how do I help my dog", "", None):
        check(f"not a capability question: {str(t)[:34]!r}", not honest.asks_capabilities(t))
    bare = NS()
    s = honest.capabilities(bare)
    check("nothing configured: only what always works", "calculate, convert units and read files" in s
          and "reminders" not in s and "lights" not in s, s)
    full = NS(reminders=object(), watcher=object(), desktop=object(), media=object(), play=object(), research=object(),
              tasks=object(), smarthome=NS(cfg=NS(url="http://ha:8123")), weather=NS(configured=True),
              calendar=NS(cfg=NS(files=("a.ics",), urls=())), inbox=NS(configured=True))
    s = honest.capabilities(full)
    for want in ("reminders, timers and alarms", "open, switch to and close your apps", "control the music",
                 "lights and plugs", "the weather", "what's on your calendar", "who emailed you", "play games"):
        check(f"configured: mentions {want!r}", want in s, s)
    check("lights only when a smart home is set", "lights" not in honest.capabilities(NS(smarthome=NS(cfg=NS(url="")))))
    check("weather only when a place is set", "weather" not in honest.capabilities(NS(weather=NS(configured=False))))
    check("calendar only when a source is set", "calendar" not in honest.capabilities(NS(calendar=NS(cfg=NS(files=(), urls=())))))
    check("email only when an account is set", "emailed" not in honest.capabilities(NS(inbox=NS(configured=False))))
    check("never claims what it can't do", "send email or messages, or make calls" in s and "I can't browse the web" in s
          and "I can send" not in s)


def h2_limits():
    print("\n=== H2. What it can't do ===")
    web = ("Show me pictures of cats online.", "search the web for pizza places", "google python decorators",
           "find me some images of volcanoes", "look up the weather on the internet", "open a new tab",
           "show me videos of cats", "ash, search the internet for jazz")
    for t in web:
        r = honest.limit_reply(t)
        check(f"web: {t!r}", r is not None and "can't browse the web" in r, str(r))
    for t, key in (("send an email to Sara", "only read your email"), ("send a message to Omar", "only read your email"),
                   ("email Omar that I'm late", "only read your email"), ("text Sara about the meeting", "messages"),
                   ("make a phone call", "phone calls"), ("call my mom", "phone calls"), ("dial 555 1234", "phone calls"),
                   ("order a pizza", "buy things"), ("buy me a laptop online", "buy things"),
                   ("order me a coffee", "buy things")):
        r = honest.limit_reply(t)
        check(f"{t!r}", r is not None and key in r, str(r))
    check("a long rambling line is not taken for a request", honest.limit_reply("show me pictures of cats online " + "really " * 30) is None)
    for t in ("Find the height of Lionel Messi", "show me the files in downloads", "what's the weather", "call me Immortal",
              "order the files by date", "google it", "open chrome", "search my files for the plan", "look up quasar",
              "what time is it", "tell me about cats", "send me the answer", "email", "", None, "x" * 200):
        check(f"left alone: {str(t)[:34]!r}", honest.limit_reply(t) is None)


def h3_sanitize():
    print("\n=== H3. After the model answers ===")
    r = ("Alright, here's how I can help.\n\n[web_browser open new tab]\n[search_engine perform search for cats]\n\n"
         "I'll let you scroll.\n\nFacts:\n{\n  \"intent\": \"clarification\",\n  \"tool_used\": null\n}")
    check("fake tool calls and the pasted Facts block go", honest.sanitize(r) == "Alright, here's how I can help.\n\nI'll let you scroll.",
          repr(honest.sanitize(r)))
    check("an inline fake call goes too", honest.sanitize("Sure [web_browser open new tab] thing.") == "Sure thing.")
    check("a brackets-and-words note is not a tool call", honest.sanitize("See [the docs] now.") == "See [the docs] now.")
    check("a trailing JSON briefing goes", honest.sanitize('Done.\n{"intent": "x", "tool_used": null}') == "Done.")
    cv = ("His contact details include a phone number, an email, and a location.\nPhone: +212 XXX-XXXXXX\n"
          "Email: khalid.iqnaibi@email.com\nLocation: Morocco")
    out = honest.sanitize(cv, known="")
    check("an invented phone and address are taken out", "+212" not in out and "@" not in out
          and out.startswith("His contact details"), out)
    check("a placeholder is never real, even if 'known'", "XXX" not in honest.sanitize("Call +212 XXX-XXXXXX.",
                                                                                   known="+212 XXX-XXXXXX"))
    check("an address the user gave is kept", honest.sanitize("Your email is khalidqnabi@gmail.com.",
                                                              known="my email is khalidqnabi@gmail.com") ==
          "Your email is khalidqnabi@gmail.com.")
    check("a number matches however it is spaced", honest.sanitize("Call 555-123-4567.", known="it is 555 123 4567")
          == "Call 555-123-4567.")
    check("a link from nowhere goes", honest.sanitize("See https://made.up/x for more.") == "I don't have that detail.")
    check("a link that was given stays", honest.sanitize("See https://real.org/p.", known="https://real.org/p") ==
          "See https://real.org/p.")
    plain = "Python 3.11 came out on 2022-10-24, runs at 3.5 GHz, and we met at 10:30 on v1.2.3."
    check("dates, versions and times are not phone numbers", honest.sanitize(plain) == plain, honest.sanitize(plain))
    asked = []
    honest.sanitize("No contact details here.", known=lambda: asked.append(1) or "")
    check("what was given is only looked up when needed", asked == [])
    honest.sanitize("Call 555-123-4567.", known=lambda: asked.append(1) or "")
    check("...and then it is", asked == [1])
    check("only fake calls: nothing left", honest.sanitize("[web_browser open new tab]") == "")
    check("empty in, empty out", honest.sanitize("") == "" and honest.sanitize(None) == "")


def rig(reply="Sure."):
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import Modality, PeripheryRegistry, SensorEvent
    clock = [T0]
    out, metas = [], []

    class Ash(tw.Ash):
        def think(self, query, image=None, context=""):
            self.calls.append(query)
            r = tw.Resp(reply(query) if callable(reply) else reply)
            r.pathway = type("P", (), {"value": "slow"})()
            return r
    ash = Ash()
    rt = AmbientRuntime(ash=ash, periphery=PeripheryRegistry(), journal=Journal(tempfile.mkdtemp()),
                        gate=AttentionGate(), privacy=PrivacyPolicy(),
                        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, reminders=True,
                                             reminders_path=os.path.join(tempfile.mkdtemp(), "r.json")),
                        response_sink=lambda t, m: (out.append(t), metas.append(m)))
    rt.clock = lambda: clock[0]

    def say(text, source="console", addressed=True):
        clock[0] += 3
        rt.submit(SensorEvent(source=source, modality=Modality.TEXT, text=text, addressed=addressed, salience=0.9,
                              urgency=0.6))
        rt._tick()
    rt.say, rt.out, rt.ash_, rt.t = say, out, ash, clock
    return rt


def h4_runtime():
    print("\n=== H4. In the runtime ===")
    rt = rig()
    rt.say("Show me pictures of cats online.")
    check("a plain 'I can't', no model", "can't browse the web" in rt.out[-1] and rt.ash_.calls == [], str(rt.out))
    rt.say("what can you do for now, Ash?")
    check("what it can do, from what is on, no model", rt.out[-1].startswith("I can set reminders") and rt.ash_.calls == [],
          rt.out[-1])
    rt.say("send an email to Omar")
    check("sending mail: said plainly", "only read your email" in rt.out[-1] and rt.ash_.calls == [])
    n = len(rt.out)
    rt.say("show me pictures of cats online", source="microphone", addressed=False)
    check("said to the room, not to ASH: the rule does not answer it", not any("can't browse" in o for o in rt.out[n:]),
          str(rt.out[n:]))
    rt.say("tell me about the roman empire")
    check("an ordinary question still goes to the model", rt.ash_.calls[-1] == "tell me about the roman empire",
          str(rt.ash_.calls))

    rt = rig("Here you go.\n[web_browser open new tab]\n\nFacts:\n{\n \"intent\": \"x\"\n}")
    rt.say("what do you think of jazz")
    check("the model's pasted briefing and fake call never reach the user", rt.out[-1] == "Here you go.", repr(rt.out[-1]))
    rt = rig("His email is khalid.iqnaibi@email.com. He lives in Morocco.")
    rt.say("what do you know about me")
    check("a sentence with an invented address is taken out, the rest kept", rt.out[-1] == "He lives in Morocco.",
          rt.out[-1])
    rt = rig(lambda q: "Your email is khalidqnabi@gmail.com." if "what is my email" in q else "Noted.")
    rt.say("my email is khalidqnabi@gmail.com, remember it")
    rt.say("what is my email")
    check("one the user gave in this conversation stays", "khalidqnabi@gmail.com" in rt.out[-1], rt.out[-1])
    rt = rig("[web_browser open new tab]")
    rt.say("do the thing")
    check("nothing left to say: it says so", rt.out[-1] == "I'm not sure how to help with that.", rt.out[-1])
    rt = rig("Phone: +212 XXX-XXXXXX.")
    rt.say("what is his phone")
    check("only an invented number: 'I don't have that detail'", rt.out[-1] == "I don't have that detail.", rt.out[-1])


def h5_small():
    print("\n=== H5. Small things ===")
    from src.daemon.voicectl import parse
    for t, want in (("Switch screen capture on.", "screen_on"), ("turn screen capture on", "screen_on"),
                    ("switch the screen capture off", "screen_off"), ("turn screen capture off", "screen_off"),
                    ("switch on screen capture", "screen_on"), ("screen capture on", "screen_on")):
        c = parse(t)
        check(f"{t!r} -> {want}", c is not None and c.name == want, str(c))
    rt = rig()
    rt.periphery.sensors  # (a real runtime: no face)
    from src.face.bridge import FaceBridge

    class Player:
        library = {"happy": object()}

        def __init__(self):
            self.played = []

        def play(self, name, force=False):
            self.played.append(name)
            return True

        def set_idle(self, n):
            pass
    rt.face = FaceBridge(Player())
    rt.say("hello ash")
    from src.senses import address
    address.conversation.note_ash_spoke(rt.t[0])
    rt.say("Smile for me.", source="microphone", addressed=False)
    check("mid-conversation, a smile without the name works", rt.face.player.played == ["happy"], str(rt.out[-2:]))
    rt.t[0] += 600
    rt.say("Smile for me.", source="microphone", addressed=False)
    check("long after the conversation, it does not", rt.face.player.played == ["happy"])
    rt.say("activate camera", source="microphone", addressed=False)
    check("(and a sensor still needs the name)", "Camera on" not in " ".join(rt.out[-1:]))

    from src.senses.mic import trusted_text

    def seg(text, lp, ns):
        return NS(text=text, avg_logprob=lp, no_speech_prob=ns)
    check("a short 'Yeah.' that whisper is unsure of is kept", trusted_text([seg(" Yeah.", -0.73, 0.58)]) == "Yeah.")
    check("a short reply that is very unsure is still dropped", trusted_text([seg(" Yeah.", -0.95, 0.58)]) == "")
    check("a longer not-speech phrase is dropped", trusted_text([seg(" Thanks for watching", -0.73, 0.58)]) == "")


def h6_cache():
    print("\n=== H6. The intent cache ===")
    import tools.classification as cl
    a = cl.EmbeddingCatalog.__new__(cl.EmbeddingCatalog)
    a.labels, a.descriptions, a.embeddings, a._source_mtime = ["x"], ["d"], [[1.0]], 1234.5
    p = os.path.join(tempfile.mkdtemp(), "emb.pkl")
    a.save_cache(p)
    check("the stamp is saved in the file", pickle.load(open(p, "rb")).get("source_mtime") == 1234.5)
    b = cl.EmbeddingCatalog.__new__(cl.EmbeddingCatalog)
    b.labels = b.descriptions = b.embeddings = None
    b._source_mtime = None
    check("and read back", b.load_cache(p) and b._source_mtime == 1234.5, str(b._source_mtime))
    src = open(os.path.join("tools", "classification.py"), encoding="utf-8").read()
    i, j = src.index("_INTENT_CATALOG.build_from_json_blocks(intents_blocks)"), src.index("save_cache(INTENT_EMB_FNAME)")
    check("it is stamped before it is saved", "_source_mtime = intents_mtime" in src[i:j])


def h7_cleanup():
    print("\n=== H7. The memory cleanup ===")
    sys.path.insert(0, "tools")
    import memory_cleanup as mc
    d = tempfile.mkdtemp()
    eps = [
        {"episode_id": "1", "timestamp": T0, "summary": "A.S.H: Phone: +212 XXX-XXXXXX, Location: Morocco", "importance": 0.5},
        {"episode_id": "2", "timestamp": T0, "summary": "A.S.H [tool]: It's 5 pm.", "importance": 0.5},
        {"episode_id": "3", "timestamp": T0, "summary": "Immortal: I'm from Palestine", "importance": 0.5},
        {"episode_id": "4", "timestamp": T0, "summary": "A.S.H: I am doing well, thanks.", "importance": 0.5},
    ]
    with open(os.path.join(d, "episodic_memory.jsonl"), "w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps(e) + "\n")
        fh.write("not json at all\n")

    def ent(name, i):
        return {"entity_id": str(i), "entity_type": "proper", "primary_identifiers": {"name": name.lower()},
                "attributes": {"canonical_name": name}}
    names = ["Omar", "Contents", "Summary", "Technical Skills", "khalidqnabi@gmail.com", "khalid.iqnaibi@email.com",
             "Let", "Raspberry Pi", "x@y.com\n- Location", "Palestine"]
    json.dump([ent(n, i) for i, n in enumerate(names)], open(os.path.join(d, "entities.json"), "w"))
    p = mc.plan(d)
    check("ASH's own prose is picked out; a tool's reply and the user's words are not",
          [e["episode_id"] for e in p["drop_episodes"]] == ["1", "4"], str([e["episode_id"] for e in p["drop_episodes"]]))
    why = {mc._name(e): w for e, w in p["drop_entities"]}
    check("headings, sentence starters, made-up addresses and broken text are picked out",
          set(why) == {"Contents", "Summary", "Technical Skills", "khalid.iqnaibi@email.com", "Let", "x@y.com\n- Location"},
          str(sorted(why)))
    check("a real address, real names and places stay", "khalidqnabi@gmail.com" not in why and "Omar" not in why
          and "Raspberry Pi" not in why and "Palestine" not in why)
    before = open(os.path.join(d, "episodic_memory.jsonl"), encoding="utf-8").read()
    rc = mc.main(["--dir", d])
    check("by default nothing changes", rc == 0 and open(os.path.join(d, "episodic_memory.jsonl"),
                                                         encoding="utf-8").read() == before)
    b1, b2 = mc.apply(mc.plan(d))
    check("--apply backs up first", open(b1, encoding="utf-8").read() == before and os.path.exists(b2))
    left = mc.read_jsonl(os.path.join(d, "episodic_memory.jsonl"))
    check("then removes only those episodes (and keeps a damaged line as it was)",
          [e.get("episode_id", "?") for e in left] == ["2", "3", "?"], str(left))
    kept = [mc._name(e) for e in json.load(open(os.path.join(d, "entities.json")))]
    check("and those entities", kept == ["Omar", "khalidqnabi@gmail.com", "Raspberry Pi", "Palestine"], str(kept))
    p2 = mc.plan(d)
    check("a second run has nothing left to do", not p2["drop_episodes"] and not p2["drop_entities"])
    empty = tempfile.mkdtemp()
    check("an empty folder is fine", mc.main(["--dir", empty]) == 0)


if __name__ == "__main__":
    for fn in (h1_capabilities, h2_limits, h3_sanitize, h4_runtime, h5_small, h6_cache, h7_cleanup):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
