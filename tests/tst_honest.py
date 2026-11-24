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
  H11 "What can you do then?" (a trailing word) and capability questions worded unlike the pattern are answered by rule, never by the model

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
    for before, after in [("The capital of Jordan is Amman. What would you like to discuss while you're here?", "The capital of Jordan is Amman."),
                          ("I see you're watching a video. How can I assist you today?", "I see you're watching a video."),
                          ("Segue means a smooth transition. Is there anything else I can help with?", "Segue means a smooth transition."),
                          ("It's 3 pm.\n\nLet me know if you need anything else!", "It's 3 pm."),
                          ("Done. Feel free to ask me anything. I'm here to help.", "Done.")]:
        check(f"a closing offer goes: {before[-40:]!r}", honest.sanitize(before) == after, repr(honest.sanitize(before)))
    for keep in ("What would you like to discuss?", "How can I assist you?", "Let me know if you need anything.", "I can help with that. Which file do you mean?",
                 "Is there anything else I can help with? Also, your battery is low.", "I'm here to help, and the answer is 42."):
        check(f"nothing real before it, or it is not last: kept ({keep[:36]!r})", honest.sanitize(keep) == keep or "battery" in honest.sanitize(keep) or "42" in honest.sanitize(keep), repr(honest.sanitize(keep)))
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
    for q in ("what do you think of my code", "explain the code to me", "Ash, can you review my code please"):
        n = len(rt.ash_.calls)
        rt.say(q)
        check(f"{q!r}: with the screen reader off it says it can't see it, at once, no model",
              rt.out[-1].startswith("I can't see it") and "look at my screen" in rt.out[-1] and len(rt.ash_.calls) == n, rt.out[-1])
    n = len(rt.ash_.calls)
    rt.say("what do you think of my girlfriend")
    check("...and other opinions are the model's", len(rt.ash_.calls) == n + 1)
    for q in ("What's your name?", "who are you", "tell me about yourself"):
        n_self = len(rt.ash_.calls)
        rt.say(q)
        check(f"{q!r}: ASH says who it is, no model", rt.out[-1].startswith("I'm ") and "what can you do" in rt.out[-1] and len(rt.ash_.calls) == n_self, rt.out[-1])
    n_w = len(rt.ash_.calls)
    rt.say("What can you do with Google Docs?")
    check("'what can you do with <a thing it has nothing for>': said so, no model, no invented features", rt.out[-1].startswith("Nothing on my own with Google Docs") and len(rt.ash_.calls) == n_w, rt.out[-1])
    check("'can you open notepad for me please' is a command, not a capability question (it was caught as 'what can you do with X')",
          honest.with_subject(rt, "Ash, can you open notepad for me please") is None and honest.with_subject(rt, "can you edit my file") is None)
    rt.say("what can you do with reminders")
    check("...and with a thing it has, from its own manual", "remind me" in rt.out[-1].lower() and len(rt.ash_.calls) == n_w, rt.out[-1])
    rt.say("send an email to Omar")
    check("sending mail: said plainly", "only read your email" in rt.out[-1] and len(rt.ash_.calls) == n + 1)
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


def h8_quiet_modes():
    print("\n=== H8. Ways to make it quieter, and 'mute' ===")
    for q in ("What modes do you have for minimum interventions?", "how do I make you quieter", "how can I stop you interrupting",
              "what ways are there to make you interrupt less"):
        check(f"{q!r} gets the list of what exists", "mute" in (honest.quiet_modes_reply(q) or "") and "focus" in honest.quiet_modes_reply(q),
              str(honest.quiet_modes_reply(q)))
    for q in ("what modes does a camera have", "how do I make you laugh", "what ways are there to cook rice", "mute the music"):
        check(f"{q!r} is not that question", honest.quiet_modes_reply(q) is None)
    rt = rig()
    rt.say("what modes do you have for minimum interventions?")
    check("in the runtime: answered by rule, no model", rt.out[-1].startswith("These make me quieter") and rt.ash_.calls == [], rt.out[-1])
    from src.daemon.voicectl import parse
    rt.ports = {"console": 8788, "control": 8787, "tabs": 47831}
    rt.say("what port are you on?")
    check("'what port are you on' comes from its own ports, no model", "console is on port 8788" in rt.out[-1] and "8787" in rt.out[-1]
          and "47831" in rt.out[-1] and "this computer only" in rt.out[-1].lower(), rt.out[-1])
    rt.ports = {}
    rt.say("what port are you on")
    check("with none open it says so", "not listening" in rt.out[-1], rt.out[-1])
    check("'what tools do you have' is the capability list", honest.asks_capabilities("what tools do you have")
          and honest.asks_capabilities("What abilities do you have?"))
    check("'what port is the printer on' is not its business", honest.ports_reply(rt, "what port is the printer on") is None)
    check("'mute' and 'mute yourself' are quiet mode; muting the music is not", parse("mute").name == "sleep"
          and parse("mute yourself").name == "sleep" and parse("mute the music") is None)


def h9_invented():
    print("\n=== H9. Made-up actions, results and places ===")
    for t in ("Minimum interventions mode activated. I'll mute notifications.", "Sure! Here are some cat pictures (pause) enjoy",
              "I found a few restaurants near your dad's clinic: Cafe X and Y.", "Focus mode enabled."):
        check(f"caught: {t[:50]!r}", honest.invented_reply(t) is not None, t)
    for t in ("Sure, I can open Chrome for you.", "Here is the answer: 42.", "I found your note on that.", "The game mode kicks in when you play."):
        check(f"left alone: {t[:50]!r}", honest.invented_reply(t) is None, t)
    check("a pretend mode gets the real list of ways to be quiet", honest.invented_reply("Minimum interventions mode activated.") == honest.QUIET_MODES)
    check("pretend pictures get 'I have no tool'", "no tool" in honest.invented_reply("Here are some cat pictures"))
    for q in ("Restaurants near my dad's clinic", "find me a good pharmacy near me", "show me restaurants near the university"):
        check(f"{q!r}: no invented names, an honest 'I can't'", "can't look up places" in (honest.limit_reply(q) or ""), str(honest.limit_reply(q)))
    check("talk about restaurants is left to the model", honest.limit_reply("tell me about restaurants") is None)
    rt = rig("Minimum interventions mode activated. I'll mute notifications.")
    rt.say("do the thing for the minimum interventions please")
    check("in the runtime, the pretend answer is replaced before it is said", rt.out[-1].startswith("These make me quieter"), rt.out[-1])
    rt = rig("Here are some cat pictures (pause) enjoy")
    rt.say("show me something cute")
    check("...pictures too", "no tool" in rt.out[-1], rt.out[-1])
    rt = rig()
    rt.say("restaurants near my dad's clinic")
    check("...and a places question is answered by rule, never reaches the model", "can't look up places" in rt.out[-1] and rt.ash_.calls == [], rt.out[-1])


def h10_disputes():
    print("\n=== H10. 'That's wrong' is not an invitation to invent a defence ===")
    for q in ("That's wrong.", "What you just said is wrong.", "All of this is wrong actually. You are just making up stuff.", "you made that up",
              "You're wrong", "no that's not true", "that is totally false bro"):
        check(f"{q!r} is a dispute", honest.disputes_last_answer(q), q)
    for q in ("what's wrong with my pc", "that's wrong, I meant the other one", "it's not wrong", "you are making a sandwich", "wrong"):
        check(f"{q!r} is not", not honest.disputes_last_answer(q), q)
    rt = rig("Version 10.1.2.")
    rt.say("what version are you")
    n = len(rt.ash_.calls)
    rt.say("what you just said is wrong")
    check("by rule, no model: it says it cannot back it up and asks for the right one", rt.out[-1] == honest.DISPUTE_FIRST and len(rt.ash_.calls) == n,
          rt.out[-1])
    rt.say("all of this is wrong actually. you are just making up stuff")
    check("said again about the same answer: it does not defend it, and does not repeat itself", rt.out[-1] == honest.DISPUTE_AGAIN and len(rt.ash_.calls) == n,
          rt.out[-1])
    rt2 = rig()
    rt2.say("that's wrong")
    check("with nothing just said, it is left to the others", rt2.out[-1:] != [honest.DISPUTE_FIRST], str(rt2.out[-1:]))
    for t in ("I'm remembering that for you, Immortal.", "I'll make sure to remember that you're on port 8788."):
        check(f"{t[:40]!r}: nothing was stored, so it is not said", honest.invented_reply(t) == honest.NOT_STORED, t)
    check("'I remember that day' is just talk", honest.invented_reply("I remember that day fondly.") is None)
    rt3 = rig("I'll make sure to remember that you're on port 8-7-8.")
    rt3.say("so what port am I on")
    check("in the runtime, the pretend memory is replaced", rt3.out[-1] == honest.NOT_STORED, rt3.out[-1])


def h12_false_reports():
    print("\n=== H12. What the model said it did, or said it can't do ===")
    for t in ("I did title the note you asked me to create earlier, so it should be there now in that folder.",
              "I checked the recent history and didn't see any new Samsung notes added since your last one was titled.",
              "That's wrong, I remember the records show Immortal and Khalid are separate entries.",
              "Got it. I'll create the new Samsung note called Lecture 5 and type hash tables are fast.",
              "I've already saved the file for you."):
        check(f"caught: {t[:50]!r}", honest.false_report(t) == honest.NOT_DONE_NOTE, str(honest.false_report(t)))
    for t in ("I can't open Samsung Notes directly from here, but you can just click the app.", "I can't launch apps directly like that."):
        check(f"a false 'can't open': {t[:40]!r}", honest.false_report(t) == honest.CAN_OPEN, str(honest.false_report(t)))
    for t in ("I've taken a quick look at your code. It's quite impressive, Immortal. Your syntax is clean, and I can see you've put some thought into the structure.",
              "I reviewed your code and it looks fine.", "I can see you've been busy."):
        check(f"looked-at-nothing is caught: {t[:40]!r}", honest.false_report(t) == honest.NOT_SEEN, str(honest.false_report(t)))
    for t in ("Alright, I can help with that. Let me mute the microphone for an hour. I'll be quiet during your meeting.", "I'll mute myself now."):
        check(f"a mute that did not happen: {t[:40]!r}", honest.false_report(t) == honest.NOT_MUTED, str(honest.false_report(t)))
    check("...but not when it really is muted", honest.false_report("Let me mute the microphone for you.", muted=True) is None)
    check("'I can see' alone is left alone", honest.false_report("I can see why you'd ask.") is None and honest.false_report("Muting is easy: say mute.") is None)
    check("a false 'can't make faces'", honest.false_report("I can't move my face, even with all that motion data.") == honest.CAN_FACE)
    check("...unless there really is no face", honest.false_report("I can't move my face.", can_face=False) is None)
    for t in ("You can open Samsung Notes from the start menu.", "I didn't title that note.", "Did you check the history?", "The note is titled Lecture 5.",
              "I can't browse the web.", "Your records show a strong year.", "I opened up to my friend about it, you said.", ""):
        check(f"left alone: {t[:50]!r}", honest.false_report(t) is None, str(honest.false_report(t)))
    rt = rig("I did title the note you asked me to create earlier, so it should be there now in that folder.")
    rt.say("ash did you title the note")
    check("in the runtime the false report is replaced before it is said", rt.out[-1] == honest.NOT_DONE_NOTE, rt.out[-1])


def h11_trailing_and_meaning():
    print("\n=== H11. A trailing word; by meaning ===")
    for said in ("What can you do then?", "what can you do exactly", "what can you do now ash", "what do you do again", "what can you do for me please"):
        check(f"{said!r} is a capability question", honest.asks_capabilities(said), said)
    for said in ("can you help me", "what can i do about this error", "what can you see", "what did you do yesterday", "what can you do about the noise upstairs"):
        check(f"{said!r} is not", not honest.asks_capabilities(said), said)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import semantic_fake as sf
    import tst_effects as te
    rt = te.rig(chooser_embed=sf.embed, chooser_dir=tempfile.mkdtemp())
    n, k = len(rt.ash.calls), len(rt.out)
    rt.say("What can you do then?")
    check("in the runtime the trailing word no longer sends it to the model", len(rt.out) > k and len(rt.ash.calls) == n, str(rt.out[-1:]))
    n, k = len(rt.ash.calls), len(rt.out)
    rt.say("what kind of tasks can you handle")
    check("worded unlike the pattern, it is understood by meaning (no model)", len(rt.out) > k and len(rt.ash.calls) == n, str(rt.out[-1:]))
    n = len(rt.ash.calls)
    rt.say("what should I cook tonight")
    check("an ordinary question still goes to the model", len(rt.ash.calls) > n)
    bare = te.rig()
    n = len(bare.ash.calls)
    bare.say("what kind of tasks can you handle")
    check("without an encoder the pattern alone decides: that wording goes to the model", len(bare.ash.calls) > n)


if __name__ == "__main__":
    for fn in (h1_capabilities, h2_limits, h3_sanitize, h4_runtime, h5_small, h6_cache, h7_cleanup, h8_quiet_modes, h9_invented, h10_disputes, h11_trailing_and_meaning, h12_false_reports):
        fn()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
