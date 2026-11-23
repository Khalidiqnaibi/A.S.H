"""
src/daemon/honest.py

Keeping ASH's answers true about itself.

In one session the model, left to itself:

  * listed "manage your calendar, send and receive emails, make phone calls"
    when asked what it can do (it can do none of the last two)
  * said "Alright, I can use a web browser to search for pictures of cats",
    printed "[web_browser open new tab]" as if it had run a tool, and told
    the user to "scroll through the results"
  * ended an answer with "Facts: { "intent": "clarification", ... }" -- its own
    briefing, read aloud
  * produced "Phone: +212 XXX-XXXXXX, Email: khalid.iqnaibi@email.com" for
    someone, from nothing

So, by rule and before the model gets a chance:

    "what can you do?"            the list of what is really switched on here
    "show me cats online",        what ASH cannot do (browse the web, send
    "send an email to Sara",      email or messages, make calls, buy things),
    "call Omar", "order a pizza"  said plainly, with what it *can* do instead

and after the model has answered, `sanitize` takes out the things it should
never have said: a pasted "Facts:" block, fake tool calls in square brackets,
and any e-mail address, phone number or link that is nowhere in what ASH was
actually given.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_A = (r"^\s*(?:(?:hey|hi|ok(?:ay)?)\s+)?(?:ash[,\s]+)?"
      r"(?:(?:please|can\s+you|could\s+you|are\s+you\s+(?:able|going)\s+to|do\s+you\s+know\s+how\s+to)[,\s]+)*")
_Z = r"(?:[,\s]+(?:ash|please|now|for\s+me|right\s+now|for\s+now|today))*\s*[.!?]*\s*$"

_CAPS = re.compile(_A + r"(?:what\s+(?:can|could|do)\s+you\s+(?:do|help\s+(?:me\s+)?with|know\s+how\s+to\s+do)"
                   r"|what\s+(?:tools|abilities|powers|functions|skills|features)\s+(?:do\s+you\s+(?:have|use|got)|can\s+you\s+use|are\s+available)"
                   r"|what\s+are\s+(?:all\s+)?(?:your|the)\s+(?:capabilities|abilities|features|skills|commands)"
                   r"|what\s+can\s+i\s+(?:ask|say\s+to|tell)\s+you|what\s+(?:are\s+)?you\s+(?:good|able)\s+(?:at|to\s+do)"
                   r"|what\s+(?:are|is)\s+your\s+(?:limits|limitations|restrictions|weaknesses)|what\s+can'?t\s+you\s+do"
                   r"|help(?:\s+me)?|list\s+(?:your\s+)?(?:commands|features))" + _Z, re.I)

_WEB = re.compile(_A + r"(?:(?:show|find|get|fetch|search|look\s+up|look\s+for|pull\s+up|bring\s+up)\b[^.?!]{0,60}"
                  r"\b(?:online|on\s+the\s+(?:web|internet|net)|on\s+google|on\s+youtube|in\s+google|from\s+the\s+(?:web|internet))\b"
                  r"|(?:search|browse|surf|google)\s+(?:the\s+)?(?:web|internet|net)\b"
                  r"|google\s+(?!it\b)\S+|(?:open|go\s+to)\s+(?:a\s+)?(?:new\s+)?(?:browser\s+)?tab"
                  r"|(?:show|find)\s+me\s+(?:some\s+)?(?:pictures|photos|images|videos|memes)\s+of\b)", re.I)
_MSG = re.compile(_A + r"(?:send\s+(?:an?\s+|the\s+|my\s+)?(?:e-?mail|mail|message|text|sms|whatsapp|dm)\b"
                  r"|(?:e-?mail|text|message|whatsapp|dm)\s+(?!me\b)[A-Z]?[a-z]+\s+(?:that|about|to|saying|and)\b"
                  r"|reply\s+to\s+(?:the\s+|that\s+|his\s+|her\s+)?(?:e-?mail|message|text))", re.I)
_CALL = re.compile(_A + r"(?:make\s+(?:a\s+)?(?:phone\s+)?call\b|(?:phone|dial|ring)\s+\S+"
                   r"|call\s+(?:my\s+)?(?:mom|mum|dad|wife|husband|brother|sister|boss|friend|home|the\s+\w+|\+?\d[\d\s-]{5,}))", re.I)
_BUY = re.compile(_A + r"(?:(?:order|buy|purchase|book|reserve)\s+(?:me\s+)?(?:an?\s+|some\s+|the\s+)?\S+"
                  r"(?:\s+\S+){0,5}?\s+(?:online|from\s+\w+|on\s+(?:amazon|ebay|talabat|uber)|for\s+me|delivered)\b"
                  r"|order\s+(?:me\s+)?(?:a\s+|some\s+)?(?:pizza|food|dinner|lunch|coffee|taxi|uber|ride)\b)", re.I)

# "What modes do you have for minimum interventions?", "how do I make you quieter?": the answer is a list of what exists,
# not something for the language model to make up.
_QUIET_Q = re.compile(_A + r"(?:what|which)\s+(?:\w+\s+){0,2}?(?:modes?|settings?|options?|ways)\b[^.?!]{0,70}?"
                      r"\b(?:interventions?|interrupt\w*|bothering|disturb\w*|quiet(?:er)?|silen\w+|mute|muting)\b[^.?!]{0,40}"
                      r"|how\s+(?:do|can|could)\s+i\s+(?:make|get|have)\s+you\s+(?:to\s+)?(?:be\s+|talk\s+|speak\s+)?"
                      r"(?:quiet(?:er)?|less|fewer|shut\s+up|stop\s+(?:talking|interrupting|bothering)|not\s+(?:interrupt|bother|talk))"
                      r"|how\s+(?:do|can|could)\s+i\s+(?:stop|reduce|limit|minimi[sz]e)\s+(?:you\s+|your\s+)?(?:interruptions?|interventions?|"
                      r"interrupting|talking|chatter)" + _Z, re.I)

QUIET_MODES = ("These make me quieter: \"mute\" or \"go to sleep\" (I stop volunteering until you say \"wake up\"), \"quiet for an "
               "hour\" or \"quiet until 9\", \"focus for 45 minutes\" or \"do not disturb\" (and I tell you in one line what I held "
               "back), \"shh\" to cut off what I'm saying, and \"go private\" to turn the mic, camera and screen off. Reminders you "
               "set still go off, except while I'm muted. I also go quiet by myself when you're gaming.")


def quiet_modes_reply(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or len(t) > 120:
        return None
    return QUIET_MODES if _QUIET_Q.match(t) else None


# "What port are you on?": from the ports it really opened.
_PORT_Q = re.compile(_A + r"(?:what|which)\s+(?:(?:tcp\s+|network\s+|local\s+)?ports?|address|url)\s+(?:are\s+you\s+(?:on|using|running\s+on|"
                     r"listening\s+on)|is\s+(?:the\s+)?(?:console|control|ui|dashboard|server|daemon|ash)\s+(?:on|at|running\s+on|using)|"
                     r"do\s+you\s+(?:use|listen\s+on|run\s+on))" + _Z, re.I)


def ports_reply(rt, text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t or len(t) > 100 or not _PORT_Q.match(t):
        return None
    p = getattr(rt, "ports", None) or {}
    parts = []
    if p.get("console"):
        parts.append(f"the console is on port {p['console']}")
    if p.get("control"):
        parts.append(f"the control socket is on {p['control']}")
    if p.get("tabs"):
        parts.append(f"the browser-tabs listener is on {p['tabs']}")
    if not parts:
        return "I'm not listening on any port right now."
    return "On this computer only: " + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] + "."


_PLACES = re.compile(_A + r"(?:(?:find|show|get|give|recommend|suggest|search\s+for|look\s+for|what(?:'s|\s+is|\s+are)?)\s*(?:me\s+)?"
                    r"(?:the\s+|some\s+|a\s+|any\s+)?(?:good\s+|best\s+|nearest\s+|closest\s+|nearby\s+|cheap\s+)*"
                    r"(?:restaurants?|cafes?|caf\u00e9s?|coffee\s+shops?|pharmac(?:y|ies)|hospitals?|clinics?|shops?|stores?|supermarkets?|"
                    r"gas\s+stations?|petrol\s+stations?|atms?|banks?|hotels?|bars?|gyms?|barbers?|salons?|bakeries|bakery|mosques?|"
                    r"places\s+to\s+eat)\b[^.?!]{0,50}\b(?:near|around|close\s+to|next\s+to|by|nearby)\b"
                    r"|(?:restaurants?|cafes?|pharmac(?:y|ies)|shops?|hotels?|clinics?)\s+(?:near|around|close\s+to|nearby)\b)", re.I)

# Things a model says that ASH has no way to do, and so did not: a mode that does not exist, pictures it cannot fetch,
# a stage direction, places it cannot look up.
_INVENTED = re.compile(
    r"\b(?:[\w-]+\s+){0,3}mode\s+(?:is\s+now\s+)?(?:activated|enabled|engaged|initiated|on|active)\b"
    r"|\bi'?ll\s+(?:now\s+)?(?:mute|silence|disable|turn\s+off)\s+(?:all\s+|your\s+)?(?:the\s+)?(?:notifications?|alerts?)\b"
    r"|\bhere\s+(?:are|is)\s+(?:some|a\s+few|a|the|your)\s+(?:[\w-]+\s+){0,3}(?:pictures?|photos?|images?|videos?|memes?|links?|results?|"
    r"restaurants?|options)\b"
    r"|\(\s*(?:pause|pauses|beat|sighs?|smiles?|laughs?|chuckles?|clears\s+throat|typing|searching)\s*\)"
    r"|\bi\s+(?:found|see|have\s+found)\s+(?:a\s+few|some|\d+|two|three|four|five)\s+(?:[\w-]+\s+){0,2}(?:restaurants?|places|cafes?|"
    r"shops?|options|results|pharmac\w+|hotels?)\b", re.I)
_MODEISH = re.compile(r"\bmode\s+(?:is\s+now\s+)?(?:activated|enabled|engaged|initiated|on|active)\b|\b(?:mute|silence|disable|turn\s+off)\s+"
                      r"(?:all\s+|your\s+)?(?:the\s+)?(?:notifications?|alerts?)\b", re.I)
INVENTED_SAY = ("I can't do that, and I shouldn't have sounded as if I had -- I have no tool for it. "
                "“What can you do?” says what I really can.")


# "I'll make sure to remember that", "I'm remembering that for you": nothing was stored by saying so.
_FAKE_MEMORY = re.compile(r"\bi(?:'ll|\s+will|'m|\s+am)\s+(?:now\s+)?(?:make\s+sure\s+(?:to\s+|i\s+)?)?(?:remember(?:ing)?|memori[sz]ing|"
                          r"keep(?:ing)?\s+(?:that|this)\s+in\s+mind|stor(?:e|ing)\s+(?:that|this|it))\b|\bi'?ve\s+(?:stored|memori[sz]ed|"
                          r"saved\s+(?:that|this|it))\b", re.I)
NOT_STORED = ("I haven't stored that -- I only keep what you tell me as \"remember that ...\". Say it that way and I will.")

# What to say to "that's wrong", "you're making stuff up": the model, asked to explain itself, invented a "version 10.1.2" and a
# "software 8.7.8" to defend an answer it had made up.
_DISPUTE = re.compile(
    _A + r"(?:(?:no|nope|but|well|wait|hold on|actually|bro)[,\s]+)*(?:"
    r"(?:that'?s|that\s+is|that\s+was|this\s+is|it'?s|what\s+you\s+(?:just\s+)?said\s+(?:is|was)|all\s+of\s+(?:this|that|it)\s+(?:is|was)|"
    r"everything\s+you\s+(?:just\s+)?said\s+(?:is|was)|that\s+answer\s+(?:is|was))\s+(?:all\s+|totally\s+|completely\s+|just\s+|so\s+|"
    r"actually\s+)*(?:wrong|incorrect|false|not\s+true|untrue|made\s+up|a\s+lie|nonsense|bs|bullshit)"
    r"|you(?:'re|\s+are)\s+(?:just\s+|totally\s+|completely\s+|all\s+)*(?:wrong|lying|making\s+(?:(?:things|stuff|it|that|all\s+of\s+it)\s+up|up\s+(?:things|stuff|facts|answers)))"
    r"|you\s+(?:just\s+)?made\s+(?:that|it|this|all\s+of\s+(?:that|it)|that\s+all)\s+up|you\s+(?:just\s+)?(?:made|make)\s+(?:stuff|things)\s+up"
    r"|that\s+never\s+happened|i\s+never\s+said\s+that)(?:\s+(?:actually|bro|man|dude|mate))?(?:[.,!\s]+(?:you(?:'re|\s+are)\s+"
    r"(?:just\s+)?making\s+(?:stuff|things)\s+up|you\s+made\s+(?:it|that)\s+up))?" + _Z, re.I)
DISPUTE_FIRST = ("You're right to doubt it -- I can't back that up, so please ignore what I just said. Tell me what's right and I'll "
                 "keep it, or ask me again and I'll say if I don't know.")
DISPUTE_AGAIN = "Okay. I won't defend it. Tell me what the right answer is, or leave it, and I'll leave it."


def disputes_last_answer(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 160:
        return False
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]
    return bool(parts) and all(_DISPUTE.match(x) for x in parts)       # "All of this is wrong. You are making stuff up."


def invented_reply(reply: str) -> Optional[str]:
    """If a model answer pretends to an action or a result ASH cannot produce, what to say instead."""
    r = reply or ""
    if _FAKE_MEMORY.search(r):
        return NOT_STORED
    if not _INVENTED.search(r):
        return None
    return QUIET_MODES if _MODEISH.search(r) else INVENTED_SAY


_LIMITS = (
    (_PLACES, "I can't look up places or their numbers -- I have no map or search tool, so I'd only be guessing names. "
              "Search for it in Maps, and I can open Chrome for you if you say “open Chrome”."),
    (_WEB, "I can't browse the web or show you things from it -- I don't have a browser tool. I can open Chrome for "
           "you if you say “open Chrome”, and I can look up a term in your own files."),
    (_MSG, "I can only read your email, not send it, and I can't send messages or texts to anyone."),
    (_CALL, "I can't make phone calls."),
    (_BUY, "I can't buy things or place orders."),
)


def limit_reply(text: str) -> Optional[str]:
    """What ASH can't do, if `text` asks for it; else None."""
    t = (text or "").strip()
    if not t or len(t) > 140:
        return None
    for rx, say in _LIMITS:
        if rx.match(t):
            return say
    return None


_ASK_FOR_HELP = re.compile(r"^\s*(?:hey\s+)?(?:ash[,\s]+)?(?:can|could|would|will)\s+you\s+(?:please\s+)?help(?:\s+me)?\s*[.!?]*\s*$", re.I)


_TAIL = re.compile(r"(?:[\s,]+(?:then|exactly|really|actually|now|again|anyway|here|though|so|ash|please))+\s*[.!?]*\s*$", re.I)

# What a Chooser (chooser.py) learns "is the person asking what ASH can do?" from. "What can you do then?" and "what can you do exactly?" matched
# nothing, went to the model, and were answered with a made-up "I know you asked about Khalid..." -- one trailing word broke the pattern.
SEEDS = {
    "capabilities": ["what can you do", "what are you able to do", "what do you do exactly", "tell me what you're capable of", "what are your abilities",
                     "what are you good at", "how can you help me", "what can i ask you", "what are your features", "what things can you do for me",
                     "show me what you can do", "what is it that you can do", "list your skills", "what are your limits", "what can't you do",
                     "what do you know how to do", "what is your purpose", "what are you for", "what do you offer", "what are your powers",
                     "what kind of tasks can you do", "what are you capable of", "how can you be useful to me", "what services do you provide",
                     "what sort of things are you able to do", "explain what you can do for me", "what could you do for me", "what are you good for",
                     "give me an overview of your abilities", "what's your skill set", "what kinds of requests do you handle", "what can you take care of",
                     "what types of things are you built for", "which tasks can you take on", "what is your toolkit", "what can you actually do for me"],
    # "What's your name?" went to the model (62 of 112 addressed lines did, and about one in ten needed to): ASH knows who it is.
    "self": ["what's your name", "what is your name", "who are you", "what are you", "tell me about yourself", "introduce yourself",
             "what should i call you", "what do i call you", "who am i talking to", "what is your name again", "so who are you exactly",
             "tell me who you are", "what are you called", "do you have a name", "what exactly are you", "describe yourself",
             "give me a quick intro"],
    "none": ["can you help me", "can you help me with my code", "what can i do about this error", "what do you think about this", "what do you remember",
             "what do you know about me", "how are you", "what time is it", "tell me a joke", "play some music",
             "what is the weather", "what can you see", "what did you do yesterday", "what should i do tonight", "what do you do when i'm away",
             "who built you", "who made you", "are you a robot", "where do you come from", "do you do windows", "can you do it again", "what does the camera do", "how do you work", "how do i use this", "thanks", "yes", "no",
             "open spotify", "set a timer", "who is omar", "what is a transformer", "what is my cpu usage", "how are you feeling"],
}


_SELF = re.compile(_A + r"(?:what(?:'s|\s+is)\s+your\s+name(?:\s+again)?|who\s+are\s+you(?:\s+(?:exactly|again))?|what\s+are\s+you(?:\s+called)?|"
                   r"tell\s+me\s+(?:about\s+yourself|who\s+you\s+are)|introduce\s+yourself|what\s+should\s+i\s+call\s+you|"
                   r"who\s+am\s+i\s+talking\s+to|do\s+you\s+have\s+a\s+name|describe\s+yourself)" + _Z, re.I)


def asks_self(text: str) -> bool:
    """"What's your name?", "who are you?", "tell me about yourself"."""
    t = (text or "").strip()
    return bool(t) and len(t) <= 60 and (bool(_SELF.match(t)) or bool(_SELF.match(_TAIL.sub("", t))))


def self_intro(rt) -> str:
    """Who ASH is, in a sentence or two, with the user's name when it has one."""
    brain = getattr(rt, "brain", None)
    name = str(getattr(brain, "name", "") or "A.S.H")
    user = str(getattr(brain, "user", "") or "")
    who = f" and you're {user}" if user and user.lower() not in ("user", "you") else ""
    return (f"I'm {name} -- Ash for short{who}. I live on this computer: I notice what you're working on, remember what you tell me, "
            f"and help with what you ask. Say \u201cwhat can you do?\u201d for the list.")


_WITH = re.compile(_A + r"(?:what\s+(?:can|could)\s+you\s+do\s+(?:with|about|for|in|on)|can\s+you\s+(?:work|help)\s+(?:with|on|in)|"
                    r"do\s+you\s+(?:work\s+with|support|handle|know))\s+(?:my\s+|the\s+|a\s+|an\s+)?"
                    r"(?P<x>[\w][\w .&+#'-]{1,40}?)\s*[?.!]*\s*$", re.I)
_STOP_SUBJECT = frozenset("me you it that this them there here now then something anything everything".split())


def with_subject(rt, text: str) -> Optional[str]:
    """"What can you do with Google Docs?": from the manual ASH keeps about itself when it mentions the thing, and a plain "nothing on
    my own" when it doesn't -- never the model's guess (it listed features that do not exist)."""
    m = _WITH.match((text or "").strip())
    if not m:
        return None
    what = m.group("x").strip(" .?!")
    words = [w for w in re.findall(r"[a-z0-9]+", what.lower()) if len(w) > 2 and w not in _STOP_SUBJECT]
    if not words or what.lower() in _STOP_SUBJECT:
        return None
    try:
        from .manual import ENTRIES
    except Exception:
        ENTRIES = []
    best, best_hits = None, 0
    for _id, _p, body in ENTRIES:
        low = body.lower()
        hits = sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"s?\b", low))
        hits += 2 * sum(1 for w in words if w in _id)         # the entry named for it beats one that only mentions it
        if hits > best_hits:
            best, best_hits = body, hits
    if best is not None and best_hits >= max(1, (len(words) + 1) // 2):
        body = re.sub(r"^ASH manual,[^.]*\.\s*", "", best).strip()
        cut = body[:300].rsplit(". ", 1)[0] if len(body) > 300 else body
        return cut.rstrip(".") + "."
    return (f"Nothing on my own with {what}: I'm not connected to it. I can open an app or a site if you say \u201copen\u201d and its name, "
            f"and \u201cwhat can you do?\u201d gives the whole list.")


def asks_capabilities(text: str) -> bool:
    t = (text or "").strip()
    if _ASK_FOR_HELP.match(t):
        return False                  # "can you help me" asks for help; the answer is "sure, with what?", not a manual
    if not t or len(t) > 80:
        return False
    return bool(_CAPS.match(t)) or bool(_CAPS.match(_TAIL.sub("", t)))        # "what can you do then?"


def capabilities(rt) -> str:
    """What is really on, here, said in a sentence or two."""
    can: List[str] = []

    def on(name: str) -> Any:
        return getattr(rt, name, None)

    if on("reminders") is not None:
        can.append("set reminders, timers and alarms")
    if on("watcher") is not None:
        can.append("watch a download, the battery or a program and tell you when it's done")
    if on("desktop") is not None:
        can.append("open, switch to and close your apps")
    if on("media") is not None:
        can.append("control the music")
    h = on("smarthome")
    if h is not None and getattr(getattr(h, "cfg", None), "url", ""):
        can.append("turn your lights and plugs on and off")
    w = on("weather")
    if w is not None and getattr(w, "configured", False):
        can.append("tell you the weather")
    c = on("calendar")
    if c is not None and (getattr(getattr(c, "cfg", None), "files", ()) or getattr(getattr(c, "cfg", None), "urls", ())):
        can.append("tell you what's on your calendar")
    i = on("inbox")
    if i is not None and getattr(i, "configured", False):
        can.append("tell you who emailed you")
    if on("tasks") is not None:
        can.append("do several things in a row once you approve a plan")
    if on("play") is not None:
        can.append("play games and tell stories")
    if on("research") is not None:
        can.append("look things up in the background")
    if on("timetable") is not None:
        can.append("tell you when your next class is")
    s = on("samsung")
    if s is not None:
        can.append("read your Samsung notes and write a new one")
    can.append("calculate, convert units and read files")
    controls = ("turn the microphone, camera or screen capture on and off, go private, go quiet, change my face, "
                "and tell you how the computer is doing")
    head = "I can " + ", ".join(can[:-1]) + (" and " if len(can) > 1 else "") + can[-1] + ". "
    return (head + "I can also " + controls + ". And I can talk, with my language model -- which I'd check before "
            "relying on it. I can't browse the web, send email or messages, or make calls.")


# -------------------------------------------------------------------- after the model has answered
_FACTS = re.compile(r"\n+\s*(?:Facts|Briefing|Context|Gist)\s*:\s*[{\[].*\Z", re.S)
_TRAIL_JSON = re.compile(r"\n+\s*\{\s*\"(?:intent|tool_used|action_class|blocked|facts)\".*\Z", re.S)
# A tool name has an underscore ("web_browser", "search_engine"); ordinary
# bracketed words ("[the docs]") are left alone.
_FAKE_CALL = re.compile(r"^\s*\[[a-z][a-z0-9]*_[a-z0-9_]+(?:[ :][^\]\n]{0,120})?\]\s*$", re.I)
_INLINE_CALL = re.compile(r"\[[a-z][a-z0-9]*_[a-z0-9_]+(?:[ :][^\]\n]{0,120})?\]", re.I)
_SPAN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"                      # an e-mail address
                   r"|\+\d[\d\s().-]{6,}\d|\+\d+[\s\d-]*X{3,}[\s\dX-]*"      # +country number, or a placeholder
                   r"|(?<![\w.])\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4}(?![\w])"   # 555-123-4567, 079 123 4567
                   r"|https?://\S+", re.I)                                  # a link
_SENT = re.compile(r"(?<=[.!?])\s+")

# Sentences that give the model away. A small local model says them whatever the prompt says, so they are taken out of the
# answer as well as asked not to be said: who it "really" is, "as an AI", grovelling, asking for "more context",
# and narrating the person it is talking to in the third person.
_MODELS = r"(?:deepseek|qwen|llama|mistral|gemma|phi|chatgpt|gpt|claude|gemini|openai|anthropic|alibaba|meta ai)"
_PERSONA_LEAK = re.compile(
    # who it "really" is: "I'm DeepSeek-R1", "I am Qwen", "made by OpenAI" -- not "I'm opening Claude" or "a fan of Gemini"
    r"\bI(?:'m| am)\s+(?:an?\s+)?" + _MODELS + r"\b|"
    r"\b(?:created|made|built|developed|trained)\s+(?:exclusively\s+)?by\s+(?:the\s+)?(?:company\s+)?" + _MODELS + r"\b|"
    r"\bas an? (?:ai|artificial intelligence|language model)\b|"
    r"\bI(?:'m| am) (?:just )?an? (?:ai|artificial intelligence|language model|chatbot|virtual assistant)\b|"
    # a whole sentence that is only the formula: "I apologize for any confusion." / "Could you please provide more context?"
    r"^\W*I apologi[sz]e for\b[^,;]*\W*$|^\W*sorry for (?:any|the) (?:confusion|inconvenience)\W*$|"
    r"^\W*(?:could|can) you (?:please )?(?:provide|give|share) (?:me )?(?:with )?(?:more|some|a bit more|additional) "
    r"(?:context|details|information)(?: or clarify[^.?!]*)?\W*$|"
    r"^\W*(?:could|can) you (?:please )?clarify(?: what you mean| your intention| what you want)?\W*$|"
    # narrating the person it is talking to: "The user wants...", "It seems like the user is...", "I see that you've typed..."
    r"^\W*(?:the user (?:is|has|was|wants|asked|seems|mentioned|said|typed|just|did|does|would)\b|(?:it )?(?:seems|looks|appears)(?: like)? (?:the|this) user\b|"
    r"I see that (?:the user|you(?:'ve| have)) (?:typed|said|entered|written|mentioned)\b)", re.I)
# An apology that leads into the answer is stripped, and the answer kept: "I apologize for the confusion, but the capital
# is Paris." -> "The capital is Paris."
_APOLOGY_LEAD = re.compile(r"^\W*(?:I apologi[sz]e(?: for [^,;]*)?|sorry for (?:any|the) (?:confusion|inconvenience))[,;]\s*"
                           r"(?:but\s+|and\s+)?", re.I)
_MDLINK = re.compile(r"\[([^\]\n]{1,80})\]\((?:mailto:|tel:|https?://)[^)\s]*\)", re.I)
# The addresses a model writes when it has none: never real, wherever they came from.
_PLACEHOLDER = re.compile(r"@(?:email|mail|example|domain|test|yourdomain|company|website)\.(?:com|org|net)$|"
                          r"^(?:name|user|username|email|someone|john\.?doe|jane\.?doe)@", re.I)


def _squash(s: str) -> str:
    return re.sub(r"[\s().-]+", "", (s or "").lower())


def sanitize(reply: str, known: Any = "") -> str:
    """The reply without what the model should never have said. `known` is
    everything ASH was actually given (the user's words, what memory
    returned), or a function that returns it -- only asked if it's needed."""
    t = reply or ""
    t = _MDLINK.sub(r"\1", t)             # [Name](mailto:x) -> Name: a made-up address in a link is still made up
    t = _FACTS.sub("", t)
    t = _TRAIL_JSON.sub("", t)
    t = "\n".join(line for line in t.split("\n") if not _FAKE_CALL.match(line))
    t = _INLINE_CALL.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return ""
    known_flat: Optional[str] = None
    out = []
    dropped = False
    leaked = False
    for line in t.split("\n"):
        keep = []
        for sent in _SENT.split(line):
            bad = False
            if _PERSONA_LEAK.search(sent):
                leaked = True
                continue                            # not a made-up fact: just not something ASH says
            lead = _APOLOGY_LEAD.sub("", sent, count=1)
            if lead != sent and lead.strip():
                leaked = True
                sent = lead[:1].upper() + lead[1:]  # the answer after the apology stays
            for m in _SPAN.finditer(sent):
                span = m.group(0).strip(" .,;:)")
                if re.search(r"x{3,}", span, re.I):
                    bad = True                      # a placeholder is never a real number
                    continue
                if _PLACEHOLDER.search(span):
                    bad = True                      # a placeholder address is never a real one, even if memory holds it
                    continue
                if known_flat is None:
                    known_flat = _squash(known() if callable(known) else known)
                if _squash(span) not in known_flat:
                    bad = True                      # an address, number or link from nowhere
            if bad:
                dropped = True
            else:
                keep.append(sent)
        if keep:
            out.append(" ".join(keep))
    text = "\n".join(out).strip()
    if dropped and not text:
        return "I don't have that detail."
    if leaked and not text:
        return ""                              # only the model talking about itself: the caller finds something in character
    return text


# -------------------------------------------------------------------- a claim of something not done
_DID = re.compile(
    r"\b(?:i'?ve|i\s+have|i\s+just|i\s+went\s+ahead\s+and|i\s+(?:already)?|done[,!.]?\s+i)\s*"
    r"(?:just\s+|already\s+|now\s+)?"
    r"(?:sent|emailed|e-mailed|texted|messaged|called(?!\s+(?:it|that|this)\b)|phoned|dialed|ordered|booked|bought|purchased|paid|reserved|"
    r"posted|tweeted|uploaded|deleted|set\s+(?:a|an|the)\s+(?:reminder|alarm|timer)|"
    r"added\s+(?:it\s+)?to\s+your\s+(?:calendar|cart|list)|scheduled|forwarded|replied)\b"
    r"|\b(?:message|email|e-mail|text|order|payment|reminder)\s+(?:has\s+been|was|is)\s+(?:sent|placed|set|scheduled|made)\b",
    re.I)
NOT_DONE = ("I haven't actually done that -- I can't send messages, make calls, place orders or post anything "
            "from a chat answer. What I can really do is in “what can you do?”.")


# 2026-11-20, all from the model with no tool behind them: "I did title the note you asked me to create earlier", "I checked the recent
# history and didn't see any new Samsung notes", "I remember the records show Immortal and Khalid are separate entries", "I'll create
# the new Samsung note called 'Lecture 5' and type ...". And the opposite lie: "I can't open Samsung Notes directly", "I can't launch
# apps", "I can't move my face" -- while "open Spotify" and "wink" had just worked by rule.
_DID_NOTE = re.compile(
    r"\bi(?:\s+did|\s+have|'ve|\s+already)\s+(?:already\s+|just\s+)?(?:titled?|renamed?|created?|made|written|wrote|saved|opened|added|moved|filed|typed)\b"
    r"|\bi\s+(?:just\s+|already\s+)?(?:titled|renamed|created|saved)\s+(?:the|that|your|a)\b"
    r"|\bi\s+(?:checked|searched|looked\s+(?:through|at|in|into))\s+(?:the\s+|your\s+|my\s+)?(?:recent\s+|latest\s+)?(?:history|records?|notes?|logs?|list|files?|folders?|database)\b"
    r"|\b(?:the|my|our)\s+records?\s+(?:show|say|says|shows)\b"
    r"|\bi'?ll\s+(?:now\s+)?(?:create|make|write|type|add|title|open)\s+(?:the\s+|a\s+|that\s+)?(?:new\s+)?(?:[\w\-]+\s+){0,2}(?:note|folder|file|document)\b",
    re.I)
NOT_DONE_NOTE = ("I didn't actually do or check that -- there's nothing behind it, and I shouldn't have said so. For a note, say "
                 "\u201cmake a new note in Samsung Notes\u201d and I'll really do it.")
_CANT_OPEN = re.compile(r"\bi\s+(?:can'?t|cannot|can\s+not|am\s+not\s+able\s+to|'m\s+not\s+able\s+to|don'?t\s+have\s+(?:a\s+way|the\s+ability)\s+to)\s+"
                        r"(?:directly\s+|actually\s+)?(?:open|launch|start|run)\s+(?:any\s+|the\s+|your\s+)?(?:apps?|applications?|programs?|samsung|spotify|"
                        r"chrome|opera|discord|edge|it\b|that\b)", re.I)
CAN_OPEN = "I can open apps -- say \u201copen\u201d and the name, like \u201copen Spotify\u201d or \u201copen Samsung Notes\u201d."
_CANT_FACE = re.compile(r"\bi\s+(?:can'?t|cannot|can\s+not|am\s+not\s+able\s+to)\s+(?:really\s+|actually\s+)?(?:move|change|make|do|show)\s+(?:my\s+)?(?:a\s+)?"
                        r"(?:face|faces|expressions?|wink)\b", re.I)
CAN_FACE = "I can make faces -- say \u201cwink\u201d, \u201csmile\u201d or \u201clook surprised\u201d."


# "I've taken a quick look at your code... your syntax is clean" -- with the screen off and no tool run, nothing was looked at.
_SAW = re.compile(
    r"\bi(?:'ve|\s+have)?\s+(?:just\s+|now\s+)?(?:taken\s+a\s+(?:quick\s+|good\s+|closer\s+)?look|had\s+a\s+(?:quick\s+|good\s+)?look|looked|"
    r"reviewed|read|gone\s+(?:through|over)|checked\s+out)\s+(?:over\s+|at\s+|through\s+)?(?:your|the)\s+(?:code|screen|script|project|"
    r"program|source|file|work|document|page)\b"
    r"|\bi\s+can\s+see\s+(?:that\s+)?you(?:'ve|\s+have|\s+are)\b", re.I)
NOT_SEEN = ("I haven't actually looked at that -- I can't see your screen unless you ask me to look. Say “what do you see” "
            "or paste the part you mean.")
# "Let me mute the microphone for an hour" -- said by the model, with nothing muted.
_CLAIM_MUTE = re.compile(r"\b(?:let\s+me|i'?ll|i\s+will|i'?m\s+going\s+to|i'?ve|i\s+have|i\s+am)\s+(?:now\s+|just\s+)?(?:mut(?:e|ed|ing)|silenc(?:e|ed|ing))\b"
                         r"(?:\s+(?:the|your|my)\s+(?:mic(?:rophone)?|sound|volume|notifications?)|\s+myself)?", re.I)
NOT_MUTED = ("I haven't muted anything. Say “mute for an hour” and I'll go quiet, then come back on my own.")


def false_report(reply: str, can_open: bool = True, can_face: bool = True, muted: bool = False) -> Optional[str]:
    """What to say instead when a model answer (no tool ran) reports doing or checking something it did not, or says ASH can't do a
    thing it can. None when there is nothing of the kind."""
    r = reply or ""
    if _DID_NOTE.search(r):
        return NOT_DONE_NOTE
    if can_open and _CANT_OPEN.search(r):
        return CAN_OPEN
    if can_face and _CANT_FACE.search(r):
        return CAN_FACE
    if _SAW.search(r):
        return NOT_SEEN
    if not muted and _CLAIM_MUTE.search(r) and not invented_reply(r):      # a pretend 'mode' has its own, better answer
        return NOT_MUTED
    return None


def claimed_unperformed(reply: str) -> bool:
    """Does a model answer say it did something (sent, called, ordered, posted,
    set a reminder...) that it has no tool to do? Only asked when no tool ran."""
    return bool(_DID.search(reply or ""))
