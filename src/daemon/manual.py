"""
src/daemon/manual.py

ASH's own manual, kept in its core memory.

Core memory holds who ASH is and the rules it keeps (core_memory.json). This
adds what it can and cannot do and the words that make it do them, one entry
per area, category "capability". Retrieval is by meaning, so when someone asks
"can you remind me when I get home?" the reminders entry comes back, and the
model answers from it instead of guessing. Entries are plain statements of
fact written to match what the code really does, including the limits.

`seed(core)` writes them under fixed ids: new ones are added, changed ones
updated, ones the user edited by hand under a different id are left alone.
It runs at start-up, so the manual never falls behind the code. What is
actually switched on *here* is answered live by "what can you do?" (honest.py).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("ash.daemon.manual")

# (id, priority, text)
ENTRIES: List[Tuple[str, int, str]] = [
    ("overview", 6,
     "ASH manual, overview. ASH can: set reminders, timers and alarms; remind you when you get home; watch things and "
     "tell you when they happen; open, switch to and close apps; control music; do several steps after you approve a "
     "plan; keep scenes, quiet time and focus mode; answer short who/what facts from Wikipedia; read your calendar, "
     "weather and mail when set up; remember people and what you tell it; play games and tell stories; read or copy "
     "your clipboard when asked; show a face and clock. ASH cannot browse the web, send email or messages, make calls, "
     "buy things or see your location. Ask \"what can you do?\" for what is switched on right now."),
    ("reminders", 5,
     "ASH manual, reminders/timers/alarms. Say: \"remind me in 20 minutes to check the oven\", \"remind me tomorrow at 9 "
     "to send the report\", \"set a timer for 5 minutes\", \"wake me up at 7\". Right after one goes off, \"snooze\" or "
     "\"snooze 10 minutes\" brings it back. \"What are my reminders?\" lists them; \"cancel the timer\" drops them. They "
     "survive a restart and are said even in quiet time, but not while muted or paused. They are one-time, not repeating."),
    ("home", 5,
     "ASH manual, \"remind me when I get home\". Say \"this is home\" once while on your home Wi-Fi, then \"remind me "
     "when I get home to water the plants\". ASH says it when the laptop rejoins that network after being away a few "
     "minutes. \"What's waiting for me at home?\", \"cancel my home reminders\". It knows home only as a network name: "
     "it has no GPS or location, and only works while ASH is running on the laptop."),
    ("watch", 5,
     "ASH manual, watching. Say \"tell me when the download finishes\", \"let me know when Chrome closes\", \"tell me when "
     "the battery is charged\" (or \"at 80%\"), \"tell me when the CPU calms down\", \"tell me when report.docx changes\". "
     "It says it once, drops it after a day, and \"stop watching\" cancels. It can't watch a website or a build it has no "
     "view of, and says so. It also warns once a day if the disk will fill within two days."),
    ("apps", 5,
     "ASH manual, apps. Say \"open Outlook\", \"switch to Chrome\", \"close Word\" (it asks first, and closing is a polite "
     "request so the app can still ask to save). Only apps it knows by name, plus any added under desktop.apps in "
     "daemon.json. \"Run my morning check\" runs a routine you kept. It cannot click inside apps or type for you."),
    ("music", 5,
     "ASH manual, music and photos. \"Pause the music\", \"play\", \"next song\", \"previous track\", \"turn it up\", \"mute "
     "the sound\": sent as media keys, so it works with Spotify or a browser video without knowing which. \"Take a "
     "photo\" saves one webcam frame to Pictures/ASH, only when asked and never while privacy is on. It can't pick a "
     "specific song or playlist."),
    ("plans", 5,
     "ASH manual, several steps at once. \"Open Outlook, set a timer for 25 minutes and turn off the desk lamp\": ASH "
     "shows the plan and asks \"Go ahead?\"; yes runs each step and says what came of each. \"When I say get me ready for "
     "the meeting, open the agenda and set a 25 minute timer\" keeps it as a named task. Only steps it can already do."),
    ("home_devices", 4,
     "ASH manual, lights and plugs. \"Turn on the living room lights\" works through Home Assistant, and only if a Home "
     "Assistant address is set up in daemon.json; otherwise ASH can't. A guest's voice can't do this."),
    ("quiet", 6,
     "ASH manual, quiet and focus. \"Quiet for an hour\", \"quiet until 9\", \"go to sleep\" (wake up ends it): ASH stops "
     "volunteering but still answers you. \"Focus for 45 minutes\", \"do not disturb until 3\", \"I'm done focusing\": "
     "same, and it tells you in one line what it held back. Reminders and watches still go off. It silences itself "
     "only, not other apps' notifications."),
    ("scenes", 5,
     "ASH manual, scenes. \"Work mode\" (quiet 90 minutes, curious face), \"wind down\" (quiet 2 hours, sleepy), \"break "
     "time\" (quiet 15 minutes). A scene can also open listed apps and say a line, and the reply names only what really "
     "happened. \"End work mode\" or \"I'm back\" ends it; \"what scenes do you have?\" lists them. Add your own under "
     "scenes in daemon.json."),
    ("pretend", 4,
     "ASH manual, pretend play. \"Be a pirate for ten minutes\", \"talk like a robot\", \"pretend you're a detective\" "
     "(pirate, robot, wizard, detective, butler, cowboy, grandma, kid, BMO, or any \"a ...\" character): the model words "
     "its answers in that voice, still true. Timers and facts stay plain. Ends on time (default 10 minutes, at most an "
     "hour) or at \"be yourself\"."),
    ("greetings", 4,
     "ASH manual, greetings. \"Good morning\" gets a greeting by the real clock plus the morning briefing the first time "
     "that day; \"good night\" gets a recap of the day. \"Brief me\" or \"what's my day look like\" gives the briefing "
     "any time; \"how was my week?\" gives the weekly review."),
    ("facts", 5,
     "ASH manual, facts. \"Who was Ada Lovelace?\", \"what is a black hole?\", \"what does ephemeral mean?\" are answered with "
     "the first sentences of the Wikipedia article, quoted, with \"(Wikipedia)\" after. Only the topic is sent to "
     "en.wikipedia.org. It cannot search the open web or give news; \"look into X\" researches in the background "
     "from Wikipedia, your own files and what you've told it, and reports later."),
    ("memory", 5,
     "ASH manual, memory. \"Remember Lieth is my friend\", \"remember my brother is Omar\": parsed and stored as people and "
     "relations, and the reply says exactly what was stored. \"Who am I?\" and questions about people it knows answer "
     "from that. \"Undo that\" reverses the last thing it learned or set up (within ten minutes); \"forget what I said "
     "about Omar\" removes that subject. It mentions events you say are coming (\"I have an interview Thursday\") and "
     "asks how they went."),
    ("face", 5,
     "ASH manual, the face. It reacts by itself: thanks gets love, a laugh gets a big smile, bad news gets sad, "
     "\"good night\" gets sleepy, and it looks around and blinks when idle. Ask for one: \"smile\", \"look surprised\", "
     "\"wink\"; \"what faces can you make?\" lists them. Clicking the face in the console pats it. The console tints with "
     "its mood and turns into a clock after two minutes untouched."),
    ("privacy", 6,
     "ASH manual, privacy and senses. \"Privacy mode\" turns the microphone, camera and screen capture off at once; "
     "\"privacy off\" in the console restores them. \"Turn off the mic\", \"turn on the camera\" (camera and screen only "
     "come on when said to ASH). Nothing is sent anywhere except what a feature says it sends. \"Learn my voice\" lets "
     "it tell you from a guest. A guest still gets answers, but is refused anything that acts or is private: opening or "
     "closing apps, scenes, plans, clipboard, home reminders, turning the mic/camera/privacy or ASH itself off, and "
     "changing or forgetting your voice."),
    ("clipboard", 4,
     "ASH manual, clipboard. Only when switched on in daemon.json and only when asked: \"what's on my clipboard\", "
     "\"summarize my clipboard\", \"copy that\" (copies what ASH last said), \"clear my clipboard\". It never watches the "
     "clipboard and never reads out or passes on anything that looks like a password or key."),
    ("info", 4,
     "ASH manual, weather, calendar and mail. Weather, calendar (an .ics file or link) and mail need setup in "
     "daemon.json; without it ASH can't tell you. Mail is read-only: who wrote and when, never sending or replying. "
     "\"What's on my calendar today?\" and the briefing use it; the console's heads-up cards show what starts within "
     "the hour."),
    ("games", 4,
     "ASH manual, games and stories. \"Let's play a game\": guess the number, a quiz about your own people, word "
     "association, twenty questions, riddles and trivia; \"stop\" ends it. It can tell stories. It sometimes offers a "
     "game when it's been quiet and you seem well."),
    ("care", 4,
     "ASH manual, looking after you. After long stretches at the computer it may suggest a stretch, water or a break "
     "(at most once an hour, only at a free moment). \"No more nudges\" stops it until restart. It celebrates "
     "birthdays and long streaks, and \"tell me something good\" recalls good moments."),
    ("routines", 4,
     "ASH manual, routines. ASH notices when you open the same apps in the same order on several days and asks once "
     "if it should remember that as a routine; yes saves it, and \"run my <name>\" opens them in order."),
    ("explain", 5,
     "ASH manual, explaining itself. Every reply in the console has a small tag saying how it was made (model, tool, "
     "rule, Wikipedia...); tapping it shows why. Things ASH sets up or learns appear with an Undo button."),
    ("limits", 7,
     "ASH manual, what ASH cannot do. It cannot browse the web or open links, send email, texts or chat messages, make "
     "calls, buy or order anything, know your location, control other people's devices, or see the screen unless "
     "screen capture is on. It must never say it did one of these: only say something was done when a tool really did "
     "it. If asked, say plainly that it can't, and offer the nearest thing it can do."),
]


def _rid(slug: str) -> str:
    return f"manual_{slug}"


def seed(core: Any) -> Dict[str, int]:
    """Write the manual into core memory (a CoreMemoryEngine). Idempotent."""
    stats = {"added": 0, "updated": 0, "same": 0}
    store = getattr(core, "store", None)
    if store is None:
        return stats
    from src.memory.core.core_model import CoreRule
    now = time.time()
    changed = False
    for slug, priority, text in ENTRIES:
        rid = _rid(slug)
        cur = store.rules.get(rid)
        if cur is None:
            store.rules[rid] = CoreRule(rule_id=rid, category="capability", text=text, priority=priority,
                                        hard=False, created_at=now, updated_at=now)
            stats["added"] += 1
            changed = True
        elif cur.text != text or cur.priority != priority:
            cur.text, cur.priority, cur.category, cur.updated_at = text, priority, "capability", now
            stats["updated"] += 1
            changed = True
        else:
            stats["same"] += 1
    if changed:
        try:
            store._save()
            core._rebuild_index()
        except Exception:
            logger.exception("Could not save the manual into core memory")
    return stats
