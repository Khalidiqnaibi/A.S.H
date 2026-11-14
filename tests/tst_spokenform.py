"""
tst_spokenform.py -- what is fine on a screen and wrong out loud.

ASH's canned lines carry where a setting lives ("turn on sensors.browser_tabs in daemon.json", "(email.allow_send in
daemon.json)"), a file's path, a song's lines joined by " / " and a wink emoji. On the console that helps whoever is fixing it;
spoken, "sensors dot browser underscore tabs in daemon dot json" is not JARVIS. speakable() -- the one place every spoken line
goes through -- turns them into something a person would say, and leaves the console's text alone.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_spokenform.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.senses.actuators import speakable  # noqa: E402

check = tw.check


def t1_settings_and_paths():
    print("\n=== Settings and paths ===")
    got = speakable("It's off by default; turn on sensors.browser_tabs in daemon.json and load the extension.")
    check("a setting is named in words, the file is 'the settings file'",
          got == "It's off by default; turn on browser tabs in the settings file and load the extension.", got)
    got = speakable("Sending is switched off (email.allow_send in daemon.json), so I've kept it as a draft.")
    check("a parenthetical about where a setting lives is dropped", got == "Sending is switched off, so I've kept it as a draft.", got)
    check("...also the short form", speakable("Texting is switched off (phone.allow_sms).") == "Texting is switched off.")
    check("...and game_mode's", speakable("Game mode is off (game_mode.enabled in daemon.json).") == "Game mode is off.")
    got = speakable("Done -- 40 rows saved to C:\\Users\\pc\\Documents\\x\\browsing-2026.json. It's plain JSON.")
    check("a Windows path is 'your files', and the sentence still ends", got == "Done -- 40 rows saved to your files. It's plain JSON.", got)
    check("a unix path too", speakable("Saved to /home/omar/notes/a.txt.") == "Saved to your files.")
    got = speakable('Run "python ashctl.py vision install" first.')
    check("a command is 'the ... command'", got == 'Run "the vision install command" first.', got)


def t2_symbols():
    print("\n=== Symbols ===")
    got = speakable("\u266a Roses are red / violets are blue / ASH is a console \u266a")
    check("a song's lines are joined with commas, the notes are gone", got == "Roses are red, violets are blue, ASH is a console", got)
    check("arrows are 'then'", speakable("Here: a \u2192 b \u2192 c.") == "Here: a, then b, then c.")
    check("an emoji is not read aloud", speakable("Wink \U0001F609 there") == "Wink there")
    check("a heart, a check mark", speakable("Done \u2705 \u2764\ufe0f") == "Done")


def t3_untouched():
    print("\n=== What must stay ===")
    for t in ("It's 5 pm.", "Use Python 3.11 at github.com and google.com.", "and/or e.g. this", "Two plus two is four.",
              "The file is called notes.txt and it is on your desktop.", "Open config.json in your editor.",
              "He said 3/4 of it is done.", "settings.json is fine"):
        check(f"unchanged: {t!r}", speakable(t) == t, speakable(t))
    check("a number with a dot is not a setting", speakable("About 3.5 hours.") == "About 3.5 hours.")
    check("nothing in, nothing out", speakable("") == "" and speakable(None) == "")


def t4_console_keeps_it():
    print("\n=== The console keeps the text ===")
    from src.daemon.drafts import Drafts
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "daemon", "drafts.py"),
               encoding="utf-8").read()
    check("the string that names the setting is unchanged in the code (only speech is cleaned)",
          "email.allow_send in daemon.json" in src)


if __name__ == "__main__":
    t1_settings_and_paths()
    t2_symbols()
    t3_untouched()
    t4_console_keeps_it()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
