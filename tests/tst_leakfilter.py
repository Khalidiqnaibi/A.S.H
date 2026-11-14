"""
tst_leakfilter.py -- what the filter takes out of a model's answer, and what it must leave.

honest.sanitize drops the sentences that give a small local model away ("I'm DeepSeek-R1", "as an AI", "I apologize for any
confusion", "could you provide more context", "the user wants..."). A first version matched too loosely and cut real answers:
"I apologize for the confusion, but the capital is Paris." lost the capital; "I'm opening Claude for you" and "a fan of Gemini"
read as the model naming itself; "please provide the file name" and "could you clarify which Omar?" -- ordinary questions --
were deleted.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_leakfilter.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import honest  # noqa: E402

check = tw.check


def t1_taken_out():
    print("\n=== Taken out ===")
    for leak in ("I'm DeepSeek-R1, an AI assistant created exclusively by DeepSeek.", "I am Qwen, a large language model.",
                 "I'm a language model built by OpenAI.", "As an AI, I don't have feelings.", "I'm just an AI.",
                 "I apologize for any confusion.", "Sorry for the confusion.",
                 "Could you please provide more context?", "Can you provide more details or clarify your intention?",
                 "Could you please clarify what you mean?", "Could you clarify?",
                 "It seems like the user is just focusing on a conversation.", "The user wants to know the time.",
                 "I see that you've typed \"hit me\" in the console.", "I see that the user mentioned a song."):
        got = honest.sanitize(leak + " The sea is salty.")
        check(f"gone: {leak[:52]}", got == "The sea is salty.", got)
    check("a reply that was only that is empty", honest.sanitize("I'm DeepSeek-R1.") == "")


def t2_kept():
    print("\n=== Kept ===")
    for ok in ("I'm opening Claude for you.", "I'm not sure about Gemini, honestly.", "He is a big fan of Claude.",
               "She works as an assistant to the mayor.", "Please provide the file name.", "Could you clarify which Omar?",
               "Can you give me more details about the meeting time?", "The user interface is fine.",
               "My phi is a Greek letter.", "I'm Meta about it, ha.", "I am the llama farmer's friend.",
               "A user on the forum said it works.", "Sorry, the kettle is on."):
        got = honest.sanitize(ok)
        check(f"stays: {ok}", got == ok, got)


def t3_apology_then_answer():
    print("\n=== An apology that leads into the answer ===")
    check("the answer after it stays",
          honest.sanitize("I apologize for the confusion, but the capital is Paris.") == "The capital is Paris.",
          honest.sanitize("I apologize for the confusion, but the capital is Paris."))
    check("...with 'and' too", honest.sanitize("I apologize for that, and it is seven.") == "It is seven.")
    check("...and a second sentence is untouched",
          honest.sanitize("I apologize for the confusion, but it's Tuesday. Rain later.") == "It's Tuesday. Rain later.")
    check("sorry for the inconvenience, then the answer", honest.sanitize("Sorry for the inconvenience, it is closed today.")
          == "It is closed today.")
    check("an ordinary 'Sorry,' is the voice's own and stays", honest.sanitize("Sorry, it is closed today.")
          == "Sorry, it is closed today.")


if __name__ == "__main__":
    t1_taken_out()
    t2_kept()
    t3_apology_then_answer()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
