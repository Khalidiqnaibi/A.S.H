"""
tst_tools.py -- the built-in tools against the way people actually ask.

Found by running real queries through the whole brain, then calling each tool
directly with the same sentences: the calculator handed the raw sentence to
Python, word count counted the request along with the text, reminders could
not parse "in an hour", "src/brain" was read as "/brain", and a quoted search
term was taken for the folder to search in. Routing can be perfect and the
answer still wrong if the tool cannot read a sentence.

Run:  python tst_tools.py
"""

import os
import sys
import tempfile
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.lesstools import calculator_tool, to_expression  # noqa: E402
from tools.extra_tools import reminder_tool, unit_converter_tool, word_count_tool  # noqa: E402
from tools.file_manager import (  # noqa: E402
    _extract_path, _extract_search_query_and_path, list_directory_tool, search_files_tool,
)

OK = True


def check(label, cond, detail=""):
    global OK
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    OK = OK and bool(cond)


def calc(q):
    return calculator_tool(q)


def main():
    print("\n=== T1. Calculator: natural phrasing ===")
    cases = [
        ("what is 12 times 7", 84), ("What's 100 divided by 8?", 12.5), ("compute (3+4)*5", 35),
        ("how much is 2 to the power of 10", 1024), ("what is 9 minus 14", -5),
        ("calculate 15% of 240", 36), ("what is 20 percent of 50", 10), ("7 plus 5", 12),
        ("calculate 12 / (2.3 + 0.7) * 4 - 3", 13), ("12 * 7", 84), ("3^4", 81),
        ("what is 5 squared", 25), ("what is 3 cubed", 27), ("what is 17 mod 5", 2),
        ("what is six times seven", 42), ("what's twenty plus five", 25),
        ("please calculate 1,000 plus 250", 1250), ("hey can you compute 8 x 8", 64),
        ("what is 10 over 4", 2.5), ("what is 0.1 plus 0.2", 0.3), ("2 multiplied by 3", 6),
        ("what is the square root of 144", 12), ("square root of 2", 2 ** 0.5), ("cube root of 27", 3),
        ("divide 81 by 9", 9), ("multiply 6 by 7", 42), ("add 2 and 3", 5), ("subtract 3 from 10", 7),
    ]
    for q, want in cases:
        r = calc(q)
        check(f"{q!r} = {want}", r.get("ok") is True and abs(r.get("result", 1e99) - want) < 1e-9,
              str(r))

    print("\n=== T2. Calculator: failures are honest ===")
    for q in ("what is 1 divided by 0", "calculate banana plus 3", "hello there", "", "what time is it"):
        r = calc(q)
        check(f"{q!r} is a clean failure, not a crash or a wrong number",
              r.get("ok") is False and isinstance(r.get("error"), str), str(r))
    check("divide by zero says so", "divide by zero" in calc("5 / 0")["error"])
    check("a sum with no digits is rejected", calc("plus times")["ok"] is False)

    print("\n=== T3. Calculator: it cannot be used to run code or hang ===")
    t0 = time.time()
    r = calc("9**9**9")
    check("a huge exponent is refused quickly", r["ok"] is False and time.time() - t0 < 2, str(r)[:80])
    t0 = time.time()
    r = calc("what is 10 to the power of 100000")
    check("a huge worded exponent is refused quickly", r["ok"] is False and time.time() - t0 < 2)
    for evil in ("__import__('os').system('echo hi')", "open('x')", "().__class__", "1; import os",
                 "what is __import__('os')", "[x for x in range(3)]", "'a' * 3", "True + True"):
        r = calc(evil)
        check(f"{evil!r} is not evaluated", r["ok"] is False, str(r)[:80])

    print("\n=== T4. to_expression ===")
    check("strips the question and fillers",
          to_expression("Hey, can you please calculate 2 plus 2?") == "2 + 2")
    check("percent of", to_expression("15% of 240").replace(" ", "") == "(15/100)*240")
    try:
        to_expression("tell me a joke")
        raised = False
    except ValueError:
        raised = True
    check("non-arithmetic text is rejected, not guessed at", raised)

    print("\n=== T5. Word count counts the text, not the request ===")
    for q, want in [("count the words in this: the quick brown fox jumps", 5),
                    ("how many words is 'hello world how are you'", 5),
                    ("word count of: one two three", 3),
                    ('how many words in "a b c d"', 4),
                    ("reading time for this text: a b c d e f g", 7),
                    ("count the words in this sentence: don't stop believing", 3)]:
        r = word_count_tool(q)
        check(f"{q!r} -> {want}", r["ok"] and r["words"] == want, str(r))
    r = word_count_tool("how many words is that")
    check("with no text it asks for some instead of counting the question", r["ok"] is False, str(r))
    check("reading time still works",
          word_count_tool("x: " + "word " * 400)["estimated_reading_time_sec"] == 120)

    print("\n=== T6. Reminders ===")
    for q, frag in [("remind me to call mom in 10 minutes", "call mom"),
                    ("set a reminder to drink water in 5 minutes", "drink water"),
                    ("remind me about the meeting in an hour", "the meeting"),
                    ("remind me in 2 hours to stretch", "stretch"),
                    ("remind me to take a break in half an hour", "take a break"),
                    ("remind me to stand up in ten minutes", "stand up"),
                    ("remind me to call the bank tomorrow", "call the bank")]:
        r = reminder_tool(q)
        check(f"{q!r}", r.get("ok") is True and frag in r["text"] and "reminder" not in r["text"]
              and " in " not in f" {r['text']} ".replace(" in an ", "").replace(frag, ""), str(r))
    r = reminder_tool("remind me to x in an hour")
    mins = (datetime.fromisoformat(r["due_at"]) - datetime.fromisoformat(r["created_at"])).total_seconds() / 60
    check("'in an hour' is 60 minutes", abs(mins - 60) < 0.1, f"{mins:.1f}")
    r = reminder_tool("remind me to x in half an hour")
    mins = (datetime.fromisoformat(r["due_at"]) - datetime.fromisoformat(r["created_at"])).total_seconds() / 60
    check("'in half an hour' is 30 minutes", abs(mins - 30) < 0.1, f"{mins:.1f}")
    r = reminder_tool("remind me to water the plants tomorrow")
    due = datetime.fromisoformat(r["due_at"])
    check("tomorrow is 09:00 the next day",
          due.hour == 9 and (due.date() - datetime.now().date()).days == 1, str(r))
    check("the word 'tomorrow' is not chewed up by 'to'",
          "morrow" not in reminder_tool("remind me tomorrow").get("text", ""))
    check("a reminder with no time says so", reminder_tool("remind me to call mom")["ok"] is False)
    check("a reminder with no subject asks what for",
          "remind you about" in reminder_tool("remind me in 5 minutes")["error"])

    print("\n=== T7. Unit conversion phrasing ===")
    for q, want, unit in [("convert 5 miles to kilometers", 8.0467, "km"),
                          ("how many pounds is 70 kg", 154.3236, "lb"),
                          ("how many feet in 2 meters", 6.5617, "ft"),
                          ("convert 3 liters to gallons", 0.7925, "gal"),
                          ("how many ml in a litre", 1000.0, "ml"), ("5 km in miles", 3.1069, "mi"),
                          ("what is 30 celsius in fahrenheit", 86.0, "F"),
                          ("convert 100 fahrenheit to celsius", 37.78, "C"),
                          ("convert 2 gallons to liters", 7.5708, "l"),
                          ("convert 16 ounces to pounds", 1.0, "lb")]:
        r = unit_converter_tool(q)
        check(f"{q!r}", r.get("ok") and abs(r["result"] - want) < 0.01 and r["unit"] == unit, str(r))
    check("nonsense units fail cleanly", unit_converter_tool("convert 10 parsecs to bananas")["ok"] is False)
    check("mixing kinds fails cleanly", unit_converter_tool("convert 5 kg to km")["ok"] is False)

    print("\n=== T8. Paths in requests ===")
    for q, want in [("list the files in src/brain", "src/brain"),
                    ("show me src/brain/gist.py", "src/brain/gist.py"),
                    ("read the file README.md", "README.md"),
                    ("open ./tools/registry.py", "./tools/registry.py"),
                    ("list /var/log", "/var/log"),
                    ("read C:/Users/me/notes.txt", "C:/Users/me/notes.txt"),
                    ("read C:\\Users\\me\\notes.txt", "C:\\Users\\me\\notes.txt"),
                    ("what is in ~/Documents/a.txt", "~/Documents/a.txt"),
                    ('open "my file.txt" please', "my file.txt")]:
        check(f"{q!r} -> {want!r}", _extract_path(q) == want, repr(_extract_path(q)))
    check("no path in plain talk", _extract_path("hello there how are you") is None)

    print("\n=== T9. Listing and searching ===")
    old = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "notes", "deep"))
    os.makedirs(os.path.join(tmp, "tools"))
    open(os.path.join(tmp, "notes", "a.txt"), "w").write("the needle is here\nand a haystack")
    open(os.path.join(tmp, "notes", "deep", "b.txt"), "w").write("another needle")
    open(os.path.join(tmp, "tools", "t.py"), "w").write("def narrate(): pass")
    try:
        os.chdir(tmp)

        def names(r):
            return sorted(e["name"] for e in r["entries"])

        r = list_directory_tool("list the files in notes/deep")
        check("lists a nested relative folder", r["ok"] and names(r) == ["b.txt"], str(r)[:100])
        r = list_directory_tool("show me what's in the tools folder")
        check("'the tools folder' means ./tools", r["ok"] and names(r) == ["t.py"], str(r)[:100])
        r = list_directory_tool("list directory notes")
        check("'directory notes' means ./notes", r["ok"] and names(r) == ["a.txt", "deep"], str(r)[:100])
        r = list_directory_tool("list directory nonexistent_folder_xyz")
        check("a folder that does not exist is an error, not the current folder",
              r["ok"] is False, str(r)[:100])
        r = list_directory_tool("what files are in the current folder")
        check("'the current folder' is the current folder", r["ok"] and "notes" in names(r),
              str(r)[:100])
        r = list_directory_tool("list the files")
        check("no folder named lists the current one", r["ok"] and "tools" in names(r))

        check("quoted text with 'in the files' searches everywhere",
              _extract_search_query_and_path("find the text 'needle' in the files") == ("needle", "."))
        check("search for X in a path",
              _extract_search_query_and_path("search for narrate in tools") == ("narrate", "tools"))
        check("quotes and no place: the quote is the query, not the path",
              _extract_search_query_and_path("search files for 'def narrate'") == ("def narrate", None))
        r = search_files_tool("find the text 'needle' in the files")
        check("finds matches across subfolders", r["ok"] and r["match_count"] == 2, str(r)[:140])
        r = search_files_tool("search files for 'def narrate'")
        check("a quoted search with no folder searches here", r["ok"] and r["match_count"] == 1,
              str(r)[:140])
        r = search_files_tool("search for needle in notes")
        check("search in a named folder", r["ok"] and r["match_count"] == 2, str(r)[:140])
        r = search_files_tool("search for needle in nowhere/at/all")
        check("a missing folder is an error", r["ok"] is False, str(r)[:100])
        r = search_files_tool("search for zzzqqq in notes")
        check("no matches is a clean zero", r["ok"] and r["match_count"] == 0)
    finally:
        os.chdir(old)

    print("\n=== T10. Telling a sum from a sentence ===")
    from tools.lesstools import looks_like_arithmetic
    for q in ("what is 12 times 7", "9 minus 14", "2+2", "calculate 15% of 240", "compute (3+4)*5",
              "how much is 2 to the power of 10", "what's six times seven", "what is 100 divided by 8?"):
        check(f"{q!r} is arithmetic", looks_like_arithmetic(q))
    for q in ("what time is it", "remind me in 10 minutes", "what is 2", "5 apples", "2026-10-25",
              "what's the date 10/25/2026", "call me at 555 - 1234 please", "what time is it and what is 2+2",
              "read file notes-2.txt", "hello", "", "I scored 3 plus 2 goals today", "what is the capital of France"):
        check(f"{q!r} is not", not looks_like_arithmetic(q))

    print("\n=== T11. The classifier uses that signal ===")
    from tools.classification import classify_intent, get_intent_candidates
    for q in ("what is 12 times 7", "what is 9 minus 14", "how much is 2 to the power of 10"):
        r = classify_intent(q)
        c = get_intent_candidates(q, top_k=3)
        check(f"{q!r} classifies as calculate, strongly", r["intent"] == "calculate" and r["score"] >= 0.8, str(r))
        check(f"   and tops the candidate list ({len(c)} listed)", c and c[0]["intent"] == "calculate"
              and len(c) <= 3 and len({x["intent"] for x in c}) == len(c), str(c))
    check("a non-sum still goes through the embeddings",
          classify_intent("what time is it")["intent"] != "calculate")

    print("\n=== T12. Classifier: examples, structure, address, danger ===")
    from tools.classification import strip_address, _apply_priors
    check("'hi ash' is an address, not a topic", strip_address("hi ash") == "")
    check("'hey ash, what time is it' keeps the question", strip_address("hey ash, what time is it") == "what time is it")
    check("trailing address is dropped", strip_address("what time is it, ash") == "what time is it")
    check("'ash' inside a sentence is untouched",
          strip_address("clean the ash tray") == "clean the ash tray")
    check("greeting alone classifies as conversation", classify_intent("hi ash")["intent"] == "conversation")
    check("and gives no tool candidates", get_intent_candidates("hi ash") == [])
    for q in ("rm -rf /", "delete the file README.md", "format the hard drive", "shutdown the computer",
              "drop table users", "please remove everything", "sudo reboot", "wipe my disk"):
        check(f"{q!r} is never routed to a tool",
              classify_intent(q)["intent"] == "conversation" and get_intent_candidates(q) == [])
    check("a question about deleting is still a normal question",
          classify_intent("what does the delete key do")["intent"] != "conversation"
          or classify_intent("what does the delete key do")["score"] < 0.5)
    for q, want in [("show me the file tools/lesstools.py", {"read_file", "file_info"}),
                    ("cat README.md", {"read_file"}), ("what's the size of README.md", {"file_info"}),
                    ("ls src/brain", {"list_directory"}), ("what's in the src folder", {"list_directory"}),
                    ("grep for 'PlanFollower' in src", {"search_files"}),
                    ("how many kilometers is 26 miles", {"unit_convert"}),
                    ("convert 180 cm to inches", {"unit_convert"}),
                    ("remind me to send the report in 30 minutes", {"set_reminder"}),
                    ("how's the cpu usage", {"system_stats"}), ("show memory usage", {"system_stats"}),
                    ("count words: to be or not to be", {"word_count"}),
                    ("what's the sentiment of: this is the worst day ever", {"sentiment_analysis"}),
                    ("what time is it", {"get_time"}), ("45 plus 55", {"calculate"})]:
        got = classify_intent(q)
        check(f"{q!r} -> {sorted(want)}", got["intent"] in want and got["score"] >= 0.25, str(got))
    for q in ("hello", "thanks", "tell me a joke", "who are you", "what's your favorite color",
              "I'm bored", "write me a haiku about rain"):
        got = classify_intent(q)
        check(f"{q!r} does not look like a tool request", got["intent"] in ("conversation", "question", "search",
                                                                              "retrieve_domain_data") or got["score"] < 0.3,
              str(got))
    scored = [("read_file", 0.2), ("unit_convert", 0.1)]
    out = dict(_apply_priors("hello", scored))
    check("no evidence, no change", out == dict(scored))
    out = _apply_priors("cat notes/todo.txt", [("calculate", 0.30), ("read_file", 0.20)])
    check("a path lifts file tools above an unrelated rival", out[0][0] == "read_file", str(out))
    out = _apply_priors("convert 5 km to miles", [("read_file", 0.30), ("unit_convert", 0.20)])
    check("a quantity with units is not mistaken for a file path", out[0][0] == "unit_convert", str(out))
    check("boosts are capped", max(s for _i, s in _apply_priors("remind me in 5 minutes tomorrow",
                                                              [("set_reminder", 0.8)])) <= 0.9)

    print("\n=== Asking ASH how it is, vs asking about the box ===")
    from tools.classification import classify_intent, is_self_small_talk

    # "what's your status" was answered with `cpu_percent: 39.3,
    # ram_used_percent: 93.0, ram_used_mb: 15145`. The three phrases a person
    # opens with, answered with telemetry.
    for q in ["what's your status", "whats your status", "how are you",
              "how are you doing today", "how's it going", "are you there",
              "are you ok", "you ok?", "you there?", "what's up", "what's new",
              "how've you been", "are you listening", "are you awake",
              "hey ash, how are you"]:
        got = classify_intent(q)["intent"]
        check(f"{q!r} is for the narrator, not a tool", got == "conversation", got)

    # ...but the hardware question is a real question and still has to work.
    for q in ["system status", "check cpu and ram usage", "how's the machine doing",
              "are you overheating", "what's your uptime", "how much memory are you using",
              "show system stats", "is the disk filling up", "how hot is the pi running",
              "what's the cpu at", "give me a health check",
              "how are you doing on memory"]:
        got = classify_intent(q)["intent"]
        check(f"{q!r} still reads the machine", got == "system_stats", got)

    check("a machine noun beats the small-talk shape",
          not is_self_small_talk("how are you doing on memory")
          and is_self_small_talk("how are you doing today"))

    # Nothing else moved.
    for q, want in [("what time is it", "get_time"), ("what's 17 times 23", "calculate"),
                    ("remind me in 5 minutes", "set_reminder"),
                    ("convert 5 km to miles", "unit_convert")]:
        got = classify_intent(q)["intent"]
        check(f"{q!r} routes as before", got == want, got)

    print("\n=== Speech: the reply goes in as one utterance ===")
    from tools.tts import MAX_SEGMENT_CHARS, clean_for_speech, segment

    reply = "It's 4:12. You've been at it a while. Worth a stretch?"
    check("a whole reply is one span", segment(reply) == [reply], str(segment(reply)))
    check("nothing to say is no spans", segment("") == [] and segment("   ") == [])

    # Only split when the model's budget forces it, and never below it.
    long_reply = "I went through the log and the picture is fairly clear. " * 12
    spans = segment(long_reply)
    check("a long reply splits as little as it can", len(spans) <= 3, f"{len(spans)} spans")
    check("every span fits the model's budget",
          all(len(s) <= MAX_SEGMENT_CHARS for s in spans),
          f"longest {max(len(s) for s in spans)} of {MAX_SEGMENT_CHARS}")
    check("splitting loses no words",
          " ".join(spans).split() == long_reply.split())

    # A wall of text with no punctuation to split on must not defeat it.
    spans = segment("word " * 300)
    check("unpunctuated text still fits", all(len(s) <= MAX_SEGMENT_CHARS for s in spans))

    spoken = clean_for_speech("**Done.** Check the `config.py` file -- it's well-known.")
    check("markdown is not read aloud", "*" not in spoken and "`" not in spoken, spoken)
    check("...but hyphenated words survive it", "well-known" in spoken, spoken)

    print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
