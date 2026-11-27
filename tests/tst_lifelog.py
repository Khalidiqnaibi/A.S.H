"""tst_lifelog.py -- screen memory, clipboard history (opt-in), the timeline of a day, targets with time counted, repeating reminders, window layout, screenshots, and
"summarize this page". Offline, and every reader/writer is a fake: no screen, clipboard, window or file of this machine is touched.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_lifelog.py"""
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import lifelog as ll  # noqa: E402
from src.daemon.activity import Topic  # noqa: E402

check = tw.check


def main():
    print("\n=== L1. What was on the screen ===")
    now = time.time()
    sm = ll.ScreenMemory(path=os.path.join(tempfile.mkdtemp(), "s.jsonl"), clock=lambda: now)
    sm.add("Traceback (most recent call last): File brain.py line 12 KeyError: 'concept_id' while loading the memory graph", "WindowsTerminal.exe", "PowerShell", now - 3600)
    sm.add("Neural processing units accelerate matrix multiplication for on device inference of language models", "opera.exe", "NPU explained", now - 600)
    check("the same screen twice is kept once", sm.add("Neural processing units accelerate matrix multiplication for on device inference of language models", "opera.exe", "x", now - 590) is False)
    a = sm.ask("what was that error I saw an hour ago", now)
    check("found by its words and when, with where", a.startswith("At ") and "WindowsTerminal.exe" in a and "KeyError" in a, a)
    a = sm.ask("where did I read about NPU", now)
    check("'where did I read about X'", "opera.exe" in a, a)
    check("nothing like it: said so", sm.ask("what was that recipe I saw earlier", now).startswith("I don't see anything about recipe"))
    sm2 = ll.ScreenMemory(path=sm.path, clock=lambda: now + 3600)
    check("kept on disk", len(sm2.items) == 2)
    sm3 = ll.ScreenMemory(path=sm.path, clock=lambda: now + 2 * 86400)
    check("gone after a day", not sm3.items)

    print("\n=== L2. Clipboard history (only when asked) ===")
    box = {"v": ""}
    ch = ll.ClipHistory(lambda: box["v"], lambda t: box.__setitem__("v", t) or True, clock=lambda: now)
    box["v"] = "invoice 4471 for the TPU board, due Friday"
    check("off by default: nothing kept", ch.poll() is False and not ch.items)
    check("asked: on", ch.hear("keep everything I copy").startswith("Okay -- I'll keep the last fifty"))
    ch.poll()
    box["v"] = "wifi password: hunter2 for the home network"
    check("a secret is never kept", ch.poll() is False)
    box["v"] = "https://example.com/paper.pdf"
    ch.poll()
    check("what did I copy about the invoice", ch.hear("what did I copy about the invoice").startswith("At ") and "invoice 4471" in ch.hear("what did I copy about the invoice"))
    check("put the previous one back", "invoice 4471" in ch.hear("put the previous one back") and box["v"].startswith("invoice 4471"))
    check("off forgets", "forgot 2" in ch.hear("stop keeping what I copy") and not ch.items)

    print("\n=== L3. A day, hour by hour ===")
    day = time.strftime("%Y-%m-%d", time.localtime(now - 86400))
    t9 = time.mktime(time.strptime(day + " 09:10", "%Y-%m-%d %H:%M"))
    rows = [{"source": "active_window", "ts": t9, "meta": {"process": "opera.exe", "subject": "Gemini", "project": "Google Gemini"}},
            {"source": "active_window", "ts": t9 + 60, "meta": {"process": "opera.exe", "subject": "Gemini", "project": "Google Gemini"}},
            {"source": "active_window", "ts": t9 + 3700, "meta": {"process": "Code.exe", "subject": "ambient.py", "project": "A.S.H"}},
            {"source": "active_window", "ts": t9 + 3800, "meta": {"process": "bank.exe", "private": True}}]
    check("'what did I do yesterday' is parsed", ll.timeline_day("show me yesterday", now) == day)
    check("a weekday is the latest one before today", ll.timeline_day("what did I do last tuesday", now) is not None)
    s = ll.timeline(rows, day)
    check("hour by hour, private windows left out", "09h opera.exe (Google Gemini); 10h Code.exe (A.S.H)." in s and "bank" not in s, s)

    print("\n=== L4. Targets ===")
    g = ll.Goals(path=os.path.join(tempfile.mkdtemp(), "g.json"), clock=lambda: now)
    spent = lambda text, since: 5400.0                                    # noqa: E731
    r = g.hear("my target is to finish chapter 3 by Friday", spent)
    check("set, with its day", r and r[0].startswith("Got it -- “finish chapter 3”, due ") and r[1] is not None, str(r))
    check("listed with the time counted", g.hear("what targets have I set", spent)[0].startswith("“finish chapter 3”: 1.5 h so far, due in"), str(g.hear("what targets have I set", spent)))
    g.items[0]["due"] = now + 86400
    check("in the morning briefing when due soon", g.briefing_part("morning", now, spent) == "Targets: “finish chapter 3”: 1.5 h so far, due tomorrow.", str(g.briefing_part("morning", now, spent)))
    check("not in the evening", g.briefing_part("evening", now, spent) is None)
    d = g.hear("my target is done", spent)
    check("done, undoable", d[0].startswith("Nice") and g.items[0]["done"] is True)
    d[1]()
    check("...undone", g.items[0]["done"] is False)
    check("a date it can't read is asked about", "I didn't get the date" in g.hear("my target is to write the report by someday", spent)[0])
    check("a browsing limit is not a target", g.hear("keep social under 3 hours a week", spent) is None)

    print("\n=== L5. Every two weeks ===")
    clk = [now]
    rc = ll.Recurring(clock=lambda: clk[0])
    r = rc.hear("remind me to call Omar every two weeks")
    check("set", r and "every 14 days" in r[0], str(r))
    check("not yet", rc.due() == [])
    clk[0] += 15 * 86400
    check("due, and moved on", rc.due() == ["Time to call Omar."] and rc.due() == [])

    print("\n=== L6. Window layout and screenshots ===")
    check("parsed", ll.parse_layout("put Chrome on the left and Word on the right") == [("Chrome", "left"), ("Word", "right")])
    moves = []
    win = ll.Windows(find=lambda p: 100 + len(p) if p in ("chrome", "winword") else None, move=lambda h, x, y, w, hh: moves.append((h, x, y, w, hh)) or True, size=lambda: (1920, 1040))
    out = ll.do_layout(win, [("Chrome", "left"), ("Word", "right"), ("Spotify", "left")], {"word": "winword", "chrome": "chrome"})
    check("each half of the screen, and what was not open", out == "Done: Chrome on the left, Word on the right. I couldn't find Spotify open." and moves[0][1:] == (0, 0, 960, 1040) and moves[1][1:] == (960, 0, 960, 1040), f"{out} {moves}")
    check("a screenshot, plain and named", ll.parse_shot("take a screenshot") == "" and ll.parse_shot("screenshot this and remember it as the TPU diagram") == "the TPU diagram")
    folder = tempfile.mkdtemp()
    saved = []
    fake_img = types.SimpleNamespace(save=lambda p: (saved.append(p), open(p, "wb").write(b"png"))[0])
    msg, path = ll.take_shot("the TPU diagram", folder, lambda: fake_img, now)
    check("saved under its name, never over another", os.path.basename(path) == "the_TPU_diagram.png" and ll.take_shot("the TPU diagram", folder, lambda: fake_img, now)[1].endswith("the_TPU_diagram-2.png"), msg)

    print("\n=== L7. In the runtime ===")
    rt = tm.rig()
    rt._life().screen.add("Traceback most recent call last KeyError concept_id while loading the memory graph in brain", "Terminal", "PowerShell", rt.clock() - 100)
    used, said = rt.say("what was that error I saw earlier")
    check("screen memory, no model", not used and said and "KeyError" in said[-1], f"{used} {said}")
    used, said = rt.say("keep everything I copy")
    check("clipboard history on, by voice", said[-1].startswith("Okay -- I'll keep the last fifty") and rt._life().clip.on)
    used, said = rt.say("my target is to finish the thesis by Friday")
    check("a target by voice", said[-1].startswith("Got it -- “finish the thesis”"), str(said))
    rt.activity.topics["project:thesis"] = Topic(key="project:thesis", name="Thesis", kind="project", by_day={time.strftime("%Y-%m-%d"): 7200.0})
    used, said = rt.say("how are my targets going")
    check("its time is counted from what the activity record holds", "2.0 h so far" in said[-1], str(said))
    rt0 = tm.rig()
    used, said = rt0.say("summarize this page")
    check("nothing read from the screen: says how to fix it, no model", not used and "turn screen reading on" in said[-1], f"{used} {said}")
    rt._life().screen.add("A long article about how neural processing units are built into laptops to run small language models without a network connection at all", "opera.exe", "Article", rt.clock())
    used, said = rt.say("summarize this page")
    check("with screen text: handed to the model, as a request to summarise that text", used and rt.calls[-1].startswith("Summarize this in three short sentences: A long article"), str(rt.calls[-1:]))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
