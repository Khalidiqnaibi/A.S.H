# Fix and feature plan for review -- from the 25 Nov 14:00-15:12 interaction

**Status (after review): implemented, each with its own test and commit, except F8 (the repeat guard misfire was not reproducible in the log), F11 (the slow
answers were the reasoning model and CPU load, already fixed; no hard cap added) and F12 (the name migration needs a decision on which name to keep).
N1 was widened to the whole computer: keys, mouse, scroll, typing, click-on-text from the screen, connected devices.**

Nothing here is implemented yet. Each item: what went wrong (seen in the journal / ashd.log), how to implement, the test, and size
(S = under an hour, M = a few hours, L = a day). Every item would be its own commit with its own test, pushed alone.

## 1. Fixes (things that went wrong in that interaction)

| # | Seen | Implement | Test | Size |
|---|---|---|---|---|
| F1 | "what was the last update / what was added last" not understood | `selfmodel.parse`: add patterns "last update", "latest change", "what's new", "what changed"; answer from the git subjects already read | `tst_selfmodel` cases | S |
| F2 | "tell me about yourself / about Ash" hijacked by the facts rule | In `_facts_request`, skip when the subject is ASH/you/yourself; route to selfmodel kind `self` | rule-order test | S |
| F3 | "what about my emotional state" went to the model | Extend `feelings.wants_user_mood` with "what about my ...", "my emotional state/status" | `tst_moodanswer` | S |
| F4 | Emotion numbers print 10.00 / 9.72 as if 0-1; "short-tempered" next to "happy, trust" | `feelings.from_engine`: print on the real 0-10 scale ("9.7 of 10"), pick the label from the top two emotions only, drop labels that contradict the top one | `tst_onestate` | S |
| F5 | "stop the video" says nothing is playing while a video is on screen | Media control: if `now_playing` is empty, fall back to the window title (YouTube/Netflix/VLC) and send the media key anyway | `tst_media*` | M |
| F6 | Mic loop: repeated words with commas ("so, so, so, ...") reach the rules | Collapse stutter runs in the transcript clean-up; ignore a line that is one word repeated 4+ times | `tst_hearing` | S |
| F7 | False "away" while a video plays (no input for minutes) | Treat audible output / a playing media window as presence in the presence detector | presence test | S |
| F8 | Repeat guard fired on legitimate re-asks ("what time is it" twice) | Only suppress an identical *answer* within 20 s when the question was also identical and the first was not a failure | `tst_nevermind` | S |
| F9 | "why did you say the one before it" broken after a "Did you mean...?" | Keep a turn history (see N2) and let the explain rule read it instead of the last model call only | `tst_whynot` | M |
| F10 | Repair mode: the 3B model ignores the NOTE (still "let's try a different approach"); apology and dispute answer back to back | (a) while repair is on, answer by rule or by the model with a 1-sentence cap and strip questions-back/offers from the reply; (b) speak the apology only, and do not answer the same line a second time | `tst_repair` | M |
| F11 | 12-67 s answers and cancel storms | Cancel at most once per turn, cap model wait at 25 s then say "I don't know that one"; drop queued turns older than 30 s | `tst_cuesilence` | M |
| F12 | Name mixed: Immortal vs Khalid in notes and replies | Migration that rewrites old headings with a backup, and reads the name from settings only | `tst_username` | S |
| F13 | "look up X" queries kept filler ("can you look up for me ...") | Clean the query: strip the command words, keep the subject; "images of X" opens image search | `tst_lookup` | S |

## 2. Features

| # | What | Implement | Size |
|---|---|---|---|
| N1 | **Voice keys** -- "press space / enter / escape", "full screen", "scroll down", "new tab", "go back", "volume up", "click the first one" | New `keys.py` with a safe allow-list sent as key presses to the window in front (pywinauto/SendInput already available); nothing outside the list; confirm for anything not on it | M |
| N2 | **Turn history** -- "what did you say before that", "say that again", "what was the question before" | Ring buffer of the last 10 turns `{heard, said, via: rule/tool/model}` in the runtime; one rule reads it | M |
| N3 | **Provenance on facts** -- "where did you get that" | `{source sentence, time, via: said/noted/guessed}` on knowledge entries; the wondering pass may only store `via: said`; guessed facts are voiced as guesses | L |
| N4 | **Correction by rule** -- "my brother is not Omar", "that's wrong" | Parse the negation, remove or replace the fact, say what changed, log to `state/failures.jsonl` | M |
| N5 | **"How do you work?"** from the docs (emotions, memory, sleep, concepts, repair mode) | Short true paragraphs in a table, picked by keyword; no model | S |
| N6 | **Installed apps and games inventory** -- "what games do I have", "open X" | Scan Start-menu shortcuts + Steam library manifest once a day, kept in a small JSON | M |
| N7 | **"Restart" / "restart yourself"** | Spawns the same launcher, exits cleanly after speaking | S |
| N8 | **Backchannel ignored** -- "yeah", "ugh", "hmm" right after an answer is not a request | Skip routing when ASH spoke less than 8 s ago and the line is a filler (repair still counts it) | S |
| N9 | **Voice follows state / face follows the person** | Speak speed from emotion energy; concerned face when repair starts (see ideas-2026-11-25 B5-B6) | M |
| N10 | **Daily failure digest** | Group `state/failures.jsonl` + `state/unrouted.jsonl` at the nightly sleep into `documentation/digest-DATE.md` | M |

## 3. Suggested order (each its own commit)

1. Quick honesty and routing: F1, F2, F3, F4, F13, N5
2. Things that hurt that afternoon: F6, F7, F8, N8, F12
3. Repair that works with a small model: F10, F11
4. Memory of the conversation: N2, then F9
5. Facts you can trust: N4, then N3
6. Control by voice: F5, N1, N7
7. Then N6, N9, N10

## 4. Rules I will keep

- No daemon copies or full test runs without asking; only the one test file for each change, run at below-normal priority.
- Runtime state files stay out of every commit; no attribution lines; rebase before each push.

## 5. Questions for you

- N1: is a fixed safe list of keys enough, or should it also be able to click by what is on screen?
- F10: if a short answer is impossible for the small model, is "I don't know that one" acceptable as the fallback?
- N3 is the biggest change (data format). Do it, or do N4 alone first?
