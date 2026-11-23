# Fixes and ideas -- 23 Nov 2026

From the last 25 commits (21-23 Nov) and the 21-22 Nov interactions (`logs/ashd.log`, `state/journal/`). Ranked; each item names what it
connects to.

## What the history shows

- **No "while I was away" answer.** 22 Nov 16:45: "Welcome back. You were away about 1 hour 8 minutes." Then "whats new while i was away" got
  "away about an hour and a half" (made up). Nothing joins activity memory, the journal, fired reminders and what the nap learned.
- **"What am I working on" goes to the model.** It answered "Khalid, who you mentioned is Immortal": it never read the window/activity
  history ASH already keeps. The Khalid/Immortal mix shows in two more answers.
- **Shutdown confirmation too short.** "shut down" -> "Shut ASH down completely? Say yes". "ye" came 42 s later and went to the model;
  "ye'" (stray apostrophe) is not read as a yes.
- **Crash watch cannot take an explanation.** 21 Nov: "8 unexpected shutdowns since I last told you". The reply "that's just me
  force-shutting down my PC" went to the model (80 s, then no answer). Those were force-stops during the CPU/memory flood.
- **"That's what I just said."** still replaces answers to different questions (22 Nov, twice).
- **"What free moments do you have lined up?"** answered "Nothing right now" with a question queued.

## Ranked ideas

1. **Resource governor.** `src/daemon/gamemode.py` already drops priority, pauses screen and microphone, and unloads the models when a
   game runs. Fire the same machinery when system memory is over ~85% or CPU is saturated, not only for games. Unload Ollama after ~30
   minutes idle (now `keep_alive` is 6 h). Add a status line: "ASH 1.8 GB, model X loaded".
2. **Settings menu** (`src/ui/settings.py`; "Your name" is there now). Add: a dropdown of installed models (answer model and light
   model), a CPU-threads slider (`ASH_OLLAMA_NUM_THREAD`), an "unload model when idle" minutes box.
3. **"While you were away" digest.** Answer "what's new", "how long was I away", "what did I do" from real data (activity memory,
   journal, reminders that fired, what the nap learned); say it on "Welcome back".
4. **Name cleanup.** When the name is set, rewrite old "Immortal:" headings in episodes and notes, and keep one name in the prompt.
5. **Crash watch learns causes.** Accept "that was me / the power went / I forced it", store the cause, stop counting those; notice
   shutdowns that follow high memory.
6. **Confirmations:** longer window, strip stray punctuation from the reply, a reminder after ~20 s.
7. **Repeat guard** (`src/daemon/repeat.py`): when the model repeats itself to a *different* question, ask once more with "answer
   differently" instead of "That's what I just said."
8. **Light model routing.** Install a ~1.5B model for small talk; `src/daemon/modelpick.py` already routes it (`llm.light_model`).
9. **"What free moments do you have" lists what is queued:** open questions, today's briefing, what it is waiting for.
10. **Connect what exists:**
    - Samsung Notes + timetable: "summarize my last NoSQL note"; offer tomorrow's lecture note at class time.
    - Weekly review + activity: a Friday "this week you spent ..." line.
    - Reminders + quiet: hold reminders that fire while muted and say them in order afterwards.
    - Calendar: still needs an iCal feed link (`calendar.urls`).

## Done in the same stretch (21 Nov)

- A model that only thinks out loud (qwen3:4b) is replaced at warm-up by the best installed one that does not, and unloaded.
- Ollama CPU threads capped to a quarter of the logical cores; TensorFlow/Keras no longer imported (250 MB, 14 s of a core per start).
- Measured live (microphone and OCR off): ASH 1.8 GB peak (2.07 GB before); a question cost ~9 s at ~100% of one core (was 90 s at 300%+).
