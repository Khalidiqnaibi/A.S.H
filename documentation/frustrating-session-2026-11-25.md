# The 25 Nov session (12:49-13:06): what went wrong, why, and the fixes

Source: `logs/ashd.log`, `state/journal/2026-11-25.jsonl`, `state/wonder/knowledge.json`, the code as of `467fec6`.
About 45 things were said to ASH in 17 minutes; about 30 answers came back; I count 4 that were clearly right. "Never mind" was said
six times, "Bruh" five, and at 12:59 "You are really frustrating me today Ash."

## What happened, in order

| Said | ASH said | What was wrong |
|---|---|---|
| What model are you using? | "You're not using any specific model here. I'm using a general-purpose model that doesn't track which one I'm using." | It does know (`llm._ollama_model`, the fallback it chose at start). The model was asked and has no facts about itself. |
| What hardware are you using? (heard "hard work") | "That's what I just said." | The repeat guard replaced a real answer. And no rule answers hardware. |
| What can you tell me about this machine? | "CPU at 31%, memory at 71% ... on for 6 min. -- anything else you need on that?" | Usage, not the machine (CPU name, cores, RAM, GPU). The suffix is a stray offer. |
| What's the temperature today? | the same CPU/memory line | Weather is not set up; it answered with system stats. Should say "no weather set up -- tell me your city". |
| Why did you say that? | "...I used the system stats tool. The reply came from thinking it through with the language model." | The second sentence is false: the turn had no model call (`tool 0.3s`). |
| Do you think that's correct? | "You're not correct. You mentioned Stremio ... about Immortal?" | Context pollution (the window, the old name) and an answer to nothing. |
| What's your emotional status / telemetry | "I'm good, thanks. Steady. How about you?" (twice, identical) | A canned line; the real state (drives, tone, fatigue) is never shown. |
| You do have that information | "I don't have direct information ..." then "You're right, I don't have that information." | False, and it flips with the last thing said. |
| Who is Lai? / Who is life? / who is lieth | Wikipedia: a German municipality | Lieth is your friend. The names in your notes are not in the vocabulary hints, so the microphone heard "Lai/life"; then the Wikipedia rule answered. |
| I meant from your memory | "From what you've said before: 'ASH manual, memory. "Remember Lieth is my friend"...'" | **It quoted its own manual as something you said.** |
| (nothing said) | `knowledge.json`: **brother = Omar** at 12:57:52 | **Invented fact.** You then typed "my brother is not omar where did you get that". See Cause 1 below. |
| You are really frustrating me today | "You're frustrated today, aren't you? I don't have any recent interactions with Khalid..." | Said by the model from your words; nothing changed in how ASH behaved. |
| What's recently added to your memory | CPU at 18%, memory at 73% ... | Wrong tool. |
| What's new with you in the new update? | "You're frustrated today ... Khalid's contacts" | A stale answer to an unrelated question. ASH has no changelog. |
| Play the video / Just click Space | "I don't have the capability to play videos or control your device's media" | **False.** `media.py` sends play/pause. The same sentence came back for "Why do you use your emotions" (stale repeat). |
| How do you use your emotions? | "That's what I just said." | Replaced; the question was never answered. |
| What do you know about yourself Ash? | "You haven't told me anything about yourself Ash." | The notes rule read "yourself" as the user. ASH has no self-description. |

## Causes

1. **My `_user_sentences()` reads core memory.** It was added so a role question could be answered from "everything the user said". Core
   memory also holds ASH's own manual (35 `capability` entries), whose examples read "remember my brother is Omar" and "Remember Lieth is my
   friend". The wondering pass (`_from_memory`) matched "my brother is Omar" in the manual and stored it as a fact; the "who is X" rule
   quoted the manual as "what you've said before". Both are in `src/daemon/ambient.py` (`_user_sentences`, `_who_is_options`).
2. **A 3B model improvises everything the rules do not catch.** Every answer above that went wrong is a model answer. It repeats the
   previous answer to a new question, denies abilities it has, agrees with the last thing said, and brings in the window title and the old
   name from the prompt.
3. **ASH has no model of itself.** The manual says what the *features* are, in prose, for retrieval by word overlap. Nothing states
   which are on or off now, which model answers, what hardware it runs on, how long it has been up, what changed in the last update, or
   what it cannot do. So "what can you do" / "what model" / "what do you know about yourself" are answered by guessing.
4. **Two separate emotion systems, neither used.**
   - The brain's `EmotionEngine`/homeostasis (drives: urgency, caution, curiosity, effort, social, fatigue) is written into the prompt as
     "[EMOTIONAL STATE] a little short-tempered ... warmth 0.42 ..." and the face reads it; a 3B model ignores it. Its only behavioural effect
     is `executive` gating (how much thinking to buy).
   - `feelings.py` (cheer, thanks, praise) answers "how are you" with canned lines and moves the face after five minutes. It does not
     read the drives. So ASH's own mood is "short-tempered" in the prompt and "Steady" in the answer.
   - The user's mood: `moments.note_mood` (a 30-minute reading used only to hold back jokes and pick the mood question) and the per-concept
     `valence` added on 18 Nov -- whose reader, `Brain.topic_valence()`, **nothing calls**. Episodes carry no mood. Your frustration is
     never stored with the episode, never changes the tone, the length, the fillers, or what ASH offers.
5. **The explanations are templates.** "Why did you say that" appends "thinking it through with the language model" to every slow-path
   turn, tool-only ones included.
6. **Slow, so it was cancelled.** 14-22 s per answer (the prompt is about 2,000 tokens), each preceded by a filler ("On it", "Let me see")
   that says nothing. Six "Never mind"s cancelled turns that then also wrote and un-wrote memory.
7. **The microphone has no idea who your people are.** "Lieth", "Omar" and the company name are not in the recogniser's hints unless
   they were typed as terms; whisper `base.en` hears "Lai", "life", "hard work", "victor", "not normal".

## The most blatant violations of what ASH is meant to be

- **"Must not invent facts"**: brother = Omar; the manual quoted as the user's words; "I don't have access to recent interactions or
  contacts" (it has both).
- **"Does not claim abilities it lacks / denies abilities it has"**: "I don't have the capability to play videos or control your
  device's media."
- **"Tools decide, the model only words it"**: the question about the model, the hardware, the emotions, the update and itself all went
  to the model.
- **"Emotion shapes tone"**: it does not. Nothing about the answer changed when you said you were frustrated.
- **"Never says 'the user'"** style rules: "You seem to be confused", "What specific information are you looking for about Immortal?"
- **Honesty about its own record**: the "why" answer is wrong in the very turn where it matters.

## Fixes, in the order I would do them

### 1. Stop the bleeding (small, each with a test)
1. `_user_sentences()` reads episodes only, and core entries only when they are the user's own (`source` chat/console), never `manual_*`
   or `capability`. Then repair `knowledge.json` (remove brother = Omar, which you said is wrong) and re-run `memory_cleanup` for
   anything else that came from the manual.
2. A **self-model** (`src/daemon/selfmodel.py`) built from the live runtime, never from prose:
   model(s) loaded and why, CPU/GPU/RAM, uptime, memory use, which sensors and features are on/off right now, what it cannot do (from the
   manual's "cannot" entry), the last N commits (`git log` or `documentation/`) for "what's new", the date of the last nap/sleep, how
   much of memory is episodes/entities/concepts. Answered by rule: "what model are you using", "what hardware", "what can you do",
   "what can't you do", "what's new / what changed", "what do you know about yourself", "how much do you remember", "when did you last
   learn something". Also **injected as a short "[ABOUT YOU]" block in the prompt** so the model cannot deny what is on.
3. A **false-denial check** (`honest.py`): "I don't have the capability to ..." / "I can't ..." for something the self-model says is
   on (media, notes, timetable, reminders, camera on request) is replaced by the true answer.
4. "Why did you say that" reads the turn's real record (rule / tool / model; `last_narration.mode`; confidence), never a template.
5. Stray suffixes ("-- anything else you need on that?") off tool answers; "temperature" with no weather set up says so and asks the city.
6. The repeat guard never replaces an answer to a *different* question, and never an answer the user has not heard (the TTS was still
   speaking the earlier one).

### 2. Make the emotions real (one model, used)
7. **One emotional state**: `feelings.py` reads the homeostasis drives (they are the source of truth); "how are you / emotional status /
   telemetry" are answered from the drives with the real numbers and one sentence ("Frustrated with myself: three answers dropped in a row,
   fatigue 0.6, curiosity 0.2"), not a canned line.
8. **The user's mood is remembered**: each episode stores `mood` (sentiment reading, -1..1) and `arousal` (repeated rephrasing, "bruh",
   cancellations); `Brain.topic_valence()` and per-person mood ("you get tense when Linda's exams come up") are read by the briefing, the
   free-moment choice and the prompt.
9. **A repair mode** when you are frustrated (signals ASH already sees: "bruh", "never mind" x2, "you do have that", "that's wrong",
   "frustrating", the same question rephrased): one short apology, no filler cues, no trailing offers, answers by rule or one sentence, and
   a plain "I can't do that, here is what I can" instead of improvising. It ends when you say something neutral or positive.
10. The face and the voice follow the same state (slower, flatter, shorter when low; livelier when you laugh).

### 3. Speed, so nobody cancels
11. Prompt diet 3 (static prefix first, drives last, shorter persona, `num_ctx` 2048) and the small-talk model; fillers only when
    the answer is more than 8 s away, and never two in a row.
12. A turn that is cancelled before it speaks writes nothing (already true); also write nothing for a turn answered by "I can't".

### 4. Hearing
13. People's names from notes, wondering and the timetable (Lieth, Omar, Co.Te.De.) feed the recogniser's hints; a second pass with
    `small.en` for low-confidence phrases; "Did you say X?" before acting on or answering a doubtful name or command.
14. The Wikipedia rule never answers a "who is X" when X is close (by sound) to a name ASH knows.

## Tests to add
A fixture of this session as a replay: each line above, with the expected rule/answer; `tst_selfmodel`, `tst_repairmode`,
`tst_moodmemory`, `tst_manualnotfact` (the manual is never a source of facts).
