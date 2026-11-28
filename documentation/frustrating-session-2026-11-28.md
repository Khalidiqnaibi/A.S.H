# The 28 Nov morning (09:35-10:06) -- why it was frustrating, and what to fix

**Status: fixes 1-12 are implemented, each with its own test and commit. Fix 11 caps the prompt blocks and logs the prompt size; the real before/after seconds still need one measured run on the live model. The 'still thinking' cue and the concept-graph editing in the feature list are not done.**

Evidence: `state/journal/2026-11-28.jsonl` (about 75 lines you said), `logs/ashd.log` 14300-14870, `state/failures.jsonl`, `state/unrouted.jsonl`
(23 lines reached the model, 10 took 15 s or more, the slowest 36 s). Model: qwen2.5:3b. CPU stayed at 18-20%, so it was not a flood.

## What happened, in one paragraph

You were testing the new concept and learning features. Nearly every question was a feature question ("how do you use your concepts", "show me your
concepts tree", "teach you from YouTube", "learn from Google ...", "remember that when I say see how something works ...") and **none of them reached
the feature**: they went to the 3B model, which took 20-36 s and said something wrong or "I don't know". You said "Nevermind" about a dozen times to cut the
wait. Each "Nevermind" cancelled the model call, ASH counted it as frustration, apologised, and went into repair mode -- which then cut every later answer
down to "I don't know." You ended up asking it why it was "saying something and writing something else" and it answered about the camera.

## The most blatant violations (things we said we were building, broken in front of you)

1. **Slow answers were punished as if they were your anger.** "Nevermind" said while the model was 20 s into an answer is "too slow", not "frustrated".
   Repair mode (my 25 Nov work) fired on it at 09:38, apologised over the top of "Okay -- dropped it", and from then on turned
   "What do you know about your code?" into "I don't know what you mean" and "Remember that ..." into "I don't know." A repair mode that makes ASH
   less useful is the opposite of its purpose.
2. **The new features did not hear you.** "From Google, learn how to open Discord" / "From Google, learn how to open a YouTube video" (said three times) went
   to the model: the learn-from-Google rule only takes "Ash, learn from Google how to ..." word order. "Show me your concepts tree" and
   "How do you use your concepts / concept memory" (asked 4 times) have no route, although concepts exist and have a page.
3. **A wrong rule answered "how to teach you new concepts"**: "I haven't been taught anyone or anything yet. Say 'remember this as my blue mug'" -- the
   face-teaching rule took the word "teach". Twice.
4. **"Remember that when I tell you to see how something works, I'm probably talking about watching a YouTube video"** was answered "I don't know." (30 s),
   three times. A user-taught rule about how to read you is the single most valuable thing it could store, and there is no way to store it
   (the remember rules only know people and "X is a Y").
5. **Window switching says one thing and does another.** "Switching to Discord." (said before doing it), then "I couldn't bring Discord/Opera to the front"
   while Opera was open and in front a minute earlier. Windows refuses SetForegroundWindow from a background process unless a key was just pressed; the
   verify added on 26 Nov covers *open*, not *switch*, and "Switching to ..." is claimed first.
6. **"Why did you say that?" explained the wrong turn.** It explained the *mis-routed* line ("I took it as conversation") instead of saying "I don't have a
   rule for that, so the model guessed", and "That's wrong" got a stock apology rather than fixing anything. "The one before that" worked once, then not.
7. **Spoken and written differ and it did not know.** You asked "why are you saying something and writing something else?" (the spoken form is the short
   version of what the console shows). It answered "I can't see you right now -- the camera is off". Nothing explained the shortening.
8. **"Who is Immortal?" / "What do you remember?" read out a blob** of unrelated notes ("a YouTube video is a type of video ...; Samsung Note is your best
   and favorite note-taking app ...; your dad's clinic is near ..."). Recall is dumping everything that loosely matches, unranked and unfiltered, and
   "Immortal" still is not you (the merge exists, it was never run).
9. **The model dies on "Nevermind", then "My brain's not answering right now"** is generated for the cancelled turn (three times in the log, dropped
   before it was spoken, but the turn is wasted and the memory write is unwritten each time).
10. **Hearing**: "Of Indiscone" (for "Discord"), "Obra", "That's strong" (for "That's wrong"), "What was that, Afif?" (for "Who is Afif"). Names in the notes are
    hints now, but app names and your own command words are not.

## Why: three root causes

- **A. Routing has holes exactly where the newest features are.** Each feature was built with its own phrasing; nothing checks that "how do you use X",
  "show me X", "teach you X", "from Google, learn X" (natural orders) reach it. The meaning chooser (6 commits ago) helps but needs examples per feature.
- **B. The model path is too slow to be a fallback.** 20-36 s with idle CPU means long prompts, not a busy machine (about-you block + working memory +
  retrieved memories + manual + repair note on a 3B model). Every miss costs half a minute, which produces the "Nevermind".
- **C. The new safety nets have no sense of cause.** Repair mode, cancel, and "I don't know" all read the *symptom* (a cancel, "wrong") and not the *cause*
  (slow turn, mis-route). They need the turn record that already exists (`_last_turn`, took_s, pathway).

## How to fix (each its own commit with its own test; nothing started yet)

| # | Fix | How | Size |
|---|---|---|---|
| 1 | Cancel is not anger | In `Repair.hear`, pass `took_s` of the turn being cancelled; a "never mind" during a turn older than 8 s counts 0.2 and says "That was slow -- sorry." once, no repair mode. Repair also never replaces an answer with "I don't know" unless the person disputed it ("that's wrong") in the last turn | S |
| 2 | Natural word order for learn-from-Google | Accept "from Google, learn ...", "learn from Google ...", "google how to ...", "look up how to ... and learn it" in `learnweb.parse`; unaddressed-but-warm lines count (same 'probably for ASH' rule) | S |
| 3 | Concept questions route to the concepts | Rules for "how do you use / learn / teach you concepts", "what can your concept memory do", "show me your concepts (tree/graph)" -> the real description (self-model `how:concepts` exists) and opens the Mind page's concept view | S |
| 4 | "Teach" is not face-teaching | Face-teach rule only on "remember this as X" / "remember <name>'s face" while the camera is on; "how to teach you ..." goes to concepts/learning | S |
| 5 | **Taught rules**: "remember that when I say X I mean Y" | New `habits.py`: store (trigger, meaning) pairs, shown in `[ABOUT YOU]` for the model and applied by rule when the trigger is a whole phrase ("see how something works" -> read the page in front / look up). "What have I taught you?", "forget that rule" | M |
| 6 | Window switch that works and tells the truth | Use the AttachThreadInput / Alt-key trick before SetForegroundWindow, check `GetForegroundWindow` after 0.5 s, and say "Switching to X" only when it worked; otherwise "X is open but Windows wouldn't bring it forward -- try alt-tab" | S |
| 7 | "Why did you say that" about the right thing | Explain the *answer*: rule / tool / model, and for a model answer with no rule, say "I have no rule for that, so I guessed with the language model" and offer "want me to learn a rule for it?" | S |
| 8 | Spoken vs shown | `speech.spoken_form` records that it shortened; "why are you saying something else" answers "I read you the short version; the full one is on the screen" and "say the whole thing" reads it | S |
| 9 | Recall that answers the question | For "what do you remember / who is X" rank by relevance to the question, cap at 3 items, say "and N more" and offer "want the rest?"; never list definitions the person asked about nothing for | M |
| 10 | Run the name merge | Offer once at start: "Immortal and Khalid look like the same person -- merge?" (the command exists in `selfcare.py`); answers to "Who is Immortal" then come from you | S |
| 11 | A faster fallback | Log prompt tokens per model turn; trim the prompt (drop the manual unless asked, cap working memory, one retrieval block), `num_ctx` 2048, keep the model warm; target under 8 s for a plain question. Measure before and after | M |
| 12 | Hearing your own commands | Add app names (Discord, Opera, Stremio), your command words ("switch to", "learn from") and "wrong/strong" confusions to the recogniser hints and to a small correction list | S |

## Features that would make the experience better, built on what exists

- **A "didn't understand" loop that teaches itself.** When a line reaches the model with no rule, say once "I don't have a rule for that" and keep it in the digest;
  after you say the same thing twice, ask "want me to learn that as <nearest feature>?" (meaning chooser + `unrouted.py` already have the pieces).
- **Say what it is doing while a slow answer runs.** A visible/spoken "still thinking (12 s)" and "say 'stop' to cancel" so you cancel on purpose, not by "Nevermind".
- **A session replay for the morning.** The digest (`state/digest/`) already groups failures; add this day's turns with how each was answered, so the next
  bad morning is a file you hand over (the 25 Nov plan's D13/D14: a thumbs-down in the console that tags "wrong" and keeps the turn).
- **Concept memory you can see and steer.** "Show me your concepts tree" opens a view; "that link is wrong" removes a link; "teach you from this video" reads the
  page title/captions in front (the page reader exists) and proposes concepts to keep.
- **Voice keys for the learning workflow.** The new control layer plus learnweb: "from Google, learn how to open Discord" then "now do it" runs the learned steps
  with the keyboard/mouse (it already can click on text), with a yes first.

## Order I would do it in

1, 2, 3, 4 (they are the ones that broke the morning and are small), then 6, 7, 8, then 5 (taught rules) and 11 (speed), then 9, 10, 12.
