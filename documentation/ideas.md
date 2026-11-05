# Ideas: what ASH could add next

Candidate features, drawn from what ASH already does (wondering, free moments,
activity memory, the predictive brain, screen and mic senses, the face and
voice) and from two reference characters: JARVIS (the capable household
assistant) and BMO (the playful companion).

These are directions, not commitments. Each entry says what it is, why it is
worth it, and what in ASH it would build on.

---

## Suggested order

1. Follow-ups on remembered events
2. Look it up before asking
3. Contradictions in memory
4. Morning and evening briefings
5. Alerts that watch for you
6. Games at a free moment

The first three make wondering much smarter with no new hardware or accounts.
Briefings and alerts are the most useful JARVIS-style behaviour and build on
activity memory and system stats that already exist. Games are the cheapest
way to make ASH feel like a companion, and slot straight into the free-moment
registry.

---

## 1. Wondering: questions from what ASH already knows

Today wondering asks about blanks: an unknown person, an unresolved role, an
unknown term, a missing fact. The bigger wins come from questions about what
ASH *does* know.

### Follow-ups on remembered events
**Status: built** (`src/daemon/followups.py`).
"You had an interview on Thursday. How did it go?"
Notice future events the user mentions ("my exam is Monday", "meeting Sara
tomorrow") and ask about them after they have happened. Nothing currently
looks forward in time, and this is the behaviour that most makes an assistant
feel like it was listening.
*Builds on:* episodic memory, the wondering queue, free moments.

### Contradictions
**Status: built** (wondering, kind `conflict`).
"You said Omar works at X, but today it sounded like Y. Which is right?"
When two facts in memory disagree, ask instead of silently keeping the latest.
*Builds on:* entity memory, knowledge store, wondering.

### Stale facts
**Status: built** (wondering, kind `stale`).
Facts about jobs, homes, projects and relationships go out of date. After a
few months without being mentioned, ask a light "still true?" instead of
trusting them forever.
*Builds on:* entity memory timestamps, wondering expiry and cool-downs.

### Confirm its own guesses
**Status: built** for people, from the company a name keeps (work, classes, family).
"You and Sara come up together a lot. Is she a coworker?"
A yes/no question about an inference is easier to answer than an open one,
and gets answered more often.
*Builds on:* activity memory's "connect" step, co-occurrence in the concept
network.

### Look it up before asking
**Status: built** (`src/daemon/lookup.py`): memory, files, Wikipedia.
Before asking what a term means, check memory, the user's files and the web.
Only ask if that fails, and then ask to confirm a guess ("I think X means Y,
right?") rather than with a blank question.
*Builds on:* `search_files`, `read_file`, memory router; needs a web search
tool.

### Learn from misroutes
**Status: built** (`src/brain/corrections.py`).
When ASH picked the wrong tool or was corrected, ask what was meant, and feed
the answer to the classifier and the concept network.
*Builds on:* the routing evaluator, three-factor reward in plasticity.

### "How do you usually do this?"
**Status: built** as routines: app openings noticed over days, kept by name, run with "run my <routine>" (`src/daemon/routines.py`, `desktop.py`). Multi-step procedures beyond opening apps are not.
For a task the user keeps doing by hand, ask for the steps once and keep them
as a plan ASH can run next time.
*Builds on:* activity memory, the planner and plan follower.

### Ask the most useful questions first
**Status: built**: ranked by how often a subject recurs across days and how recent it is (not by the predictive model).
Rank open questions by how many future decisions depend on the answer, not
only by urgency.
*Builds on:* the predictive model in `src/brain/`.

### "What are you wondering about?"
**Status: built**.
Let the user ask for the open questions and answer several at once, instead
of waiting for a free moment.
*Builds on:* `ashctl wonders`, the console.

---

## 2. From JARVIS: the capable assistant

### Morning and evening briefings
**Status: built** (`src/daemon/briefing.py`). Weather is not in it yet.
One spoken summary: today's calendar and reminders, what the user was working
on yesterday, the weather, and any open questions.
*Builds on:* activity memory, reminders, the weather tool, wondering.

### Alerts that watch for you
**Status: built** (`src/daemon/watch.py`). Builds cannot be watched directly.
A build failing, the disk filling up, the battery low, a long download
finishing. ASH can read system stats but nothing watches them over time.
*Builds on:* `system_stats`, attention rules for when to interrupt.

### Desktop control
**Status: built** for opening, switching to and closing apps (`src/daemon/desktop.py`). Typing into windows is deliberately not.
Open and close apps, switch windows, type or paste, run a saved routine. ASH
can now see the screen but cannot act on it.
*Builds on:* screen sense, actuators; needs a confirm-before-acting rule.

### Calendar and email
**Status: calendar built**, read-only from .ics files and feeds (`src/daemon/calendar_ics.py`). **Email is not built**: it needs account access and a send-confirmation flow.
Read the schedule, notice clashes, draft replies. Needs connectors and a
strict "never send without confirmation" rule.

### Research on request
**Status: built** (`src/daemon/research.py`): Wikipedia summary, your files, your past mentions.
"Look into X and tell me tonight": gather, summarise, and report at the next
free moment.
*Builds on:* free moments; needs web search.

### Knowing who is speaking
**Status: built as a rough voice match** (`src/daemon/voiceid.py`), opt-in, tested on synthetic voices only.
Voice identification, so a guest saying "Ash" is not treated as the user.
*Builds on:* the mic sense, the social learner.

### Explaining itself
**Status: built** (`src/daemon/explain.py`).
"Why did you do that?" answered from the executive's decision and the drive
state at the time.
*Builds on:* the executive, homeostasis, the decision explorer.

### Smart home control
**Status: built** through Home Assistant (`src/daemon/smarthome.py`); locks, doors and heaters are never operated by voice.
Lights, plugs and similar devices, if there is hardware for it.

---

## 3. From BMO: the playful companion

### Games
**Status: built** (`src/daemon/play.py`): a number game and a quiz on your own people.
Quick word, number and trivia games, including trivia from ASH's own memory
("Who is Sara to you?").
*Builds on:* the free-moment registry (a new activity kind).

### Music and sounds
**Status: built** (`src/daemon/media.py`): media keys and chimes. Sounds per face expression are not.
Play, pause and skip music; small sounds that go with face expressions; humming
while thinking.
*Builds on:* the face player, TTS.

### Its own small wants and moods
**Status: partly built**: "I'm a bit bored, want a game?" when curiosity is up and your mood is good. The face does not yet show it.
The drives already exist internally, but ASH never voices them ("I'm bored,
want to play something?"). The face could show them more clearly.
*Builds on:* homeostasis, the face bridge, free moments.

### Stories and pretend play
**Status: built** as an evening story offer, told by the language model. Pretend play is not.
A short story or bit of role-play when the user is idle in the evening, never
when the mood answer says the user is low.
*Builds on:* free moments and the mood rule already in it.

### Alarm clock and timers with personality
**Status: built** (`src/daemon/reminders.py`), with snooze and a chime.
Using the face and voice rather than a plain beep.
*Builds on:* reminders, the face, TTS.

### Take a photo or record a moment
**Status: built** (`src/daemon/media.py`).
On request, saved as an episode ASH can bring up later.
*Builds on:* episodic memory; needs a camera sense.
