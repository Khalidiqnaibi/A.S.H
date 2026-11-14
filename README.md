# A.S.H — Canonical System Documentation

**Version: Brain-First Runtime Architecture**

---
# setup
``` bash
pip install -r "requirements.txt"
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## LLM backend

ASH picks its chat backend from `ASH_LLM_MODE`, and the default is **`ollama`** --
fully local, nothing leaves the machine.

| `ASH_LLM_MODE` | Needs network | Config |
|---|---|---|
| `ollama` (default) | no | `OLLAMA_URL` (default `http://127.0.0.1:11434/api/chat`), `OLLAMA_MODEL` (default `mistral:latest`) |
| `openrouter` | yes | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |
| `local` | to whatever you point it at | `MISTRAL_LOCAL_URL` |

If the backend cannot be built -- no key, bad endpoint -- ASH logs a warning and
runs without an LLM rather than refusing to import. The brain still routes,
remembers and acts; it just stops narrating.

Prefer a **non-reasoning** model. With `qwen3`/`deepseek-r1`, ASH's long system
prompt plus chain-of-thought can consume the entire token budget and return
empty content. `tools/LLM.py` detects that case, warns, and falls back to the
reasoning text -- but `mistral:latest` simply works.

## Project layout

| Path | What is there |
|---|---|
| `ashd.py`, `ashctl.py` | the daemon, and the command line that talks to it |
| `app.py` | the older web app |
| `src/brain/` | routing, the critic, planning -- deciding what to do with a sentence |
| `src/daemon/` | the always-on runtime (`ambient.py`) and one module per feature: media, desktop, reminders, the camera (`looking.py`), saying commands loosely (`commands.py`), ... |
| `src/senses/` | sensors and actuators: microphone, camera, screen, speech, address scoring |
| `src/memory/` | core, entity and episodic memory (`paths.py` says where their files live) |
| `src/vision/`, `src/face/`, `src/ui/`, `src/concepts/` | the camera's models, the animated face, the console, the concept graph |
| `tools/` | the brain's tools (calculator, files, system stats, ...), the LLM client, the classifier; `tools/web/` the face editor and the decision explorer (open them in a browser) |
| `tests/` | every test, and `run_all.py` to run them -- see `tests/README.md` |
| `deploy/`, `offline/` | running as a service; the air-gapped install |
| `documentation/` | notes, ideas and the roadmap |
| `entities.json`, `episodic_memory.jsonl`, `core_memory.json` | ASH's memory, in the folder it runs from (or `ASH_MEMORY_DIR`) |
| `state/`, `logs/`, `daemon.json` | what the daemon writes while it runs; personal, not committed |

## Tests

```bash
python tests/run_all.py             # every test, one after another, with a summary
python tests/tst_commands.py        # one of them
```

Test files named in this document (`tst_*.py`) live in `tests/`. A run never touches your own memory files, and no test
acts on the machine: media keys, apps, the camera, mail and the network are all fakes. `tests/tst_e2e.py` is the end-to-end
evaluation -- the real brain inside the real daemon, fed sentences the way the microphone would, and scored.

## Air-gapped install

`offline/` builds a flash drive that installs ASH on a machine with no network
at all -- Python, Ollama, wheels, the LLM, the HuggingFace cache and the spaCy
and TTS models, all bundled.

``` powershell
# on an ONLINE machine
powershell -ExecutionPolicy Bypass -File offline\build_usb.ps1 -Out D:\ASH_OFFLINE_KIT

# rehearse it here first, with the network blocked
powershell -ExecutionPolicy Bypass -File offline\simulate_airgap.ps1

# copy the folder to the drive, then on the OFFLINE machine
powershell -ExecutionPolicy Bypass -File E:\ASH_OFFLINE_KIT\scripts\install_offline.ps1
```

Full instructions, sizes and troubleshooting: [offline/OFFLINE_README.md](offline/OFFLINE_README.md).

---

# Identity

**ASH** is a brain-first, deterministic, tool-oriented AI runtime designed for:

* Low-resource systems (e.g., Raspberry Pi)
* Always-on operation
* Harsh environments
* Controlled behavior
* Structured memory
* Stable personality

ASH is not a chatbot.

Conversation is a rendering layer.
Cognition happens deterministically.

---

# Core Philosophy

ASH operates under six principles:

1. Deterministic Routing, Distributed Authority
   No single subsystem decides alone. Actions are arbitrated by a weighted
   board with a hard cap on any one member's share of the vote.

2. Two-Speed Cognition
   Cheap, confident, reversible actions bypass the LLM entirely. Novel,
   contested, or consequential ones buy deliberation.

3. Layered Memory
   Memory is structured, separated by type, and retrieved at a depth
   proportional to how hard the turn actually is.

4. Predict, Then Act
   A latent forward model scores its own prediction error every turn.
   Surprise is what turns a reflex into deliberation.

5. Drives Over Emotions
   Emotion still shapes tone only. It is compressed into latent drives that
   shape how much thinking gets purchased -- never which action is correct.

6. Bounded Learning
   The only thing that learns online is the critic's estimate of *when it is
   safe to skip thinking*. Nothing learns what to say.

---

# High-Level Architecture

```
        +-----------------------------------+
        |  Homeostatic Modulator (Limbic)   |
        +-----------------+-----------------+
                          | Latent Drives
                          v
+-------------------+  +--------------------+  +--------------------+
| Sensory Extractors|->|   Executive Core   |<>|  Predictive Model  |
|   (Text / ViT)    |  |  (arbitration +    |  |  (JEPA, runs in    |
+---------+---------+  |   pathway gate)    |  |   concept space)   |
          |            +---------+----------+  +--------------------+
          v                      ^
+-------------------+            |
| Concept Graph     |------------+
| (k-WTA + Hebbian  |  proposes, recalls, supplies the latent
|  rewiring)        |
+-------------------+
              Fast Reflexes      |      Retrieval
                       v         |         v
        +-------------------+    |   +-------------------+
        |  System 1 Policy  |    |   | Episodic/Semantic |
        |   (Actor-Critic)  |    |   |  Memory Engine    |
        +-------------------+    |   +-------------------+
                                 v
                       +--------------------+     +-----------------+
                       | System 2 (slow)    |<--->|   VLA Channel   |
                       | candidates ->      |     | (embodied /     |
                       | imagination ->     |     |  irreversible)  |
                       | narration          |     +-----------------+
                       +--------------------+
```

Implemented in `src/brain/`. `src/py/ash.py` is now a thin adapter over it.

---

# Execution Pathways (fast / slow)

Every turn is assigned exactly one pathway by `ExecutiveCore.gate()`.

| Pathway | Cost | Tool | Memory retrieval | LLM |
|---|---|---|---|---|
| `OBSERVE` | ~0.3 ms | no | working memory only | no |
| `REFLEX` | ~0.02 ms | none (cached) | working memory only | no |
| `FAST` | ~1 ms + tool | yes | shallow probe, hard core rules only | **no** |
| `SLOW` | seconds | yes | full three-store | yes |
| `ESCALATED` | slow + wasted fast | yes | full | yes |

`OBSERVE` is what daemon mode spends ~99% of its cycles in: perceive,
update novelty/surprise/drives, remember, say nothing.

Any one of these forces the slow path:

* critic value below the drive-modulated floor
* classifier match below the drive-modulated floor
* effective surprise above ceiling
* percept novelty above ceiling
* board contested (top-two margin under 8%)
* action class is `MUTATION`, `IRREVERSIBLE`, or `PHYSICAL`

Plus one exploration case: high curiosity on a moderately novel input buys
deliberation the gate didn't strictly require, so the critic gets training
signal on cases it would otherwise always shortcut.

**Post-hoc escalation.** A fast turn that executes and then fails -- tool
error, empty result -- is discarded and re-run as `SLOW`. A reflex that
overshoots gets caught by deliberation, which is the point of having both.

---

# Distributed Control

No subsystem has full authority. This is enforced structurally, not by
convention:

* **Weight cap.** `MAX_MEMBER_WEIGHT = 0.34`. `ExecutiveCore.register()`
  raises if any member would exceed it. Carrying a motion requires at least
  three concurring interests.

| Member | Weight | Jurisdiction | Veto |
|---|---|---|---|
| System 1 | 0.26 | all | no |
| System 2 / LLM | 0.26 | all | no |
| Predictive Model | 0.20 | all | no |
| VLA | 0.18 | physical, irreversible, mutation | bounded |
| Homeostasis | 0.10 | all | no |
| Constitution | 0.00 | all | **absolute** |

* **Bounded vetoes.** The VLA may only veto inside its own jurisdiction, and
  any bounded veto is overridden by normalized support of 0.72 or higher from
  the rest of the board. It can stop a reckless motor command; it cannot hold
  the system hostage over a calculator call.

* **Split powers.** The Constitution holds the only absolute veto and has
  *zero* positive voting power -- it can forbid, never cause. Authority to
  stop and authority to act are held by different bodies.

* **Deadlock is a valid outcome.** When the top two proposals are within 8%,
  the board does not break the tie; it escalates to deliberation, and the
  narration asks a clarifying question.

The LLM's standing: it may **nominate** one tool per slow turn, which then
goes to the board like anyone else's proposal, subject to VLA and
Constitution veto. It executes nothing and holds 0.26 of the vote -- below
the cap, so it cannot pass a motion alone. The original "LLM as narrator"
constraint survives; the LLM gained the right to suggest, and nothing else.

---

# Cognitive Layers

## 1. Classification Layer (now: System 1 actor)

Purpose:

* Detect user intent
* Map query to command
* Avoid LLM reasoning for routing

Characteristics:

* Uses sentence-transformer embeddings (e.g., all-MiniLM-L6-v2)
* Single unified command classification
* Unknown tag for unsupported queries
* No runtime retraining
* Commands updated during maintenance phase

**Keeping the fast path open.** A confident match runs its tool directly,
with no model. Three feedback loops once closed that path entirely, so every
request, even "what's 15 times 23", went to the model:

* **The critic's reward.** The critic, System 1's sense of what a reflex is
  worth, was trained with −0.4 on every turn that went slow, tool or not.
  Since most turns are talk, it learned that everything is worth −0.4, and
  then fewer requests went fast. Now only turns where a tool ran train it: a
  tool that worked is rewarded, fast or slow, and one that failed is
  punished. A critic saved under the old reward starts again from its
  warm-start priors (`reward_version` in `state/brain/critic.json`).
* **Outcome surprise.** It rose with every first successful use of a tool,
  because an untried tool's 0.5 counted as a prediction. It is now weighted
  by the evidence behind the prediction.
* **The novelty ceiling.** A question on a new topic is "novel" next to the
  one before it. Novelty no longer sends a confident match on a harmless,
  reversible tool to the model. Weak matches and anything that changes
  things still deliberate.

* **A clear reflex needs no vote.** The board's tally was mostly the
  homeostasis nudge, and it left "read the file" and "convert km to miles"
  (matches 0.64–0.66) a hair under the bar. When the classifier clearly
  knows the request (match ≥ 0.60), the critic values it (≥ 0.30) and the
  tool is harmless and undoable, only a positive score is needed. The gate
  still checks surprise and the rest, and anything that changes things
  still deliberates.

`tst_fastpath.py` checks these, using the numbers from a real session.

---

## 2. Deterministic Tool Layer

Tools are pure Python functions.

Examples:

* date_time_tool
* calculator_tool
* future hardware commands

Rules:

* Tools never call the LLM
* Tools never modify core memory directly
* Tools log usage
* Tool outputs are treated as facts

---

## 3. Memory System (Multi-Layer)

ASH does not use a single vector database.

It uses layered memory:

---

### 3.1 CoreMemory

Purpose:

* Identity
* Constraints
* Goals
* Standards
* Behavioral rules

Characteristics:

* Small
* High-precision
* Embedding indexed
* Rarely modified
* Loaded at startup

Examples:

* “ASH must not invent facts.”
* “ASH prefers deterministic execution.”
* “ASH is loyal to its user.”

---

### 3.2 EpisodicMemory

Purpose:

* Recent interactions
* Temporal continuity
* Experience history

Characteristics:

* Time-based
* Append-only
* Lightweight JSON storage
* Optional pruning by age

Examples:

* User asked about time
* User was frustrated
* Tool failed

---

### 3.3 EntityMemory

Purpose:

* Structured knowledge about people and things

Stored as:

* entity_id
* name
* type (person, object, location)
* attributes
* embedding for disambiguation

Used for:

* Remembering user preferences
* Tracking known individuals
* Persistent real-world grounding

---

### 3.4 MemoryRouter

The cognitive dispatcher.

Responsibilities:

* Retrieve relevant CoreMemory items
* Retrieve recent episodic entries
* Retrieve matching entities
* Build structured context block
* Save new episodes
* Trigger entity updates if needed

MemoryRouter replaces generic VDB retrieval.

---

# Emotional System

ASH maintains a persistent emotional state:

```
{
  calm: 80,
  focus: 75,
  frustration: 5,
  curiosity: 60
}
```

Properties:

* Values range 0–100
* Stored in persistent state
* Influences tone only
* Does NOT override deterministic behavior
* Slowly decays toward baseline

Emotion affects:

* Response warmth
* Verbosity
* Directness
* Slight phrasing shifts

Emotion never affects:

* Tool selection
* Command execution
* Memory structure

---

# Runtime Behavior

One turn = one `Brain.think()` call, nine steps:

1. Settle the previous turn's deferred reward (the critic learns).
2. **Reflex check** -- exact recent repeat? Return the cached answer, done.
3. **Sense** -- fuse modalities, compute salience and novelty.
4. **Predict** -- score last cycle's prediction, emit surprise.
5. **Modulate** -- run the emotion engine, derive latent drives, broadcast.
6. **Propose** -- System 1 reflex arc, plus the VLA if it has standing.
7. **Arbitrate + gate** -- the Executive picks an action *and* a pathway.
8. **Execute** -- fast (tool + template) or slow (candidates -> imagination
   -> re-arbitration -> full retrieval -> narration).
9. **Commit** -- memory write, outcome recorded, prediction staged.

ASH does not:

* Retrain the classifier or the LLM at runtime
* Rebuild intent embeddings during interaction
* Let any single subsystem execute without board approval
* Depend on a cloud VDB for core cognition

The only online learning is the System 1 critic (11 weights) and the
predictive forward model. Both are bounded, inspectable, and persisted to
`state/brain/`.

---

# Maintenance Phase (Sleep Mode)

Triggered on idle by `MaintenanceScheduler`. In addition to episodic pruning,
`brain.sleep()` now runs:

* **Replay** -- retrain the forward model on its replay buffer for several
  shuffled epochs. Online SGD gets one gradient step per transition; replay
  gets many, which is the difference between tracking and generalizing.
* **Recovery** -- repay accumulated `fatigue`, relax all drives back to their
  homeostatic setpoints, so a session that ended frustrated doesn't start the
  next one frustrated.
* **Cleanup** -- clear the reflex cache, working memory, and the cached
  constitution so new hard rules take effect.
* **Persist** -- flush critic weights and the forward model to disk.

---

---

# Always-On Mode (daemon)

```bash
pip install -r requirements-daemon.txt   # optional; everything degrades gracefully
python ashd.py --check                   # what can it actually perceive/do?
python ashd.py                           # run it
python ashctl.py status                  # talk to a running daemon
```

Config is `daemon.json`, written with defaults on first run. Everything
invasive is **off** by default and every actuator is **dry-run** by default.

## Closer to Jarvis and BMO

Each of these is its own module in `src/daemon/` (or `src/senses/`), on by default in `daemon.json` unless it says
otherwise, and has its own test in `tests/`. All of them work by rule -- the language model is used only where it
says so.

| Say | What happens | Module, setting |
|---|---|---|
| (nothing -- it notices) | One well-timed line from what ASH already knows: "Heads up: Standup starts in 10 minutes -- and you're still gaming", a low battery, memory nearly full, a birthday from your notes, "it's past 1 -- the coding can wait". Offered through the attention gate, so muting, quiet hours, focus and the hourly budget still apply. | `situation.py`, `situation.enabled` |
| "Why is my computer slow?", "what's using my memory / cpu?", "am I running out of space?" | The real numbers, programs grouped (40 Chrome processes are one Chrome). Under real pressure it offers to close the biggest program, and does it only on "yes" -- never ASH, its model server or Windows itself. | `pcdoctor.py`, `pcdoctor.enabled` |
| "Find the contract I was editing last week", "what did the lease say about pets?", "open it" | Your Documents, Desktop and Downloads, indexed on this computer in the background (names, dates, the start of the text; by meaning too, with the encoder ASH already loads). The sentence about pets is read back from the file. | `filesearch.py`, `filesearch.folders` |
| "Show the heads-up display" / `ashctl hud` | A small always-on-top panel: the clock, what's coming up, what's on, what's playing, who's here, ASH's last line, a box to type in. Same themes as the console. | `src/ui/static/hud.html` |
| "Reply to Omar saying I'll be late", "make it more formal", "send it" | A draft from your words, threaded under Omar's email, read back in full. Sent only on "send it", and only with `email.allow_send` switched on (it is off). Rewording is the one step that uses the model. | `drafts.py`, `email.allow_send` |
| "Find my phone", "send it to my phone", "text Omar saying I'm late" | Through KDE Connect (install it on this computer and the phone, pair them). Texts are read back and sent on "send it". On Windows it can't hear incoming calls -- KDE Connect's tool only sends. | `phone.py`, `phone.contacts`, `phone.ntfy_url` |
| "Guard mode on" ... "I'm back" | While you're away the camera watches; someone unknown or movement is sent to your phone as words (no pictures). Your own face coming back ends it, with what happened. | `guard.py`, `guard.enabled` |
| (talk over ASH) | It stops at once -- from the sound level, above its own echo, before a word is transcribed. Transcription starts as you go quiet, so the words are ready about 0.3 s sooner, and speech is made and played a sentence at a time. An on-device wake word is optional (`sensors.microphone.wake_word_model`, an openWakeWord model; not installed by default). | `src/senses/livevoice.py`, `sensors.microphone.barge_in` |
| "How are you feeling?" | From feelings that last: thanks and laughs lift them, being told off lowers them, hours alone bore it, late nights make it sleepy. They fade over hours, survive a restart, and show on the face. | `feelings.py`, `feelings.enabled` |
| "Tell me a joke", "sing a song about my cat", "what are our inside jokes?" | Original jokes, none repeated until all are told; made-up songs; a line you laughed at is remembered and comes back now and then. | `fun.py`, `fun.enabled` |
| "Kids mode for 30 minutes" | Grown-up things refused kindly (email, texts, files, notes, apps, smart home, buying...), the model told it's talking with a child, a not-for-kids word filter on everything said, a time limit. Only a grown-up turns it off (`kids.passphrase` optional). | `kids.py`, `kids.minutes` |
| "Take a silly photo", "take a photo with the cartoon filter" | "Three, two, one -- cheese!", one frame, a filter (cartoon, sketch, mirror, retro, comic, BMO frame), saved to `~/Pictures/ASH`. Never in privacy mode, never for a guest. | `photobooth.py`, `photobooth.dir` |

What has **not** been tried on real hardware: the microphone and speaker changes (tested with synthetic audio and a fake
player), KDE Connect (tested with a fake tool; it isn't installed here), and guard mode and the photo booth on a real
camera (tested with synthetic frames). The thresholds -- how loud counts as talking over ASH, how much change counts as
movement -- are starting points to tune.

## Attention: when ASH speaks

Every ambient event gets one of three dispositions, and ~99% of a day's
events get the middle one:

| | meaning | cost |
|---|---|---|
| `IGNORE` | below the noise floor | nothing |
| `OBSERVE` | perceived, remembered, silent | one embedding, sub-ms |
| `RESPOND` | full cognitive cycle | seconds |

Two independent routes to `RESPOND`:

1. **Addressed** — the user said ASH's name or made a direct request.
   Unconditional: no threshold, no refractory, no budget. Being spoken to and
   ignored is the worst failure an assistant has.
2. **Volunteered** — nobody asked, but something crossed the bar. The bar
   moves: raised by refractory decay, user busyness, fatigue and caution;
   lowered by urgency and the curiosity drive. Backed by a **hard** cap on
   unsolicited remarks per hour that no combination of drives can override.

`ashctl.py pause` stops *perceiving*. `ashctl.py mute` stops *speaking* while
continuing to observe and remember. Those are different needs.

## Briefings

`briefings` in `daemon.json`, `src/daemon/briefing.py`. One short spoken
summary at the edges of the day instead of scattered remarks:

* **Morning** -- the first time you are at the computer and pause: today's
  events (from what you mentioned: "your interview at 10:00"), what you
  worked on yesterday, and whether ASH has questions waiting.
* **Evening** -- at a free moment: what the day went on, with times
  ("brain.py (2h30) and A.S.H (20 min)"), and tomorrow's events.
* **On request** -- "brief me", "what's my day look like", "what did I do
  today".

The weather is part of both once a place is set (`weather.city`, or `lat` and
`lon`): today's in the morning, tomorrow's in the evening, from Open-Meteo
(no account or key; only the place is sent, at most hourly), and "what's the
weather like today?" is answered from it without the model -- another place
is left to the brain's weather tool.

Once a day each, never while muted or in quiet hours, and nothing at all on
a day with nothing to say. Extra parts (reminders, weather) are one function
each (`Briefings.add_provider`). `ashctl.py briefing` shows what it would say.

## Watching things for you

`watch` in `daemon.json`, `src/daemon/watch.py`. Say "tell me when the
download finishes", "...when chrome closes", "...when the battery is
charged" (or "at 80%"), "...when the CPU calms down", or "...when
report.docx changes". It is checked against the machine first ("I don't see
firefox running right now"), acknowledged without the model, checked every
few seconds, said once when it happens -- even unprompted, since you asked --
and let go after a day if it never does. A download counts once it has
settled and nothing is still downloading. Muted or paused, the news is kept
and said afterwards. "Stop watching" or "never mind the download" drops
them; `ashctl.py watches` lists them.

Builds are out of reach and it says so: "tell me the program's name, like
'tell me when npm finishes'". Separately, disk use is sampled hourly, and if
at the current rate it will be full within two days, ASH says so, once a day.
It says so only for a steady climb: at least four hours of samples, growing
in most of them and still growing now. A burst of downloads isn't enough.

**Disk space advice.** "How do I free up space?", "storage
recommendations", or a plain "what do you recommend?" within three hours of
that warning, get an answer without the model. It covers how much is free,
whether it is falling and how fast, the quick wins (Recycle Bin, temporary
files, Storage Sense), and the biggest things in Downloads.

## "Why did you say that?"

`src/daemon/explain.py`. Everything ASH says leaves a small note of why, and
"why did you say that?", "why are you asking?" or "why did you do that?" is
answered from it in plain words, without the model: an answer says what it
read your request as and which tool it used; a question says what it did not
know and how it noticed ("you'd mentioned Omar 3 times"); a remark says what
made the moment seem right; a watch or an action says you asked for it. The
full decision trace is still there for debugging (`ashctl.py why`).

## Calendar

`calendar` in `daemon.json`, `src/daemon/calendar_ics.py`. Read-only, from
iCalendar: exported `.ics` files (`files`) or a private feed address
(`urls`: Google's "secret address in iCal format", Outlook's "publish
calendar"), with no account access and no extra library. One-off, all-day and
repeating events (daily, weekly on given days, monthly, yearly, with counts,
end dates, intervals and exceptions) go into the briefings ("On your calendar
today: 10:00 Standup, 14:30 Design review") and answer "what's on my calendar
today / tomorrow" without the model. Files are re-read when they change;
feeds are fetched every half hour on their own thread.

## Smart home

`smarthome` in `daemon.json`, `src/daemon/smarthome.py`, through Home
Assistant's local API: set its `url` and a long-lived access `token`. "Turn
on the living room lights", "switch the kettle off", "dim the bedroom lamp to
30%" and "is the porch light on?" are matched to devices by their friendly
names and done without the model. Only lights, switches, fans and on/off
helpers are touched; locks, doors, garages, alarms, heaters and covers are
never operated by voice -- that belongs in Home Assistant, not with a
speech recogniser that can mishear.

## Voice match

`voice` in `daemon.json`, `src/daemon/voiceid.py`. Nothing happens until you
say "learn my voice": the next six things you say into the microphone are
averaged into a rough voiceprint (MFCC statistics, `src/senses/voiceprint.py`).
After that, a clearly different voice is a guest: still answered, but not
allowed to act for you (apps, watches, the house, research, your calendar --
"Sorry, I only do that for you") or to teach ASH about you. Anything in
between gets the benefit of the doubt, and typed input is always you. It is
a rough match, not security -- a recording of you would pass -- and it has
been checked on synthetic voices, not yet tuned on real ones. The print is
the only thing saved (`state/voice.json`); "forget my voice" deletes it.

## Reminders, timers and alarms

`reminders` in `daemon.json`, `src/daemon/reminders.py`. "Remind me to call
mom in 20 minutes", "remind me tomorrow at 9 to send the report", "set a
timer for half an hour", "wake me up at 7" are kept and said when due -- the
reminder tool used to parse them and nothing ever looked again. "Snooze" (or
"snooze 10 minutes") right after one goes off brings it back; "cancel the
timer", "what are my reminders?" do what they say. Being asked for, they go
off even in quiet hours; muted, they wait. They survive a restart, one missed
while ASH was off is said late ("I was off when this was due at 14:10"), and
today's are in the morning briefing.

## Playing

`play` in `daemon.json`, `src/daemon/play.py`. "Let's play a game" starts a
number game (higher / lower) or a quiz on your own people, built from what
ASH has been told ("Who is Omar to you?" -- and it says when its memory
disagrees with you). A game holds the conversation until it ends or you say
"stop". At a free moment, when your mood is good and ASH's own curiosity
drive is up, it may say "I'm a bit bored. Want a quick game?"; in the
evening it may offer a short story, which the language model then tells.
Neither comes to someone who is low.

### More games

`src/daemon/games.py`. Besides the number game and the people quiz:

- **Word association.** "Let's play word association": ASH says a word, you
  say the first one it brings to mind, and it answers yours. After six
  rounds it reads back the chain.
- **Twenty questions.** ASH thinks of a thing. You ask yes-or-no questions
  (alive? bigger than a breadbox? can you eat it?) and guess with "is it
  a ...?".
- **Riddles.** "Tell me a riddle": "hint" for a hint, "I give up" for the
  answer.
- **Trivia.** "Ask me a trivia question". There is also one question a day,
  offered at a free moment. A riddle is offered now and then when things are
  going well.

All of them are read by rule, with no model.

### Good moments and celebrations

`cheer` in `daemon.json`, `src/daemon/cheer.py`.

**Good moments.** A follow-up that ended well ("how did the interview go?"
"great!") is remembered. Say you're nervous or worried about something, and
ASH brings one back: "You've done this before -- remember your interview in
March? That went well." It picks the same kind of thing first. "Tell me
something good" lists the recent ones.

**Celebrations,** each said once at a free moment:
- your birthday (from the fact you told it), or first thing in the morning
  briefing;
- streaks of 5, 7, 10, 14... days in a row on one project;
- a stretch of two hours or more finished.

What was celebrated is kept in `state/cheer.json`, so a restart doesn't
repeat it.

### A story that continues, and a voice for games

`src/daemon/storybook.py`. The evening story is told by the brain. ASH keeps
where it got to, and the next evening offers "Want the next part of last
night's story?". The brain is then given the words it ended on and asked to
carry on, not start over. "What happens next?" asks for the next part right
away; "tell me a new story" starts fresh, and so does a story left for a
week. It is kept in `state/story.json`.

Game lines are spoken in ASH's own voice. `play.voice` can name a different
Kokoro voice for them (e.g. `bm_george`); if that voice fails, the usual one is
used. (It used to default to `bm_george`, which made the trivia sound like someone
else.)

### Caring nudges

`care` in `daemon.json`, `src/daemon/care.py`. ASH adds up how long you've
been at the machine; ten minutes away resets it. At a free moment it may
say:
- after 90 minutes, "maybe stand up and stretch";
- at 2 hours, "have you had some water lately?";
- at 3 hours, "you've been at it for three hours straight".

Each is said once a stretch, at most one an hour, and only when your mood
is fine. "No more nudges" turns them off until the next restart.

### Face sounds

`face.sounds` in `daemon.json`, off by default. You can turn it on in the
console's settings. Curious, happy, very happy and alert get a small chime,
synthesized on the spot. It plays only on lines ASH volunteers, never in
quiet hours, and at most one every 20 seconds. When you're away, the face
already goes sleepy.

## Music, sounds and photos

`media` in `daemon.json`, `src/daemon/media.py`. "Pause the music", "play",
"next song", "previous track", "turn it up / down" and "mute the sound" are
sent as the keyboard's media keys, so they work with whatever is playing.
A short system chime goes with reminders and watch alerts, so a spoken line
is not the only sign. "Take a photo" saves one webcam frame to
`~/Pictures/ASH` named by the time, and remembers it as an episode -- only
on request, never while privacy is paused, never for a guest's voice.

## Research on request

`research` in `daemon.json`, `src/daemon/research.py`. "Look into the Raft
algorithm", "research ZXQ", "find out about WebGPU when you can" is
acknowledged and done on its own thread: a Wikipedia summary, a strong
definition in your own files or in something you said, which of your files
mention it, and how often you have. The report is told at the next free
moment (ahead of a joke) or in the evening briefing, once, crediting where
each part came from -- and says plainly when nothing was found.

## The face reacts on its own (`src/face/moods.py`)

Only the animations that already exist, by rule (no language model):

- **What you say.** "Thank you" shows love, a laugh shows very_happy, "good night" or "I'm tired" sleepy,
  "wow" surprised, "you're wrong" sad, "shut up" annoyed, "hello" a wink, "I'm sorry" a nod. The face answers
  before the voice, and again when the reply lands (so a nod doesn't cover the love).
- **Idle life.** After 45 s with nobody talking ASH looks around, double-blinks, now and then perks up curious
  or winks (one every one to three minutes, never the same twice), and late at night gets sleepy. It stays
  still while ASH is answering or speaking, when you're away, and in quiet hours.
- **Things that happen to ASH.** Mic or screen on: curious; camera on: surprised; a sense off or privacy mode: sleepy; privacy off: happy.
- **When asked.** "Wink", "show me a big smile", "what faces do you have?" still work (voice commands).

## Quiet till a time, undo buttons, "why" card

- "Quiet till 9", "do not disturb for 2 hours", "quiet until 3:30 pm" mute what ASH volunteers until then
  (it still answers you) and it returns on its own; "wake up" ends it early (`src/daemon/quiet.py`).
- The console: a ring flashes on the face when ASH hears it was spoken to; the face stage takes a slow tint
  from its mood; each reply's route tag opens a "how I got this" card; each thing ASH noted or set up has an
  undo button for ten minutes.

## Scenes, pretend play and greetings

All by rule, on in the daemon (`scenes`, `personas`, `rituals` in `daemon.json`, `"enabled": false` to stop one).

- **Scenes** (`scenes.py`): "work mode", "wind down", "break" set quiet time, a face and, if listed, apps to
  open, in one phrase, and the reply says only what really happened. Add your own under `scenes` (`quiet_min`,
  `apps`, `face`, `say`); "end work mode" or "I'm back" ends it; "what scenes do you have?" lists them.
- **Pretend play** (`personas.py`): "be a pirate for ten minutes", "talk like a robot", "pretend you're a
  detective". The model words its answers in that voice (told to stay true); rule answers like timers and
  commands are untouched. Ends after the time (default 10 min, at most an hour) or on "be yourself".
- **Greetings** (`rituals.py`): "good morning" gets a greeting by the real hour, plus the morning briefing the
  first time; "good night" gets a one-line recap of the day.

## ASH's manual (in its core memory)

`src/daemon/manual.py` writes one entry per area into core memory at start-up (category `capability`, ids
`manual_*`): what ASH can do, the words that make it do it, and what it can't (no web browsing, sending mail
or messages, calls, buying, location). Retrieval is by meaning, so "can you remind me when I get home?" brings
up the reminders entry and the model answers from it instead of guessing. It is idempotent and keeps the
manual in step with the code (`tst_manual.py` runs every phrase it teaches). "What can you do?" still answers
live with what is actually switched on here.

## Macros, and who may do what

- **Save what you just did** (`tasks.py`): after a plan you approved, say "save that as start work"; or after a
  few commands, "save the last three things as quiet desk" / "make that a macro called wrap up". Saying the
  name later shows the plan and asks once. "What tasks do you know?" lists them; "forget start work" removes one.
- **Voice gating** (`voiceid.py`, once "learn my voice" is done): a guest's voice is still answered, but is
  refused the clipboard, scenes, plans, home reminders, turning the mic, camera, privacy or ASH off, and
  changing or forgetting the enrolled voice. A voice that is only unsure is treated as yours.

## A small model for small talk

Set `"llm": {"light_model": "qwen2.5:1.5b"}` in `daemon.json` (or `ASH_OLLAMA_LIGHT_MODEL`) and ollama keeps that
model loaded too. A greeting, a thanks, "how are you", "I'm tired" or a joke (`src/daemon/modelpick.py`) is answered
by it; any question, task, number, link, picture or pretend-play line goes to the main model. When in doubt, main.
With no light model set nothing changes.

## Coming home, the clipboard, heads-up cards and the clock

- **"Remind me when I get home to ..."** (`arrival.py`, `arrivals` in `daemon.json`, on by default): home is a
  Wi-Fi network. Say "this is home" while on it (or set `arrivals.home`); the reminder is said when the laptop
  rejoins it after being away a few minutes. "What's waiting for me at home?", "cancel my home reminders".
  No location services; only the network name is looked at, on this machine.
- **Clipboard, only on request** (`clipboard.py`, off unless `"clipboard": {"enabled": true}`): "what's on my
  clipboard", "summarize / explain / translate my clipboard" (that one copy goes to the model with the
  question), "copy that" (what ASH last said), "clear my clipboard". Text that looks like a password or key
  is never read out or passed on. ASH never watches the clipboard.
- **Heads-up cards** (`headsup.py`): reminders due in the next 30 minutes, calendar events in the next hour
  and what waits at home show as small cards under the face. Silent; speaking stays the reminders' job.
- **Idle clock**: after two minutes with the console untouched (and ASH neither talking nor thinking), the
  page becomes a big clock with the date, the face's mood and the cards. Any tap or key brings the console back.

## Short facts from Wikipedia (on by default)

`facts` in `daemon.json`, `src/daemon/facts.py`; the switch is "Answer short
who/what is... questions from Wikipedia" in the console's settings. "Who was
Ada Lovelace?", "what is a black hole?", "what does ephemeral mean?" is
answered with the first sentences of the article and "(Wikipedia)" after
them, quoted, not guessed. It sends the topic, and only the topic, to
en.wikipedia.org. The daemon has it on so a small model never has to guess a fact;
set `"facts": {"enabled": false}` or untick the console setting to stop it. It only runs for
something said to ASH, never for talk overheard or text read off the screen.
A topic with several meanings asks which; one with no page says so; no
connection says nothing and lets ASH answer as usual. It leaves alone what is
not a topic: the time, the date, the weather, sums, "what is your name",
"what is this", questions about you, and quantities ("the population of...").

## "Remember...", "who am I?", "where did you get that?"

`src/daemon/remember.py`, `src/daemon/explain.py`; `tst_remember.py`. By rule,
never the model, because the model made up the confirmation:

- "Remember Lieth is my friend" / "...is Khalid's best friend" / "my brother
  is Omar" / "Omar is Sara's cousin" is written to the people ASH knows (and
  entity memory) and answered as written ("Got it -- Lieth is your friend.").
  "Best", "close" and "good" friend are all *friend*. A family tie that clashes
  with one on record is refused, saying so; "undo that" takes it back.
- Anything else to remember is a **note** (`src/daemon/notes.py`; `tst_notes.py`):
  "remember that I parked on level 3", "note that the wifi password is hunter2",
  "keep in mind I'm allergic to peanuts", "don't forget the passports are in the
  top drawer". It is kept as said, turned round to be said back ("Noted: you
  parked on level 3."), in `state/notes.json` on this computer. Asked "where did
  I park?", "when is my sister's birthday?", "do you remember where I put the
  passports?", it answers from the note -- the newest one, and when it was told
  ("You told me you parked on level 3 (this morning)."). Every word that matters
  in the question must be in the note, and a note about someone the question
  does not name is never used: "when is my birthday?" is not answered with your
  sister's. "What did I tell you about the wifi?" lists them, or says it was told
  nothing; "what do you remember?" reads the last few. "Forget where I parked"
  and "undo that" drop one. "Remember to call mom" is a reminder, not a note.
  Not from overheard speech, not for a guest, and an answer waits while someone
  ASH doesn't know is in view. `notes.enabled` switches it off.
- "Who am I?" / "who's the user?" / "what do you know about me?" says your name
  and the people on record, or that it doesn't have your name.
- "Where did you get that?" says whether the last answer was a tool, the
  model's own training (and so may be out of date), Wikipedia, or ASH's rules.

Also: the model stays loaded for 6 h (`ASH_OLLAMA_KEEP_ALIVE`) so only the first
question after a long idle pays a load; brain consolidation ("a night") runs
at most once per 6 h (`sleep_brain_gap_s`; `ashctl sleep` forces one); a
repeated observation within 20 s isn't re-processed; `ashctl history` lists
answered turns (add `all` for observations too); a markdown link or a
placeholder address (`@email.com`) in a reply is dropped even if memory holds it.

## Likes and dislikes (a profile, from what is said)

`src/daemon/taste.py`; `tst_taste.py`. ASH knew who people were and nothing
about them. Now what you say about taste is kept, per person:

- **Learned from** "I love Interstellar", "I can't stand horror films", "my
  favourite band is Radiohead", "I'm not a fan of country music", "Interstellar
  is amazing", "I skipped Tenet", "I listen to lo-fi when I'm coding", and the
  same about someone else on your word -- "Ali likes jazz", "he hates cilantro"
  (he/she is whoever you just mentioned; with nobody mentioned it is not
  guessed). "Love" and "hate" count for more than "like"; "kind of" for less.
  Saying it again strengthens it, saying the opposite replaces it. The occasion
  is kept with it ("when he's sad", "while coding") from a short list of moods
  and activities. Questions, "if/wish/would", "I like how this works" and
  "I like your idea" are not tastes. A capitalised word that entity memory
  already has as an organisation ("Netflix likes...") is not a person.
- **Where it lives**: in the knowledge store (`state/wonder/knowledge.json`,
  `tastes`), so it is saved, undone ("undo that"), forgotten ("forget
  Interstellar", "forget Ali") and mirrored into entity memory as a short
  `likes` / `dislikes` on the person by the code that already does all of that.
- **Asked back by rule**, never guessed: "what do I like?", "what do I hate?",
  "what music do I like?", "what do I like to watch?", "what does Ali like?",
  "do I like jazz?". "Who am I?" mentions it too.
- **Used** only when the question is about choosing something -- "what should I
  watch tonight?", "recommend some music" -- as background the model is told
  not to read back. Never for a guest's voice, which also never writes to it.
- **Also from what is played** (below), as a weaker source: it never overrides
  what you said.

`ashctl.py profile` shows it, `ashctl.py profile forget [name]` clears one
person's or everyone's. Settings under `taste` in `daemon.json`: `enabled`,
`learn_from_others`, `mirror`, `in_answers`, `from_playing`, `listen_s`, `watch_s`.

## What is playing (a song or a video, from window titles)

`src/senses/nowplaying.py`; `tst_nowplaying.py`. A player puts what it is
playing in its window title, so ASH reads the titles of the windows that are
open -- and only of known players: Spotify ("Artist - Song"), a YouTube,
YouTube Music, Netflix, Disney+, Crunchyroll, Twitch or SoundCloud tab in
Chrome, Edge, Firefox, Brave, Opera or Vivaldi, VLC, mpv, foobar2000,
MusicBee. Every other title is dropped on the spot. Nothing on the screen is
read and no sound is heard; a window on your blocked list, or a private one,
is not reported. It looks every 5 seconds (about 30 ms) and says nothing
unless something starts, changes or stops. Those events are never written to
the journal.

- **Each play is followed from start to finish**, one slot per player, so a
  song and a show at the same time are two plays. A change has to be seen
  twice (titles flicker); a pause shorter than 12 seconds is not a stop; a
  pause and a resume of the same song is one listen, not two fragments.
- **What it says about taste** (`Taste.watch`): a song heard for 90 seconds or
  more counts +0.10 for it (the artist gets half), one replaced within 30
  seconds counts -0.20 against it (the artist is not blamed), a stop is never
  a skip, a show or video watched for 10 minutes counts +0.10, a show dropped
  under two minutes counts against it, and hopping between videos counts
  for nothing. Three listens make a song something you like; two skips,
  something you don't. Behaviour never goes past +/-0.60 ("likes", never
  "loves") and never changes what you said yourself. When a song first
  makes it into the profile that is an undo step ("undo that").
- **Limits, stated in the manual too**: a title says what is *open*, not that
  it is making sound, so a paused browser tab looks like a playing one (Spotify
  is the exception: it goes back to its plain name when paused); a browser
  window shows only its front tab; Stremio keeps a blank title, so it is not
  seen; and ASH cannot tell whose music it is, so a guest's counts as yours.
  `NowPlayingSensor(audible=...)` takes a function returning the pids that are
  making sound, for a platform that can say.

What is playing can be asked and talked about: "what's playing?" / "what song
is this?" is answered from the player (never for a guest's voice, never by the
model), and while something plays, "I love this song" / "I can't stand this" /
"this is amazing" is kept as that song, said (so it stands over what was seen),
and "do I like this?" answers for it. "This" means the song only while one is
on; with nothing playing, or with "this place", nothing is guessed. `tst_taste.py` T8.

Settings: `sensors.now_playing` (`enabled`, `interval`, `skip_below_s`) and
`taste.from_playing`, `listen_s`, `watch_s`. The manual in core memory has a
`playing` entry.

## What goes with what ("you listen to this while coding")

`src/daemon/occasion.py`; `tst_occasion.py`. A song and a stretch of work are
two facts; the link between them is kept only once it has happened more than
once.

- **What you were doing** comes from which program was in front: editors and
  IDEs (VS Code, Cursor, PyCharm, JetBrains, Visual Studio, Sublime...) are
  *coding*; Word, Obsidian, Typora *writing*; Excel, PowerPoint, Outlook, Teams,
  Slack, Zoom *working*; PDF and e-book readers *reading*; Figma, Photoshop,
  Blender *designing*; Steam and a few games *gaming*. A browser or a terminal
  says nothing (it could be anything, and ASH's own terminal is one), so the
  occasion is left empty rather than guessed. Idle time is not an activity.
  Only the program name is used -- no titles, no screen text -- and two hours of
  segments are kept in memory.
- **The link**: a song or show heard through is counted against what covered
  at least half of it. Once the same activity has been seen **three times and
  in at least half of the plays**, the item is kept as liked "while coding",
  and that is said once and is an undo step. Skips never count. What you said
  ("I listen to lo-fi while I'm studying") stands over what was seen.
- **Asked back**: "what do I listen to while coding?" (also "when I'm sad",
  "what do I watch while exercising?"). With nothing seen it says so and how to
  teach it. "What should I listen to while coding?" is a request, so it goes
  to the model with what goes with coding listed first. Entity memory's `likes`
  carries it: "Lofi Beats by Lofi Girl (while coding)".
- **Not yet**: one app opened while another is in front ("opens this while
  looking at that"), and other people's habits -- those are only ever from what
  you say.

## Staying out of a game's way (`src/daemon/gamemode.py`, `tst_gamemode.py`)

ASH is always on, and on a gaming PC that is the problem: with League of Legends running it read the screen,
transcribed the game's sound, ran a vision model on every frame and held 2 GB -- 100% of CPU and RAM. When a game is up
ASH steps aside, and comes back by itself.

- **What counts as a game**: a known one (League of Legends, Valorant, CS2, Dota, Elden Ring, Minecraft... about 90
  names; launchers like Steam and Epic do not count), a known game running behind the window in front, or any program
  that fills the whole screen with no title bar -- unless it is a browser, a video player, Office, an editor, a
  terminal and the like. The window sensor now reports `fullscreen`.
- **What it does**: ASH (and Ollama's processes) drop to idle priority; the screen sensor (capture, OCR, vision) and the
  microphone (speech recognition on the game's audio) are paused -- only if they were on, and only until the game is
  gone; the vision model and OCR engines are released from memory; Ollama is told to unload the language model (the
  first question afterwards takes a few seconds while it loads again); and nothing is said unprompted (the console
  window is not opened over the game either).
- **When**: 8 seconds of a game before it acts, 45 seconds without one before it comes back, so alt-tabbing does not
  flap. `ashctl game` says what it is doing. `game_mode` in `daemon.json` switches each part: `enabled`,
  `enter_after_s`, `leave_after_s`, `lower_priority`, `pause_screen`, `pause_mic`, `free_model`, `fullscreen_counts`.
- **Always, game or not**: ASH starts below normal priority, with 2 threads for torch, onnx and BLAS (`ASH_THREADS`
  to change, `ASH_PRIORITY=normal` to opt out), Whisper on 2 threads, and the ViT vision model is no longer loaded just
  because a screenshot arrived (only with `ASH_ENABLE_VISION=1`, and then only from the local cache).

## What survives a crash (`src/memory/atomic.py`, `tst_lastsession.py`)

On a PC that froze or lost power 53 times in a month, `dwell.json` and `crashes.json.runs` were found empty at the next
start and the history was lost. Everything that holds ASH's memory is now written temp file -> flush -> `fsync` ->
replace, with the previous good copy kept as `.bak`; read back, a file that is empty or torn is set aside as `.bad` and
the backup is used. Entity and core memory no longer stop the daemon starting when damaged; an episode file with a torn
last line skips it and does not glue the next line to it; the concept network, system 1 and the predictive model are
written atomically and also on a clean stop (not only in the sleep phase, which a PC that is gamed on rarely reaches).
A full disk is said once a minute, not as a traceback every second; the journal keeps up to 500 lines it could not write
for the next try; a failed save waits instead of retrying every tick. Closing the console window, logging off and
shutting down now stop ASH cleanly (they used to just kill it, which showed up as an "unexpected shutdown"). A sensor
that failed five times rests for two minutes and is tried again instead of staying dead; a microphone that dies is
restarted. Speech is recognised on this machine or not at all unless `privacy.require_local_asr` is switched off (the
online fallback used to run whenever Whisper failed).

## Other fixes from that session

- Games and trivia are spoken in ASH's own voice (`play.voice` defaults to ""; it was `bm_george`).
- "Hit me", "bring it on", "let's go", "go for it", "I'm in" answer an offer, for half an hour (a plain yes still
  needs to be within two minutes). After trivia or a riddle ASH asks "Another one?"; "another one", "next", "that's
  easy" (a new question, with a remark) and "too hard" (sympathy and an offer) are about the game that just ended.
- "Thank you ash .. tell me a joke" is a joke by rule, not a 78-second model answer.
- One window that changes every frame (a film) is said once in a while (`sensors.screenshot.min_repeat_s`, 120), not
  405 times a day.
- A "never mind" can no longer leave the language model blocked for good, one event that raises no longer drops the
  rest of the tick's events, and the console no longer shows an answer that was cancelled or unfiltered kids-mode text.
- Answers are cleaned of what neither JARVIS nor BMO would say -- "I'm DeepSeek-R1", "as an AI", "I apologize for any
  confusion", "could you provide more context", "it seems like the user..." -- and the persona prompt says it
  (composed and a little dry, with a bit of BMO; play along when it sounds like a game; never "the user"). The stall lines
  while the model thinks are different each time and in character; a screen full of log lines is not read out.

## Where a sentence goes (and the test that pins it)

`tst_replay.py` runs real utterances from the real sessions through the real
brain: clear tool requests ("what time is it", "what's 15 times 23", "convert
10 km to miles", "list the files in my documents", "how much memory am I
using", and the same with "please" or "could you") stay on the fast path with
no model call; conversation goes to the model; and the control commands are
rules that never reach it. A tool that declines because it is switched off or
the machine has no such sensor ("Screen capture is off") is a correct answer,
not a failure: it is said as it is, and the critic and the outcome model learn
nothing bad from it. Politeness and the name ("please", "thanks", "hey ash",
"could you") are stripped before a sentence is classified and before a tool is
handed it (`src/brain/phrasing.py`). A clear reflex also isn't undone by the
deliberator proposing the same tool, by the effort-scaled critic floor, or by
a rival that is a far worse match.

## Desktop control

`desktop` in `daemon.json`, `src/daemon/desktop.py`. "Open Outlook",
"launch Spotify", "switch to Chrome" and "run my morning check" (a routine
you kept) are done at once, without the model. "Close Word" is asked first
("Anything unsaved there may be lost. Say yes to close it.") and is a polite
close the app can still interrupt to save, never a kill. Only names it knows
as apps are taken -- "open the readme" or "run the tests" still go to the
brain and its tools -- and you can add your own under `apps`. Nothing is
typed into windows and nothing is deleted.

Apps and lights are only opened, closed or switched on a line meant for ASH:
typed, said with its name, or said in a conversation with it. Speech it
overhears, from a video, a call or someone in the room, never does it. A
bare "yes" to its own "Close Word?" still counts. An app can run as more
than one process: Outlook is `outlook.exe` (classic) or `olk.exe` (the new
Outlook for Windows), and closing or switching to it finds whichever is
running. For your own apps, separate the processes with `|`.

## Focus mode

`focus` in `daemon.json`, `src/daemon/focus.py`. "I'm focusing for an
hour", "focus mode for 45 minutes", "do not disturb until 3" or just "let me
focus" (an hour): until then ASH volunteers nothing -- no questions, jokes,
check-ins, briefings or unprompted remarks -- while what you asked for still
happens (reminders and watches go off, questions to ASH are answered). When it
ends, on time or with "I'm done focusing", ASH says in one line what it held
back ("I held back 3 things; 2 questions are waiting; your research on Raft is
ready."). It keeps ASH quiet; it does not silence other apps.

## The weekly review

`weekly` in `daemon.json`, `src/daemon/weekly.py`. "How was my week?" gets a
line like this, built from what ASH already keeps:

> This week: 9 hours on A.S.H and 2 hours on thesis. 3 follow-ups closed, 4
> of my questions answered, 1 reminder. Omar came up most, then Sara.

- **Time** is activity memory's per-day time, by project, for the last seven
  days.
- **The counts** are follow-ups closed, questions answered and reminders that
  went off this week.
- **Who came up** comes from names you said, counted per day. Only the names
  are kept, for two weeks, in `state/weekly.json`.

It is also part of the evening briefing on `weekly.day` (Sunday by
default; set it to "" for never).

## Email, read-only

`email` in `daemon.json`, `src/daemon/inbox.py`. Nothing happens until an
IMAP account is set: server, account and an app password. You can set these
in the console's settings panel. After that, ASH checks the inbox at most
every ten minutes, on its own thread.

- **The morning briefing** says "Email: 4 unread, from Sara, Omar and
  GitHub."
- **"Any new email?"** gives the same; **"who emailed me?"** and **"what are
  they about?"** add the subjects.

It only ever reads who sent what. The folder is opened read-only and only
headers are fetched, with `BODY.PEEK`, so nothing is marked read. It never
reads a message body, and never sends, moves or deletes. The password is
kept in `daemon.json` in plain text, which is why an app password is
recommended. It is never shown back in the console, the chat or the status.

## Help when stuck

`src/daemon/stuck.py`. When the same error keeps coming back, ASH offers
help ("That error keeps coming back (KeyError: 'id') -- want me to take a
look?"). If you say yes, it does two things without the model:

- It looks for that error in your own files, under the lookup folders
  (`wondering.lookup.roots`), and says where it was met before, by file and
  line.
- It starts a background look-up of the error, and tells you when that is
  ready.

Only the error's own words are searched for. Paths, line numbers and
addresses that change every run are left out. If you answer with a question
of your own instead, the model answers as before.

## Multi-step tasks

`tasks` in `daemon.json`, `src/daemon/tasks.py`.

**One request, several steps.** "Open Outlook, set a timer for 25 minutes
and turn off the desk lamp" is split into its steps and said back as a plan:
"Here's the plan: 1) open Outlook, 2) ... Go ahead?". Nothing runs before a
yes. On a yes, each step goes to the part of ASH that already does it:
apps, timers and reminders, music, lights, watches, look-ups. What came of
each step is said back in one line, naming anything it couldn't do. A
"Close Word?" in the middle is taken as answered by the approval. "Undo that"
still takes back a timer it set.

**Named tasks.** "When I say start work, open Outlook and set a timer for 50
minutes" keeps the steps under that name. Saying "start work" later shows the
same plan and asks once. "What tasks do you know?" and "forget the start
work task" list and remove them. They are kept in `state/tasks.json`.

**What counts as a step.** A request is a plan only when every part starts
like a command ASH can carry out. "Remind me to buy milk and eggs" stays one
reminder. The console shows a pending plan in its "waiting for you" strip.

## "Open it", "remind me about that", "who was that?"

`src/daemon/context.py`. ASH keeps track of the last app opened, closed or
in front, what you said last, the last person named, and its own last line.
A sentence that leans on them is rewritten into the full request, then
handled as usual:

- **"Open it" / "close that" / "switch to it".** The app last opened or
  closed, or the one in front. ASH's own windows and blocked windows don't
  count.
- **"Remind me about that later".** "I need to call the bank" becomes
  "remind me to call the bank in 1 hour". A time you give is kept.
- **"Look it up".** The subject of a recent question ("what's a quasar"), or
  a short phrase. Never a whole sentence.
- **"Who was that?"** The last person named, and who they are to you if ASH
  knows.
- **"Say that again".** ASH's last line, again.

All of these go stale after ten minutes. What it can't resolve is left to the
brain.

## Stop, shh, the short version

`speech` in `daemon.json`, `src/daemon/speech.py`.

- **"Stop" or "shh".** Cuts ASH off mid-sentence, and it says nothing back.
  This also works for a few seconds after it finishes, since transcription
  lags. Typed in the console works too. When ASH is quiet, a plain "stop"
  or "enough" is left for the game, the music or the timer. Only "shh",
  "hush" and "stop talking" always mean stop talking.
- **"The short version" or "tl;dr".** Says the last answer again, cut to its
  first sentence or two.
- **"Keep it short".** Long answers are spoken short from now on, while the
  console still shows all of it. "You can talk more" undoes it. The setting
  is kept in `state/speech.json`.

The phrases are whole-sentence matches. So ASH's own "I'll stop asking about
that", picked up by the microphone, is not taken as a "stop". Speech plays
asynchronously, which is what lets it be cut.

## Undo and forget

`src/daemon/forget.py`. Whatever ASH learned or set up on its own leaves an
undo step: a relation or a definition picked up from a sentence, an answer to
one of its questions, a follow-up, a reminder, a watch, a correction. "Undo
that" (or "scratch that", "take that back") reverses the latest one within
ten minutes and says what it undid ("Undone: Omar is your cousin."), entity
memory included. "Forget what I said about Omar" (or "forget ZXQ", "forget my
morning check") removes the subject from what ASH believes -- people, how
they know others, roles, terms, routines, entity memory -- and closes the
questions about it. Conversation history is not rewritten. A guest's voice
can do neither.

## Talking less, by learning

`src/daemon/backoff.py`. The attention gate learns per broad kind of remark;
on top of that each specific thing -- the joke activity, the mood check, the
story offer, a kind of question such as "stale" or a guess about someone --
keeps its own score from how you reacted: "no thanks" or a swipe counts 1,
"not now" half, silence nothing, and a welcome clears it. Three points within
a week bench that one thing for three days; the rest carry on. Urgent
questions are never benched, and asking "any questions?" still gets them.
Benched things are listed in `ashctl.py status`.

## Reviewing what ASH said

`remarks` in `daemon.json`, `src/daemon/remarks.py`. Everything ASH says is
logged with its kind (an answer, a question, a joke, a briefing, a
reminder...), whether you had asked, and how it landed (engaged, accepted,
deferred, rejected, met with silence). `ashctl.py review [days]` reads it
back: how much was said per kind and per hour, how each kind landed, lines
said over and over, anything in the small hours -- and what to tune ("wonder:
unanswered 6 of 8 times -- it may be coming at the wrong moments"). Kept two
weeks under `state/remarks/`; `keep_text: false` keeps no words at all.

`tst_remarks.py` also runs a whole simulated day through the runtime with
everything on and checks the budgets, the quiet hours and repetition hold.

## Status and settings in the console

`src/ui/settings.py`. When ASH is waiting on an answer, a strip above the
chat says so: "Close Word?", a question it just asked, a game in progress, an
offer, what it didn't understand, or that you are focusing. It has buttons
for the likely replies, so a stray "yes" never surprises anyone. The
**Status** button opens a drawer with two parts:

- **Up to.** The next reminders, watches, open questions, follow-ups,
  look-ups in progress, routines it has noticed, what it is holding off on,
  and recent corrections.
- **Settings.** A fixed list:
  - feature switches;
  - the weather city;
  - calendar feed links;
  - the Home Assistant address and token;
  - the folders to look terms up in.

A change applies at once wherever that part of ASH is running. It is also
saved to `daemon.json` (only that key, written atomically), so it survives a
restart. A part that was off when ASH started says "applies after a
restart". The token is never sent back to the page or written to the chat.

## Questions ASH asks

Everything here is rule-based, paced by the same budget, quiet hours and
"not now" as any other remark, and spent only at a free moment unless it is
urgent. Each part can be switched off in `daemon.json`.

The user's mood, which decides whether a light remark or an idle question is
welcome, comes from the sentiment model the brain already runs on every
query: each turn moves it (one bad turn pulls a good mood down rather than
replacing it), and the answer to "how are you holding up?" is read by the
same model. An unsure reading counts for nothing. No word lists.

* **Follow-ups** (`follow_ups`, `src/daemon/followups.py`). "I have an
  interview on Thursday at 10" is remembered with its time. Once it is over,
  ASH asks how it went, once, and reads the answer with the sentiment model
  (went well, badly, so-so), plus "not yet" and "didn't happen". Moving or cancelling the event updates it, and
  saying how it went first means it is never asked. An exam outranks a
  routine meeting. `ashctl.py followups` lists what it is waiting to ask.
* **Look it up first** (`wondering.lookup`, `src/daemon/lookup.py`). Before
  asking what a term means, ASH checks episodic memory, text files under
  `roots`, and, for acronyms and quoted terms only, a Wikipedia summary
  (only the term is sent; set `web: false` air-gapped). Whatever it finds is a
  guess to confirm: "I think ZXQ means Zone eXchange Queue. Is that right?".
  A yes keeps it, "no, it's ..." keeps the correction, a bare no asks once
  what it is, then. For a file it keeps seeing (`brain.py`), it reads what the
  file says about itself first.
* **Contradictions** (kind `conflict`). Two answers that cannot both be true
  are not silently overwritten: "Omar is my cousin" and later "Omar is my
  brother", two definitions of ZXQ, two birthdays. ASH keeps the first and
  asks which is right. Family ties conflict; a colleague who becomes a friend
  does not. A replaced role is asked as a change: "Is Sam your boss now,
  instead of Dana?", and Dana is kept as your former boss.
* **Stale facts** (kind `stale`, `stale_after_s`, default 180 days). A
  relation that can change (boss, roommate, partner, landlord...) is checked
  when the person comes up again after a long while: "Is Dana still your
  boss?". Yes refreshes it; "no, she's my friend now" or "not anymore" updates
  it. Family is never asked about.
* **Being corrected** (`corrections`, `src/brain/corrections.py`). "No, I
  meant search my files" right after a reply that used a tool is answered as
  what was meant; a bare "that's not what I asked" gets "Sorry -- what did you
  want?". The reading that was wrong counts as a failure for the brain, and
  the correction is kept as a bounded nudge on the classifier for the same
  wording next time. `ashctl.py corrections` lists them; `corrections forget`
  clears them.
* **Guesses about people.** When a name keeps coming up next to work words
  (a meeting, a deadline, the team), classes or family, "Who is Sara to you?"
  becomes "You often mention Sara around work. Is Sara a colleague?". A yes
  keeps it; a no asks the open question. No pronoun is ever guessed.
* **Routines** (`routines`, `src/daemon/routines.py`). The first apps opened
  at the start of a sitting, in order. One that repeats on three different
  days is offered once: "You often start your morning by opening Outlook,
  then Teams, then Jira. Want me to remember that as a routine?". "Yes, call
  it morning check" keeps it under that name. Only app names are kept.
  `ashctl.py routines` lists what it noticed and what you kept.
* **Which question first.** Besides how urgent a gap is, questions are
  ranked by how much the answer would be used: a subject that comes back day
  after day outranks one said in a single burst, and one from today outranks
  one not heard of for days.
* **"What are you wondering about?"** (or "any questions for me?") gets the
  open questions there and then, one at a time, each next one riding on the
  acknowledgement of the last, without the hourly pacing. Muted subjects are
  never offered; "that's all" stops it and keeps the rest for later.

## Ambient memory: two-stage consolidation

An always-on ASH sees 5,000–20,000 events a day. Writing each into episodic
memory destroys that store within a week.

```
event ──► journal (JSONL, daily, TTL 7d)  ──sleep phase──► rollup
                                                              │
                                            activity blocks ──┴──► episodic memory
```

So a day is remembered as *"worked in VS Code for two hours, three breaks,
one call at 14:00"* rather than as 6,000 window switches. Rollup is
deterministic bucketing — no LLM, nothing to hallucinate. Verified in the
test: 430 raw events → 2 episodes.

## Sensors and actuators

Add one by subclassing `Sensor` (implement `read()`) or `Actuator` (implement
`act()` plus declare an `action_class`). Both self-register and self-report
availability.

| Sensor | Default | Notes |
|---|---|---|
| `active_window` | on | title + process + dwell; blocklist-filtered |
| `idle` | on | drives "is the user interruptible" |
| `system` | on | CPU/RAM/disk/battery, **edge-triggered only** |
| `screenshot` | off | perceptual-hash dedup drops >90% of captures |
| `microphone` | off | 3-gate pipeline: level → VAD → transcribe |
| `camera` | off | motion-gated; an empty room produces zero events |
| `serial` | off | JSON or `key=value` lines from any Arduino/ESP32 |

| Actuator | Class | Default |
|---|---|---|
| `notify` | mutation | on, live |
| `speak` | mutation | off |
| `clipboard` | mutation | off, dry-run |
| `launch` | irreversible | off, dry-run, **allowlist required** |
| `input` | irreversible | off, dry-run, rate-limited, blocklist-checked |
| `shell` | irreversible | off, dry-run, **allowlist, no shell=True** |
| `motor` | physical | off, dry-run, bounds-clamped |

Actuators are bound to **both** the VLA channel and the tool registry. The
registry binding is how the classifier routes to them; the VLA binding is
what puts them under the Executive's veto. A `physical` actuator therefore
inherits the grounding requirement automatically — **no fresh vision percept,
no motor movement** — without the actuator author writing any safety code.

## Privacy

A process with a mic, a camera and a screenshot loop is one bug away from
being spyware. The difference is entirely in what it refuses to keep.

* **Redaction on the write path** — cards, emails, tokens, hex secrets and
  `password=` patterns are scrubbed *before* anything is journaled. The secret
  never lands on disk.
* **Blocklist before capture** — password managers and banking windows are
  matched on title, and no pixels are read at all. There is no sensitive
  frame sitting in RAM waiting to be filtered.
* **Retention** — raw journal 7d, frames 2d, raw audio never. Summaries
  survive; raw observation does not.
* **Local only** — install `faster-whisper` for local ASR.
  `require_local_asr` defaults to True and will refuse the Google fallback
  rather than silently upload room audio.
* **One-call pause** — `ashctl.py pause` halts every privacy-sensitive sensor
  within a tick.

A microphone in a shared space records people who did not opt in. Code cannot
solve that; `announce_on_start` at least makes ASH's presence visible.

## Controlling ASH by voice

`src/daemon/voicectl.py`. What the console's buttons do, said out loud. It
is done by rule and said as done; the language model never gets these.

| Say | It does |
|---|---|
| "privacy mode", "go private" | turns the mic, camera and screen off (type "privacy off" or press Resume to undo) |
| "turn off the mic", "stop listening" | turns just the microphone off |
| "turn on / off the camera", "screen capture on / off" | turns that sensor on or off |
| "go to sleep", "quiet mode" / "wake up" | brings nothing up, with a sleepy face / ends it |
| "shut down", "turn off" | asks first, then stops the daemon cleanly |
| "smile", "look surprised", "wink", "make a sad face" | shows that face |
| "what faces can you make?" | lists the faces |
| "what's your status / CPU / memory?" | gives the real numbers and which sensors are on |

Turning something off works from any line the microphone hears, since it
only makes ASH more private. Turning something on needs a line meant for
ASH, and nothing turns on while privacy mode is on. If the model's answer
claims one of these ("I'm turning off now"), ASH says the truth instead and
names the command.

### Saying it any reasonable way (`src/daemon/commands.py`)

A command does not have to be said exactly as written above. When no rule takes a line as it was said -- and only
when it was said *to* ASH (by name, in a conversation with it, or typed) -- the same rules are tried again on:

1. **the line cleaned up**: no wake word, no politeness or filler. "Ash, can you pause the music for me please"
   is "pause the music".
2. **the line spelled right**: a word that is not a word at all ("pasue the musci", "opne notepad", "go to slep") becomes
   the command word it is a letter or two from. A real English word is never changed -- the local encoder's word list
   says what is one -- so "what should I cook tonight" stays as it is.
3. **the nearest known command**: "pause the music that's playing" is "pause the music", "start a focus session for 45
   minutes" is "focus for 45 minutes". Sure: it is done. Close but not sure: ASH asks "Did you mean "turn the volume
   up"?" and a yes does it.

This takes well under a millisecond and never uses the model: a misspelled command is done in a couple of milliseconds
instead of waiting 20-70 seconds for an answer. A question for the brain with a typo in it ("whats the tiem") is passed
on spelled right, so it still gets the fast answer.

It is never stretched further than that:

- overheard speech (no name, no conversation) is never read loosely -- a video saying "play that again" presses nothing;
- a negation ("don't pause the music") or a question about a command ("why did you pause the music?") is never run;
- only the handlers that *do* things see a loosened line; "can you help me" does not become a command "help me".

`runtime.loose_commands` (on by default) switches it off.

An instruction said *without* ASH's name that no rule could do ("turn left at the next light") was almost always said to
someone else: it is taken in, not answered. A question in the same band ("what time does the store close?") is still
answered, as before.

Weather, the calendar and the smart home, asked about before they are set up, say so and say where to set them up --
the model is not given the question, because it would guess a forecast or say "Done!" and switch nothing.

### Commands act on the real thing

A command is only done when the device has changed, not when a flag or the
screen says so. `tst_effects.py` runs them against the real classes.

* **Microphone off** stops the capture thread and throws away what it had
  heard; on starts it again. A microphone that can't work (no audio library)
  stays off and says why, rather than starting a thread behind its own back.
* **Camera off** lets go of the device, so its light goes out, and forgets
  the frame it was comparing against.
* **Screen capture off** drops a reading that hadn't been delivered, and a
  reading that finishes after the switch is thrown away.
* **Privacy mode** (voice, console button or `pause`) really switches every
  privacy-sensitive sensor that was on **off**, and brings back exactly those
  when it ends, not ones you had turned off yourself. It is also checked
  every tick: anything sensitive found running while it is on is stopped.
* **Turning something on** (voice or the console) is refused in privacy mode,
  and a sensor that is unavailable ("no camera at index 0") is reported with
  its reason instead of "Camera on."
* **A face** is the animation actually playing, or ASH says it couldn't.
* **Stop, quiet mode, shutdown** stop the speaker, mute the attention gate
  and ask the daemon to stop, each checked.

### "Never mind" while ASH is thinking

A model answer takes seconds to a minute and can't be aborted once started.
Saying "never mind", "forget it" or "cancel that" meanwhile is heard at
once (the fast lane) and the answer, when it arrives, is dropped instead of
said. The console's status shows what ASH is thinking about and for how
long, with a "Never mind" button. Said when nothing is being worked on, the
words keep their old meaning (dropping a question ASH asked).

**Hearing itself.** With speakers, the microphone picks up ASH's own voice
(`src/senses/echo.py`). ASH keeps what it is saying and when. A line heard
during or just after that, made mostly of its own words, is dropped. You
can still talk over it ("stop") because your words are different. Whisper is
told to expect the name, and "Ass …" at the start of a sentence is read as
"Ash …".

**Not trusting a doubtful transcript.** faster-whisper scores every phrase
it writes. A phrase it is unsure of (average log-probability below −1.0, or
probably-not-speech and unsure) is room noise or another voice, and is
dropped rather than answered. Each phrase is read on its own, so one doesn't
carry words into the next.

**Not saying things twice.** `src/daemon/repeat.py`. A model answer loses any
sentence ASH already said in the last ten minutes, comparing words in order,
so "the microphone is off and the camera is on" is not "the camera is off and
the microphone is on". If nothing new is left, it says "That's what I just
said." A rule's lines count as said, but are never trimmed.

**Questions you've since said again.** If you say a question again, or
correct it, while ASH is still busy with the first, the first is skipped and
only the later one is answered. Two different questions are both answered,
and a command in between is done.

**Smaller things that made it slow.** Fewer episodes and entities go into
the model's prompt (5 each, was 8). Episodes older than two weeks have to be
more on topic to be brought up, and older than two months more still (a
status question was answered with "the YOLO pipeline we discussed in late
June"). The model is asked for at most 200 tokens (`ASH_LLM_MAX_TOKENS`),
because answers are clipped to about 150 before they are said. The
embedding model is loaded once, on the CPU, and shared (`tools/shared_model.py`).
The disambiguator used to create its own with no device, which makes
sentence-transformers probe CUDA, the probe that can crash natively on a bad
GPU driver.

**Telling you the computer crashed.** `src/daemon/crashwatch.py`. At start,
and every six hours, ASH reads Windows' record of unexpected shutdowns. It
tells you about new ones in the morning or evening briefing, or when you ask
"did my computer crash?" ("The computer shut down unexpectedly at 05:33 today,
on a driver touching memory it shouldn't. That's 4 times in the last 30 days,
which is a pattern: worth testing the memory and updating the drivers."). The
first time it runs, what's already in the log is history, not news.

## Keeping ASH honest, and keeping its memory clean

**Its own words are not memories.** ASH used to save every answer it gave as
an episode. When the model invented a CV ("Phone +212 XXX-XXXXXX, Location:
Morocco") that became a memory, was found again four months later, and was
said back as fact. Now only what you said is stored, plus ASH's reply when a
tool was behind it (marked `A.S.H [tool]:`). The working buffer still has
the reply for the rest of the conversation.

To clear out what is already there, with ASH stopped:

```bash
python tools/memory_cleanup.py
```

It only shows what it would remove: ASH's own free-form episodes, and
entities that aren't things (document headings like "Contents" and "Summary",
words that only start a sentence, made-up placeholder addresses and numbers).
An address or number you really gave ASH is kept. Add `--apply` to back up
`episodic_memory.jsonl` and `entities.json` first and then clean them.

**It says what it can't do.** `src/daemon/honest.py`. "Show me cats online",
"search the web", "send an email to Sara", "call my mom" and "order a pizza"
get a plain "I can't do that" and what ASH can do instead, without the model,
which used to answer "I can use a web browser..." and print `[web_browser open
new tab]`. "What can you do?" is answered from what is actually switched on
(lights only when a smart home is set, the weather only with a place, and so
on).

**It doesn't read out its briefing or invent contact details.** After the
model answers, a pasted `Facts: { ... }` block and fake tool calls in square
brackets are removed, and so is any sentence with an e-mail address, phone
number or link that is nowhere in what ASH was given (what you said lately,
what memory returned). Dates, versions and times are never mistaken for phone
numbers.

**Commands don't wait for the model.** The loop is one thread and a model
answer takes 20-70 s. A "fast lane" thread runs only while the loop is inside
a model answer: it keeps listening to the microphone and runs control commands
at once ("privacy mode", "turn off the mic", "stop", "go to sleep", faces and
status), so none of them waits behind the answer. A question is left for the
loop. Text read off the screen is never a command.

**Smaller things.** "Switch screen capture on" parses. A smile or a status
question works without the name while you are mid-conversation (turning a
sensor on still needs it). A short "yeah" that whisper is unsure of is no
longer dropped. The intent-embedding cache remembers what it was built from,
so it isn't rebuilt on every start.

## If the machine crashes while ASH is answering

The language model runs in Ollama, which puts part of it on the graphics
card. A blue screen that comes while ASH is thinking (IRQL_NOT_LESS_OR_EQUAL
and the like) is a kernel driver failing under that load, usually the GPU
driver. ASH, a normal program, can't cause one on its own.

The fix is an up-to-date GPU driver. Until then, `llm.gpu: false` in
`daemon.json`, or "Use the graphics card for the language model" off in the
console's settings, keeps the model on the CPU. That's slower, but leaves
the driver nothing to trip over. The console switch takes effect on the
next answer, with no restart.

### Was ASH running when it crashed?

ASH writes down when it starts, keeps a heartbeat (once a minute) and notes
a clean stop (`state/crashes.json.runs`). A run with no clean stop was cut
off, and the crash is what cut it (Windows stamps a crash at the reboot, so
that run counts until the next start). With a day of history, "any crashes
lately?" adds "ASH was running during 5 of those 6, and it runs about 40% of
the time", and "is ASH causing the crashes?" answers from it, saying when the
numbers are more than chance and when they are not, and that it is a link,
not a cause. Too few crashes or too little history: it says so.

## Running it as a service

`deploy/` has a systemd user unit, a Windows Task Scheduler script, and a
launchd plist. The process handles its own signals and cleanup but does not
daemonize itself — the platform supervisor is better at restarts. An internal
watchdog restarts the ambient *loop* if it stalls; the supervisor restarts the
*process* if it dies.

```bash
python tests/tst_daemon.py    # verifies attention, privacy, journal, ambient loop
```

---

# Associative Layer (concepts & Hebbian rewiring)

A concept graph that learns which things go together and rewires itself
nightly. It **competes with** the embedding classifier at the Executive
board; it does not replace it.

That is the brain-faithful arrangement, not a hedge. Your brain doesn't route
everything through cortical association — the basal ganglia run a fast
stimulus→response habit path in parallel, and the two compete downstream. The
embedding classifier *is* the habit path. Keeping both is more faithful than
replacing one with the other.

## What a unit is

A concept owns a prototype vector in the same space MiniLM already produces.
An input activates the top-k by cosine (k≈12 of a few thousand) — sparse,
because sparsity is what makes Hebbian learning cost k² instead of N².

Seeds come from the tool registry, CoreMemory and EntityMemory, so they
arrive with labels. Units grown at runtime are **unnamed**, identified the way
a cortical unit is: by what fires them and what they connect to. `describe()`
builds that on demand — `c47<charger/desk>`.

Prototypes drift toward the inputs that fire them, so seeded intents migrate
away from wherever the catalog put them and toward how *you* actually phrase
things. The flat classifier is stuck with its catalog forever; this isn't.

## "Fire together, wire together" — and the four corrections

`w += η·aᵢ·aⱼ` fails in four predictable ways. Each has a specific fix:

| Failure | Fix |
|---|---|
| Weights diverge | **Oja's rule** — the update subtracts `post·w`, so growth decelerates to equilibrium instead of running away |
| Learns base rates, not associations | **PMI gate** — potentiate on co-activation *above chance*. This is the one people skip, and it decides whether the graph means anything |
| A few hubs eat the graph | **Homeostatic scaling** — each node's outgoing weight is multiplicatively rescaled to a fixed budget, preserving relative strengths while capping total influence |
| Only ever adds edges | **Decay + pruning** — unreinforced edges weaken and are deleted, freeing the unit to wire elsewhere. This is the half of "rewireable" that additive learning can't provide |

**A PMI consequence worth knowing before it surprises you:** two concepts that
fire in *every* cycle have PMI of exactly zero and will never wire together.
That's correct — if both are always on, one tells you nothing about the other
— and it's the same property that stops "the user typed something" from
associating with the entire vocabulary. But a unit with a base rate near 1.0
is inert as far as plasticity goes. The fix is to split it into something that
discriminates, not to lower `pmi_floor`.

## Three factors, not two

Pure Hebb learns correlation, which isn't usefulness. Every update is gated by
a third factor: the System 1 critic's TD error, which the brain package
already computes. Positive error consolidates associations that were active
when things went well. Measured in the tests: rewarded pairs consolidate **5×
harder** than unrewarded ones — but unrewarded ones still form something,
because you do learn things nobody rewarded you for.

## Two timescales

Every synapse carries two weights, which is early-LTP versus late-LTP:

* `w_fast` — potentiated on every cycle, decays with a ~90 min time constant
* `w_slow` — persistent, and **only ever written during the sleep phase**

So an association forms immediately, and becomes permanent only if it survived
until sleep and the reward signal agreed. A single-weight design forces a
choice between "learns nothing during the day" and "permanently rewires itself
on one coincidence".

## Growth and merging

Resonance: if the best match falls below `vigilance`, the input is unfamiliar
and a unit is allocated on the spot. Growth is deliberately generous — better
a spurious unit than a blurred distinction. Sleep-phase merging is what makes
that safe: duplicates created on different days by slightly different phrasing
collapse back into one, so the vocabulary tracks the diversity of your life
rather than the number of things you've said.

## What is open in the browser (no model)

`extension/ash-tabs`, `src/senses/browsertabs.py`, `src/daemon/pagekind.py`,
`src/daemon/ledger.py`; `tst_tabs.py`. Off until you turn it on: it is a record
of what you browse. Set `sensors.browser_tabs.enabled` in `daemon.json`, restart
ASH (it writes the extension's `config.json` with a port and a token), then load
`extension/ash-tabs` unpacked in Chrome/Edge (`chrome://extensions`, developer
mode, Load unpacked). The extension reports the tab list to `127.0.0.1` only;
the listener refuses anything without the token or from a web page's origin.

**Nothing in this path uses a language model or the network** -- `tst_tabs.py` T6
fails if it imports a model or HTTP client, touches the model object, or opens a
non-loopback connection. A page is classified by, in order: the user's own tag
("that's work"), a table of well-known sites, title keywords, and a small Naive
Bayes trained only by those tags; otherwise it is `unknown`, not guessed.

Kept: per stretch of attention the site (host only, never path or query), a
category, and the seconds it was the active tab of a focused browser window
(idle time and a quiet extension end it). Tabs opened and dropped within seconds
are counted as churn, not pages. Raw rows expire after `retention_days` (30) into
daily and per-site totals. Titles are used in memory to classify and stored only
with `keep_titles`.

Never recorded, dropped on arrival and only counted: private windows; banks,
payments, mail, chat, health, adult, dating, password managers, VPN, login and
account pages, private-network and router pages; pages whose title says
"symptoms", "password", "recovery key" and the like; non-web pages. The ledger
is `state/ledger.sqlite3` and can be deleted at any time.

### Asking about it (`src/daemon/browsing.py`, `tst_browsing.py`)

Said back by rule, from the ledger, with the evidence in the answer -- no model,
and a test that fails if one is touched:

| You say | It answers |
|---|---|
| "what have I been browsing this week / today?" | time per category |
| "how much time did I spend on youtube / on social sites?" | one site or category, for a period |
| "what sites do I use the most?" | top sites |
| "what am I into lately?" | interests: sites that came back on 3+ separate days, weight halving every 14 days |
| "when do I focus best?" | the two hours holding most of your code/docs/learning/work/reading time, and your longest unbroken stretch -- or what it still needs |
| "what do you know about my browsing?" | what is kept, what is not, how much |
| "tag this site as work" | teaches the site in front (and the words of its title) |
| "stop watching my tabs" / "resume watching my tabs" | pause and resume; nothing is read while paused |
| "forget my browsing" / "forget everything about github.com" / "forget my social browsing" / "forget my tags" | deletes it, really |

A pattern needs three separate days to be called one, and the answer says how
many days it rests on. It describes where time went, not what you are like. A
guest's voice is refused all of it.

### What else it does with that (`tst_insights.py`)

* **"While coding" in a browser.** A code, learning, work, docs, reading or games
  tab in front is an *activity* for `occasion.py` (a browser's window title never
  was), so a song heard through three coding tabs becomes "listens to this while
  coding". Video, social, news and shopping are "leisure"; nothing else is guessed.
* **The social learner knows what you are doing.** It already learned, per kind
  of remark and per situation, how welcome ASH is (a contextual bandit: shrinkage
  to priors, exploration bonus, forgetting, capped shifts). The situation now
  includes the activity -- `idle/evening/coding` -- with a middle level (kind
  while coding), so a joke can land at leisure and not at work. With no activity
  known nothing changes. This is the "RL-based UX": it learns from reactions
  (thanks, "not now", a swipe, silence), inside limits it cannot move.
* **It says what it learned.** "How are your nudges landing?" -- overall rate, best
  and weakest kind, most and least welcome by activity, what you told it to hold
  off on. "Forget how I react" resets it to neutral.
* **Suggestions and habits, as observations.** "Any suggestions about my
  browsing?" compares this week with last (leisure categories up by an hour and
  half again, focused browsing down or up, more tab hopping, time after
  midnight, when focus tends to fall) and says nothing if nothing stands out.
  "What are my browsing habits?" says how much, in which part of the day, how long
  a stretch, how many dropped tabs an hour. Counting and templates; it does not
  tell you what to do, and it describes where time went, not what you are like.
  With tab tracking off, these phrasings go to the model as before.
* **What I'm learning about you.** A panel in the console lists what each part
  holds right now -- tastes, what the ledger holds and in what form, interests,
  focus hours, reaction rate -- and how to delete each.

## The camera: seeing, recognising, remembering (`tst_vision.py`)

`src/vision/` (memory, backends, eye, presence, vlm) and `src/daemon/looking.py`. **Off until you turn it on**
(`vision.enabled` in `daemon.json`). Nothing here uses a language model except the optional last step.

What it keeps is **labels and numbers describing how a face or a thing looked -- never the picture**, unless you say "and
save the photo" (then only a path is kept, and the file is yours). Frames are never stored: one is held in memory for a
minute so "save that photo" can mean the photo just taken, and dropped when used, when old, and when privacy mode goes on.
Nothing happens in privacy mode, for a guest's voice, or from overheard speech.

| You say | What happens |
|---|---|
| "look around" / "what's in front of the camera?" / "can you see me?" | one frame; people (by name if taught), things you taught, and what the detector says, read out by rule. (Bare "what do you see?" is still the *screen*.) |
| "what am I holding?" / "look at this" | the thing nearest the middle: a thing you taught first, else what the detector calls it, hedged when unsure |
| "remember this as my blue mug" / "remember this person as Omar" / "remember my face" / "this is Omar, remember his face" | teaches it. Refuses when there is no face, when two faces are similar in size, and when the face already matches someone under another name. A second look improves it. "...and save the photo" keeps one |
| "who's this?" / "is this my mug?" / "is this Omar?" / "where are my keys?" | matched against what it was taught, cautiously: a clear lead over the next label is needed, a close call is "might be", a stranger is "someone I don't know". "Where" says in view now, or when it was last seen -- ASH has no map of your room |
| "what have you saved?" / "forget Omar's face" / "forget my blue mug" / "forget all the faces" / "save that photo" / "export the faces" | manage it. "Forget X" only counts when X was taught, so everything else still goes to the memory. A person's numbers go; what you *said* about them is separate |
| "what does this label say?" / "describe what you see" / "look and tell me if the stove is on" | **opt-in**, needs `vision.open_questions` and `vision.vlm_model` (a *local* multimodal Ollama model you pulled): one downscaled JPEG to a loopback address, told never to identify a face, and the reply says it is the model's reading |

**Presence** (`vision.presence`, off by default; needs `sensors.camera` on): every few seconds the camera sensor's *own* latest
frame is read (never a new capture) and only names and times are kept: who arrived, who left, who has been in view.
"Who's here?" is then answered without a new picture, and the console shows a card. **A stranger in view** (an unknown face on
two readings in a row) makes ASH hold back reading out the clipboard, mail, calendar and the day's briefing for two minutes
("Someone I don't know is in view -- ask me again when it's just you"). Saying "Omar just came in" aloud is a separate switch
(`announce_known`, `announce_strangers`), off, and never while muted, paused or quiet.

**Will it work here?** `python ashctl.py vision doctor` (with the daemon running) takes one picture -- only because you asked -- runs each model on it, and says what is ready, what failed and how many milliseconds each step took. Nothing is kept or learned. Teaching a face refuses a bad look (too small, dim, washed out, blurry, cut off) and says what to change; `vision.enrol_checks` switches that off. A model that breaks is reported as broken, never as an empty room, and the presence reading runs on its own thread so a slow detector cannot stall the daemon.

**Models.** `python ashctl.py vision install` shows what it would fetch -- the two face models from OpenCV's own model zoo
(YuNet, 0.3 MB; SFace, 37 MB) -- and fetches nothing unless you add `--yes`. The object detector is not downloaded for you: any
YOLOv8/YOLO11 ONNX file at `vision.objects_path` (`yolo export model=yolov8n.pt format=onnx` makes one). Without models it says
so instead of guessing. `python ashctl.py vision` lists what it knows; `vision forget [name]` deletes.

**Limits, plainly.** Things are told apart by colour and layout, not by a trained model: it separates a blue mug from a red one,
and will confuse two mugs of the same colour -- a second look in another position helps. Faces use SFace and are far better, but
dim light and odd angles lower it, which is why every answer has a confidence floor and an honest "not sure". Faces are
biometric data: only the people you enrol, numbers only, on this computer, deletable by voice or by deleting `state/vision.json`.
Tell someone you have taught ASH their face. **What was tested:** everything above against fake cameras and fake detectors (270+
checks), the detector output parsing against a synthetic tensor with known geometry, and the OpenCV/ONNX backends' interfaces.
**What was not:** a real camera, and the real YuNet/SFace/ONNX model files -- no model was downloaded in building this. Expect to
tune `face_threshold` (0.40) and `thing_threshold` (0.88) on your own camera.

## What ASH asked, and where you are (`src/daemon/asks.py`, `src/daemon/dwell.py`)

Two gaps that made ASH feel like it had no memory of its own conversation.

**What it asked.** Every question ASH puts to you is written down when it is said, with the situation (part of the day,
the program in front, how long you had been in it, whether you were typing or reading), in `state/asks/asks.jsonl`.
Your reply settles it -- *answered, said yes, said no, said later, talked past it, never replied* -- and the pair goes into
episodic memory (`A.S.H asked at 10:12 while on A.S.H body blueprints: "..." -- Immortal said no`), which is where the
conversation looks. "I have a question for you when you have a minute" stays open for hours, so *"go ahead"*,
*"what questions do you have?"* or a plain *"yes"* asks the question then and there instead of reaching the model cold;
*"not now"* puts it off. The model is told what it had asked when your message is the answer.
The answers also teach it *when* asking is welcome: a no or a "later" lowers the odds for that situation (at night, in
Edge, while you read), a yes or an answer raises them, and the free-moment chooser multiplies that in
(`state/asks/timing.json`). Ask it: *"what have you asked me?"*, *"when do I like to be asked things?"*, or
`python ashctl.py asks`. Set `asks.keep_replies` to `false` to keep only outcomes, not your words.

**Where you are.** `state/dwell.json` keeps the window in front and for how long, whether you are typing, pausing,
*reading* (same window, hands off the keyboard) or *away* (ten minutes without input, so a long read is not mistaken
for leaving), and the day's time by program and window. What it does with that, all by rule:
* reading one window for a few minutes is a **free moment** -- ASH says what it noticed ("you've been on *A.S.H body
  blueprints* for 12 minutes without typing -- anything you want me to keep a note of?") or asks a question it was saving;
* coming back after ten minutes away is **welcomed**, once, at your first pause ("Welcome back. You were away about 40 minutes.");
* *"how long have I been on this?"*, *"what have I been doing today?"*, *"how long was I away?"* are answered from it;
* the model is told which window you are in and for how long, and what you did earlier today.
Private windows (a chat, a mailbox) and blocked ones are counted as a program only -- their titles are never kept, said or
given to the model. Settings: `dwell.away_after_s`, `dwell.reading_after_s`, `dwell.welcome_back_after_s`.
Tests: `tst_asks.py`, `tst_dwell.py`.

### Free moments: when the user can't talk, and every other moment (`src/daemon/availability.py`, `tst_availability.py`)

A free moment used to be "a pause after two minutes of steady typing" -- rare, and the wrong question. The point is the
opposite: when *can't* the user be spoken to? ASH keeps out of those and is there in every other moment.
* **Never spoken to:** away (ten minutes without input), in a call (Zoom, Teams, a Meet tab), gaming, presenting, focus
  mode, in the middle of a conversation with ASH, typing right now.
* **Busy -- only something worth it:** coding or writing with input in the last minute, a video in front.
* **Free:** present, hands off, for 20 seconds (`free_moments.free_after_s`; 0 gives back the old rule). The longer free,
  the better the moment. The budget is 4 an hour, 5 minutes apart (`free_moments`), questions 3 an hour; the ask ledger's
  timing model weighs each moment by whether you have welcomed asking in that situation.
`python ashctl.py status` shows the current state under `availability`.

### Yes and no, said any way (`src/daemon/yesno.py`)

Every question that wants a yes (shut down, close an app, run a plan, play a game, send a text or a mail) reads the answer
the same way: *ye, yea, yeah, okay, do it, please, confirm, "yes, shut it down"* are yes; *no, nah, not now, cancel* are no;
"yes but later" is neither. A mail or text goes only on a clear yes, and only right after it was read back; "ok" or "mhm"
never sends. Saying "shut down" again confirms a shutdown.

### Never mind, the console clock, and what the camera says unprompted

* "Never mind" (said or the button) now cuts the model request loose at once -- the loop is free and the answer is dropped --
  and "Okay -- dropped it" is said once. The server may still finish its answer; nobody is listening.
* The console's full-screen clock appears after 15 quiet minutes (was 2) and goes away by itself when ASH says anything.
* The camera no longer says "you picked up a remote" (`vision.announce_objects`, off; it is still recorded on `/vision`).
  After a look, *"try again"*, *"how about now?"*, *"what's this?"* look again; *"nope"* owns up and *"it's a watch"* teaches it.

## Follow-ups to the browsing profile (`tst_followups.py`)

Each of these is its own commit, with its own section in `tst_followups.py`.

**Fixes** (each was confirmed by a probe before it was changed):

* A film with no keyboard input is not "idle": the idle cutoff is lifted while the tab in front is audible, up to four
  hours (six minutes of Netflix used to be kept as three).
* Churn means a tab opened and closed within ten seconds -- not a glance (it was counted twice) and not a quick
  switch between tabs that stay open (it counted each one). The answers say "opening and closing tabs within seconds".
* Forgetting a site removes its time from the totals too: the daily totals carry the site now (`daily_site`); older
  ledgers are migrated, their totals kept as "(older)".
* The never-record list covers web chat (Slack, Teams, Skype...), more mail, Drive/Photos/Keep, adult sites by name
  and by word, and patient portals.
* A bare "I hate this" is the playing song only if the track has just started (45 s); "this song" always is; "it's"
  is never kept as something liked or hated.
* The social learner's exploration bonus in a finer activity context is never larger than in the situation it came
  from (a new activity used to make ASH five times keener to speak up).
* One Ollama hiccup no longer switches the small model off for the session; only "model not found" does.

**Controls**

| You say | What happens |
|---|---|
| "don't record the next hour" / "go private for 30 minutes" | a timed pause that ends by itself ("resume watching my tabs" ends it early) |
| "never track youtube" / "start tracking youtube again" / "what sites are you ignoring?" | keeps a site out, deletes what was held, and says so |
| "show me everything you hold on me" | what is held, over which days, the top sites, what you told it |
| "export my browsing data" | one plain JSON file in `state/exports/` (sites, categories, time; no addresses or titles) |

**Profile and insight**

| You say | What happens |
|---|---|
| "what sites can't you place?" / "tag acme.dev as work" | lists the biggest unplaced sites; tagging also re-labels what was already recorded as unknown |
| "keep social under 3 hours a week" / "how am I doing on my goals?" / "remove my social goal" | a goal (per day or rolling week), progress by rule; an exceeded goal is first in the suggestions, a card in the console, and a line in the weekly review. Goals are settings, so "forget my browsing" keeps them |
| "how many tabs do I have open?" | the count, and how many have not been in front for six hours (ids and times only) |
| "how was today compared to yesterday?" / "...this week vs last week" | like for like (the same hours of yesterday), and where the time moved |
| "how long did I listen to music this week?" | sound from a background tab counts as listening, in its own stretches, never counted twice |
| "why did you say that?" (after something ASH volunteered) | adds how such remarks have landed in that situation, and what that did to how readily it spoke up |

The console also shows a card when focus mode is on and a video, social, games, shopping or news page has been in front
for ten minutes. Cards are never spoken.

Not built (and why): an encrypted ledger (needs key management the daemon does not have), pausing during screen sharing
(needs OS-level detection), learning the *style* of a nudge (a larger change to the social learner), and a
"stuck on an error, then drifts to social" pattern (too easy to say something wrong about a person).

## How ASH tells what a sentence is (`tst_intents.py`)

Three layers decide what happens to a sentence, and each had its own way of being wrong. A measurement of 267
real and realistic sentences found:

| Layer | What was wrong | What changed |
|---|---|---|
| **Addressing** (`senses/address.py`) | 98 commands ASH's own handlers understand ("switch to spotify", "work mode", "next song", "quiet for an hour") scored 0 and were silently dropped; "can you pass me that cable" scored 1.0; feedback like "you are working really fast" scored 0.25; "bash" for "Ash" was not heard as the name | `daemon/grammar.py` asks the handlers' own parsers ("is this a command I have?") and lifts a recognised command to *probably for ASH* (never to certain). Cues for the room ("you guys"), for a pair of hands ("can you pass...", "did you see..."), and kitchen appliances cap a sentence below "probably" unless the name was said or a handler recognises it. Remarks about the listener count as for it. "bash"/"as" at the end of a greeting-style question are read as the name. `address.grammar` in `daemon.json` (on by default) |
| **Routing** (the rule chain in `ambient.py`) | "make it louder", "lower the volume a bit", "play some music", "play a game", "open the browser", spoken numbers ("quiet for forty five minutes", "until three") matched nothing; "can you help me" recited the capability list; "forget start work" was taken by memory, so a saved task could not be forgotten by voice | those phrasings; `numwords.py`; `honest` no longer answers a request for help with a manual; a saved task wins "forget X" |
| **The tool router** (`tools/classification.py`) | the embedding threshold is 0.21, so a similar word called a tool: "I'm so stressed today" -> the clock, "what are you doing" -> a CPU table, "open the browser" -> read a file, "it rained all weekend" -> a forecast (27 of 267) | each tool that fires on weak similarity needs evidence in the sentence itself: a time/date word for `get_time`, a weather word for `get_weather` (both only for a question or a short request, not a statement about your day), a reminder word for `set_reminder`, a machine noun for `system_stats`, a file noun or path for the file tools, a question or order for the general lookups. A user's correction ("no, I meant X") still overrides it |

Also found and fixed: the media handler was the one acting handler with no "meant for ASH" check, so overheard
speech could press a media key; and "pause"/"stop" with nothing playing would have *started* something (the
now-playing sensor is now asked). Real attempts are caught: the test makes the real media and desktop executors
unable to act and records any try -- an earlier version of this measurement launched Excel, Calculator and Chrome on
the machine it ran on.

Not changed on purpose: a greeting with no name ("hello", "good morning") and a lone "yes"/"no"/"stop" are still not
taken for ASH unless it just spoke or was named -- they are as likely said to a person.

## What it buys you

Spreading activation (2 steps — one gives direct associates, three gives
noise) lets ASH notice that separate events are **one situation**. From the
end-to-end test, after a simulated week:

```
probe = late_night + low_battery        (in_editor NOT presented)
  matched : low_battery=0.78  late_night=0.78
  inferred: in_editor=0.66  save_work=0.66
  noise   : z5=0.14  z11=0.12
```

Two cues in, the third cue *and* the intent out. That's the missing piece in
the daemon's volunteer path: `battery 15%`, `still in the editor` and `02:00`
each fail the attention threshold alone, but they co-activate a shared region
and the sum crosses it.

Inference is always capped below perception (`inferred_cap`, 0.85). Without
that ceiling two connected units reverberate to saturation and an inferred
concept becomes indistinguishable from a perceived one — which silently breaks
the proposer, since it discounts associative routes by checking exactly that.

## Safeguards

**Frozen core.** Hard CoreMemory rules get frozen *inhibitory* edges to the
intents they forbid. Every plasticity pass skips them. The graph may learn
anything at all; it may not learn its way around a constraint.

**Board cap.** The associative member holds 0.15 of the vote — below the
classifier, because an association that *fires* a tool nobody asked for is
worse than one that merely suggests it.

**Rollback on regression.** A held-out set of query→intent pairs, built
automatically from ASH's own uncontested successes, is scored before and after
each consolidation. A drop beyond the margin reverts the entire pass and logs
it. Without this, a system that rewires itself nightly will eventually degrade
and you'll have no idea which night it started.

## Inspecting it

```bash
python -m src.py.ash        # then: 'concepts', or 'why' after any turn
ASH_CONCEPTS=0 python ...   # disable; ASH falls back to classifier-only routing
python tests/tst_concepts.py
```

`explain_last()` now carries a `concepts` block separating **matched** (what
the input hit) from **inferred** (what ASH connected) — which is almost always
the interesting half.

---

# Complementary Learning Systems

The associative layer above had one flaw that everything else followed from:
**one learning system with one rate**. Fast learning overwrites (catastrophic
forgetting); slow learning needs thousands of examples. A single network
cannot do both, and every continual-learning system that tries ends up bad at
both.

The brain's answer is two learners and a transfer channel:

| | rate | code | lifetime | learns from |
|---|---|---|---|---|
| **Hippocampus** | high | sparse, pattern-separated | days | **one** exposure |
| **Cortex** (concept graph) | low | overlapping, statistical | permanent | replay only |
| **Sleep** | — | — | — | moves the former into the latter |

So ASH knows something the moment you say it, and that knowledge becomes part
of its general structure only after being rehearsed against everything else it
already knows. Verified end to end:

```
told once      -> immediately recallable from hippocampus
after one sleep -> alice-coffee present in cortex (w=0.226)
                -> prior kitchen schema intact (0.600 -> 0.481)
                -> still episodic too
```

## What wires with what

Not everything co-active wires. Four constraints, each doing a distinct job:

**Typed edges.** `ASSOC` (symmetric co-occurrence), `CAUSAL` (directional),
`ROUTE` (concept→intent), `INHIBIT` (suppression), `PART_OF` (hierarchy). Each
has its own learning rate, decay rate and spreading gain. Causal and route
edges forget more slowly than coincidences, because a learned sequence is
worth more than a co-occurrence.

**STDP.** Pre-before-post potentiates, post-before-pre depresses. This is what
turns correlation into direction: `cause→effect` reaches 0.79 while
`effect→cause` goes *negative*, and the edge promotes itself to `CAUSAL` once
order evidence crosses threshold. Without it the graph can only ever represent
"these go together", never "this leads to that".

**Adaptation.** A concept firing in nearly every cycle is scaled down to near
silence for learning purposes. This is habituation, and it's why you don't
form associations with the hum of your own fridge. It also covers STDP, which
PMI can't — spike timing has no notion of mutual information, so without
adaptation an always-on concept forms strong timing edges to literally
everything.

**Frozen edges.** Hard CoreMemory rules keep their inhibitory links to
forbidden intents through every pass.

## What uses which system, and what trumps what

Four neuromodulators, derived from signals ASH already computes:

| | job | driven by |
|---|---|---|
| **ACh** | encode ↔ consolidate **switch** | novelty; floors during sleep |
| **DA** | third factor — which associations get stamped in | critic TD error |
| **NE** | global plasticity gain | surprise, urgency |
| **5-HT** | patience, fast/slow bias | inverse fatigue |

ACh is a switch, not a dial: the same network cannot encode and consolidate
without one corrupting the other. High ACh (novel, awake) → hippocampus
dominant, cortical plasticity suppressed. Low ACh (sleep) → hippocampal output
drives cortex. Measured: encode gain 1.15 / cortex 0.48 while awake, inverting
to 0.45 / 0.91 asleep.

This also fixes a plain waste: without neuromodulation, ASH spent the same
plasticity budget on the ten-thousandth window switch as on the one sentence
that mattered.

**Arbitration is by uncertainty, not by fixed weights.** Habitual and
goal-directed control compete in the brain and whichever currently estimates
itself more reliably takes over (Daw, Niv & Dayan 2005). Board weights are now
a *prior*, multiplied by each member's earned reliability. A member that keeps
being confidently wrong loses influence; one that earns it gains. The 0.34 cap
still binds after adaptation, and nobody can be silenced completely — a
silenced member could never earn its way back.

## When and how to save

Sleep has stages, and the order matters:

1. **NREM — replay.** Hippocampus re-presents episodes to cortex, **interleaved
   old with new**. Replaying only recent material teaches cortex the last day
   and overwrites the rest, which is exactly the forgetting the split exists to
   prevent.
2. **NREM — downscale.** Every synapse is multiplicatively weakened in
   proportion to the day's potentiation (synaptic homeostasis). Waking is net
   potentiating; without renormalization the graph ratchets upward until it
   can't discriminate anything. Multiplicative, so rank order survives — you
   wake with the gist, not the noise.
3. **REM — recombination.** Replay with spreading loosened, forming links
   between things that never co-occurred but sit in overlapping structure.
   This is where schemas get abstracted, and also where nonsense comes from,
   so it runs last, at half learning rate, inside the rollback guard.

Replay must precede downscaling (else you weaken what you were about to
consolidate) and pruning must follow both.

## Learning without much data

Five mechanisms, all of which are what a baby has and an adult network doesn't:

* **One-shot encoding** with pattern separation, so similar experiences on
  consecutive days stay distinguishable rather than blurring
* **Pattern completion** — two-stage recall (dense candidates, sparse
  verification) so a degraded fragment still retrieves the whole episode
* **Developmental annealing** — plasticity starts 3× and decays with lifetime
  experience. Useful on day one without being permanently volatile
* **Resonance growth** — a genuinely novel input allocates a unit on the spot
  rather than being forced into an existing category
* **Schema fast-track** — an association consistent with existing structure
  consolidates up to 2× faster (Tse et al. 2007). The tenth example in a
  familiar domain is cheap; the first in a new one is expensive. That
  asymmetry is the whole point, and a uniform learning rate gives you the
  opposite

```bash
python tests/tst_neuro.py    # CLS, STDP, adaptation, ACh switch, downscaling, arbitration
```

---

# Robot Face

Two-eye OLED-style face. No mouth, no brow sprites, no bitmaps — every
expression is two rounded rectangles and four levers: **shape** (w/h/radius),
**position** (offset + gaze), **lids** (top/bottom coverage), and **slant**
(lid angle).

Slant is what earns its keep. An angled top lid reads unmistakably as an
eyebrow, so one number replaces an entire second layer of art. It is
*anatomical*, not geometric: a single positive `lid_angle` means "inner
corners down" on both eyes, and the renderer works out that this is opposite
screen-space directions for left and right. Authors never mirror by hand.

```bash
python -m src.face list              # 27 builtins
python -m src.face preview thinking  # ANSI preview, works over SSH
python -m src.face gif happy         # render to happy.gif
python -m src.face export            # dump builtins as JSON files
python -m src.face edit              # author new ones (or: open tools/web/face_editor.html)
```

## The editor

```bash
python -m src.face edit              # editor + save-to-animations/ (opens your browser)
```

`tools/web/face_editor.html` is a single file with no dependencies. Opened with a double
click it still works (Save downloads the JSON); run through `python -m src.face
edit` it also reads and writes the `animations/` folder directly.

* **Every animation is editable**: all builtins plus your saved ones, in one
  list (`★ saved` = your own, `✎ edited` = overrides a builtin). Load one,
  change it, **Save**. The file lands in `animations/` and the robot picks it up
  on its next start. **Delete file** removes your copy and the builtin returns.
* **Everything a file can hold**: per-eye or both-eye values, per-corner radii,
  lids and slant, look/tilt/gap/scale/brightness, ten easings, priority,
  interruptible, suppress-auto-blink, hold-last-pose, loop, tags.
* **Editing tools**: drag keyframes on the track, duplicate, copy/paste a pose,
  unset a value (✕ - it then inherits from earlier keys), scale time, mirror
  left/right, undo/redo (Ctrl+Z / Ctrl+Y), Ctrl+S to save, an unsaved marker.
* **Check vs robot** sends the current animation to the Python interpolator and
  compares 49 poses with the editor's JS copy, so "looks right here" is checked,
  not assumed.

Files in `animations/` **override the builtin of the same name** at load time.
The server only listens on 127.0.0.1, refuses other sites and hosts, validates
every file before writing it, and writes atomically.

The editor reimplements the interpolator and renderer in JS so preview is
instant. That duplication is the one real maintenance hazard here: if the two
drift, animations look right in the editor and wrong on the robot. Both are
deliberately small, and the porting notes sit inline at each point where they
had to differ.

## Playback

Three things run simultaneously, and keeping them separate is what makes the
face feel alive rather than scripted:

1. **The active animation**, crossfaded (180ms) from whatever it replaced.
2. **The auto-blinker** — an independent layer multiplying `open`. Because
   it's a layer, every animation blinks correctly without its author thinking
   about it. Inter-blink intervals are log-normal (~2–8s, long tail); fixed
   intervals read as mechanical. `blink_suppress` turns it off for
   `thinking`, because concentration suppresses blinking in people too.
3. **Drive modulation** — a continuous bias from the homeostatic modulator.
   Fatigue lowers the lids, curiosity widens the eyes, caution narrows them.
   This is why the face still looks like it belongs to something when nothing
   is playing.

Scheduling is priority-based with an `interruptible` flag. Losing requests are
**dropped, not queued** — a face working through a backlog of stale reactions
is worse than one that misses a few.

## When each animation plays

| Trigger | Animation |
|---|---|
| Executive veto | `refuse` (head shake — the face says no before the sentence does) |
| Board deadlock | `confused` (ASH is about to ask a clarifying question) |
| Slow pathway begins | `thinking`, blink-suppressed |
| Slow pathway ends | `acknowledge`, or `confused` on tool failure |
| Addressed by voice | `listening` — wide, still, locked forward |
| Urgency ≥ 0.85 | `alert` |
| User away / returns | idle becomes `sleepy` / `idle` |
| Sleep phase | `sleeping` (slow brightness "breathing") |
| Fatigue / curiosity / caution extremes | `sleepy` / `curious` / `skeptical` |
| A question ASH asks (wondering, "any questions?", an unprompted "...?") | `curious` |
| A reminder, timer, alarm or watch going off | `alert` |
| A briefing, a joke | `happy` |
| A game: a win / higher-lower / the end | `very_happy` / `skeptical` / `happy` |
| A plain acknowledgement ("Opening outlook.", "Got it -- ...") | `acknowledge` |
| Quiet hours | idle becomes `sleepy`, and wakes after |

`FAST` and `REFLEX` pathways deliberately animate **nothing** — they finish in
milliseconds and animating them produces a flicker, not an expression. Mood
changes are held for a minimum of 6s and must clear a margin to switch, which
turns a jittery emotion signal into something that looks like temperament.

## Output

`pick_driver()` chooses by fidelity: OLED → TFT → window → terminal → null.
Animations are authored against 128×64 and scale to any panel.

* `OLEDDriver` — SSD1306/1309 over I2C
* `TFTDriver` — ST7789/ILI9341 over SPI (colour makes the brightness
  modulation and alert flash read much better)
* `WindowDriver` — Tkinter desktop preview
* `TerminalDriver` — ANSI half-blocks; the only preview that works on a
  headless robot over SSH
* `GifRecorder` — for documentation and debugging

```bash
python tests/tst_face.py    # interpolation, mirroring, blink layering, scheduling
python ashctl.py face happy    # trigger an animation on a running daemon
```

---

# Personality Specification

ASH personality is:

* Loyal
* Direct
* Efficient
* Slightly dry but not cold
* Emotionally aware
* Non-theatrical
* Not overly apologetic
* Never submissive
* Never dominant
* Stable

ASH does not:

* Flatter excessively
* Fabricate knowledge
* Overexplain simple tasks
* Engage in chaotic humor
* Break its constraints

---

# Design Constraints

ASH must:

* Run on desktop-class hardware (the brain layer targets desktop; the
  numpy-only forward model and critic remain Pi-viable if vision is disabled)
* Operate with minimal RAM footprint
* Avoid heavy vector DB usage
* Use deterministic routing
* Keep memory structured
* Remain stable under long uptime

---

# What ASH Is Not

ASH is not:

* A generic chatbot
* A cloud-dependent assistant
* A pure RAG system
* A retraining-at-runtime model
* An emotionally uncontrolled AI
* An LLM-driven decision agent
* A system where any one component can act unilaterally

ASH is a structured cognitive runtime.

---

# Current State Summary

ASH currently supports:

* Multimodal sensory extraction with salience and novelty scoring
* Latent forward model (JEPA-style) emitting a calibrated surprise signal
* Actor-critic fast reflexes with online, deferred-reward learning
* Weighted-quorum executive arbitration with capped weights and bounded vetoes
* Four-tier execution: reflex / fast / slow / escalated
* VLA channel with jurisdiction over embodied and irreversible actions
* Constitutional absolute veto sourced from hard CoreMemory rules
* Homeostatic drive modulation with fatigue accumulation and recovery
* Working memory plus depth-tiered retrieval over the three memory stores
* Sleep-phase replay consolidation and state persistence
* Full per-cycle introspection (`ash.explain_last()`, `ash.brain.status()`)
* 24h daemon with attention gating, ambient journaling and two-stage memory
* Pluggable sensors (window, idle, system, screen, mic, camera, serial)
* Pluggable actuators under VLA jurisdiction (notify, speak, launch, input,
  shell, clipboard, motor) -- dry-run and allowlisted by default
* Privacy controls: redaction, retention TTLs, capture blocklist, one-call pause
* Robot face: 27 premade animations, keyframe editor, layered auto-blink,
  drive-modulated baseline, OLED/TFT/window/terminal output
* Associative concept graph with three-factor Hebbian rewiring, resonance
  growth, sleep consolidation, frozen constraint edges and rollback guard
* Complementary learning systems: one-shot hippocampus + slow cortex, joined
  by interleaved replay during staged (NREM/REM) sleep
* Typed synapses with STDP, so the graph represents direction and not only
  correlation
* Neuromodulators (ACh/DA/NE/5-HT) gating which system learns and when
* Uncertainty-based board arbitration, developmental annealing, schema
  fast-track

Verify the architecture end-to-end with no heavy dependencies:

```bash
python tests/tst_brain.py     # architecture: pathways, arbitration, vetoes, learning
python tests/tst_daemon.py    # always-on: attention, privacy, journal, ambient loop
python tests/tst_face.py      # face: interpolation, mirroring, blink, scheduling
python tests/tst_concepts.py  # associative: PMI, Oja, scaling, pruning, rollback
python tests/tst_neuro.py     # CLS: one-shot memory, replay, STDP, neuromodulation
```

---

# Future Extensions

* Real ViT/CLIP checkpoint behind `VisionExtractor` (interface already there)
* Real JEPA or torch MLP behind `LatentForwardModel.predict/learn`
* Real VLA policy (RT-2 / OpenVLA style) behind `VLAChannel.propose`
* Hardware actuators via `VLAChannel.register_actuator()`
* Local speech-to-text as a third sensory extractor
* Entity relationship graphs
* Learned (rather than hand-set) board weights, still under the 0.34 cap

---

## ASH is:

> A two-speed cognitive runtime with layered memory, predictive world modeling,
> and homeostatic drive modulation, in which no single subsystem -- including
> the LLM -- holds unilateral authority over what it does.


