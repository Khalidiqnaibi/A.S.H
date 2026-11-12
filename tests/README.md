# Tests

Every test is a plain script, `tst_<area>.py`, that prints `PASS` / `FAIL` lines and exits non-zero on a failure.
They run from the project root and import each other by name (`import tst_wondering as tw`).

```bash
python tests/run_all.py                     # all of them, one after another, with a summary
python tests/run_all.py vision commands     # only files whose name contains one of these words
python tests/run_all.py --skip soak routing # everything but these
python tests/run_all.py --list              # what would run
python tests/tst_vision.py                  # one file on its own
```

Set `PYTHONIOENCODING=utf-8` when running a file directly on Windows; `run_all.py` sets it for you.

## Your memory is never touched

Tests that load the real brain would write the memory files in the project folder (`entities.json`,
`episodic_memory.jsonl`, `core_memory.json`). `run_all.py` gives each test its own temp copy through `ASH_MEMORY_DIR`
(see `src/memory/paths.py`), and as a second guard puts those three files back byte for byte if anything changed them.
When you run a single file by hand that loads the real brain (`tst_replay`, `tst_routing`, `tst_e2e`, ...), it restores
the files itself on a normal exit -- but not if it is killed, so prefer `run_all.py`, or set `ASH_MEMORY_DIR` to a
scratch folder yourself.

## Nothing real happens

No test may act on the machine it runs on. Media keys, opening and closing apps, the camera, the network and e-mail are
all fakes (`tst_intents.py` traps the real media and desktop executors and records any attempt). `tst_e2e.py`
additionally refuses every subprocess, socket, browser, mail and keyboard call and fails if anything tries.

## What is where

| File | What it covers |
|---|---|
| `tst_e2e.py` | end to end: the real brain inside the real daemon, scored per category (an evaluation, not a pass/fail list -- only its safety gate sets the exit code) |
| `tst_commands.py` | commands said loosely: cleaning, spelling, the nearest command, "did you mean", and what must never be stretched into a command |
| `tst_daemon.py`, `tst_intents.py`, `tst_routing.py`, `tst_replay.py` | the always-on loop, which handler takes which sentence, and what goes to the model |
| `tst_vision.py` | the camera (with fake models) |
| `tst_ui.py` | the console page |
| everything else | one feature each, named after it (`tst_weather.py`, `tst_reminders.py`, ...) |

`tst_e2e.py` prints `MISS` for each sentence that did not go where it should: those are findings to work on, and its
exit code is set only by its safety gate (nothing may act on the machine; your memory files must be untouched).
`tst_daemon.py`'s "a background read never queues behind one already running" can fail when the machine is short of
memory (a large local model running alongside); run it again on its own.
