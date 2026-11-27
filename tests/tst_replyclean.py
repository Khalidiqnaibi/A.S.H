"""tst_replyclean.py -- what 26 Nov answers said that ASH should not: "Now, how can I assist you today?", "Also, if you need help with anything else,
feel free to ask. ASH is here to a...", "Let's make this an exciting journey together!", "Hey there, Immortal." opening a reply to someone who did not
greet; an answer cut for speech stopping at "Some of these include:"; "Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz" read out; "gta 6" said as
written; and "Okay -- dropped it." seven times for a never mind a second or two in. Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_replyclean.py"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import honest, machine, speech  # noqa: E402

check = tw.check


def main():
    print("\n=== R1. Closers ===")
    for before, after in (
            ("I see you're building me, that's fascinating! Let's make this an exciting journey together! Now, how can I assist you today?",
             "I see you're building me, that's fascinating!"),
            ("Try re-pairing them. Also, if you need help with anything else, feel free to ask. ASH is here to a", "Try re-pairing them."),
            ("Restart the buds. I'm always happy to help!", "Restart the buds."),
            ("It needs 16 GB. Don't hesitate to ask if anything is unclear.", "It needs 16 GB."),
            ("Let me know.", "Let me know.")):                       # never down to nothing
        check(f"{before[-45:]!r}", honest.sanitize(before) == after, repr(honest.sanitize(before)))
    check("an offer in the middle of real content stays", "feel free to rename" in honest.sanitize("You can feel free to rename it. It works."),
          honest.sanitize("You can feel free to rename it. It works."))

    print("\n=== R2. A greeting no one asked for ===")
    g = honest.strip_greeting("Hey there, Immortal. Sounds like you're having some trouble with your Live Buds.", "I'm having a trouble with my live buds.")
    check("dropped when the person did not greet", g == "Sounds like you're having some trouble with your Live Buds.", g)
    check("kept when they did", honest.strip_greeting("Hello! Good to see you.", "hello ash") == "Hello! Good to see you.")
    check("a greeting that is the whole answer stays", honest.strip_greeting("Hi there!", "what's up") == "Hi there!")
    check("'Hello world' code talk is not a greeting", honest.strip_greeting("Hello world is the first program.", "what is hello world")
          == "Hello world is the first program.")

    print("\n=== R3. A list cut for speech ===")
    s = speech.shorten("Some of these include:\n\n1. Complete Blood Count (CBC): measures cells\n2. **CEA**: a marker\n3. CA-125 - ovarian\n4. PSA: prostate")
    check("folded into the sentence, not stopped at the colon", s == "Some of these include: Complete Blood Count, CEA and CA-125, among others.", s)
    s = speech.fold_list("You need:\n- a cable\n- a charger")
    check("a short list in full", s == "You need: a cable and a charger.", s)
    check("no list: left alone", speech.fold_list("Note: this is fine.\nNext line.") == "Note: this is fine.\nNext line.")

    print("\n=== R4. Hardware and names said the way a person says them ===")
    m = machine.Machine(procs=lambda: [], mem=lambda: (50.0, 1, 2), own_pid=1,
                        spec=lambda: {"cpu": "Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz", "threads": 12, "ram": 16 * 1024 ** 3,
                                      "gpus": ["Intel(R) UHD Graphics", "NVIDIA GeForce GTX 1650 with Max-Q Design"]})
    say = m.can_run("gta 6")
    check("the CPU, memory and graphics card as said aloud", say.startswith("This computer has an Intel Core i5-10500H (12 threads), 16 GB of memory "
                                                                           "and a GTX 1650 (and Intel graphics)."), say)
    check("'gta 6' is GTA 6", "what GTA 6 needs" in say and "look up GTA 6 system requirements" in say, say)
    for w, want in (("elden ring", "elden ring"), ("nba 2k25", "NBA 2k25"), ("halo 3", "halo 3"), ("cyberpunk 2077", "cyberpunk 2077")):
        check(f"{w!r} -> {want!r}", machine.spoken_name(w) == want, machine.spoken_name(w))

    print("\n=== R5. A quick never mind ===")
    rt = tm.rig()
    rt._thinking, rt._cancelled, rt._think_started = True, False, time.time() - 2.0
    out = []
    sink = rt.response_sink
    rt.response_sink = lambda t, meta: (out.append(t), sink(t, meta))
    rt.cancel_thinking()
    check("two seconds in: 'Okay.'", out == ["Okay."], str(out))
    rt = tm.rig()
    rt._thinking, rt._cancelled, rt._think_started = True, False, time.time() - 40.0
    out = []
    rt.response_sink = lambda t, meta: out.append(t)
    rt.cancel_thinking()
    check("forty seconds in: 'Okay -- dropped it.'", out == ["Okay -- dropped it."], str(out))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
