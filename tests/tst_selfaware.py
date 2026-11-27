"""tst_selfaware.py -- 26 Nov lines about ASH itself: a bare "Ash." (went to the folder tool), "are you there?", "its you .. i'm building you"
(66 s on the model), answers to its questions said back with I/my turned round ("Got it -- Branch is im working on my first novel."), "that's you"
as an answer, and a long spoken answer read back as a question before it is kept. Fakes only.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_selfaware.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon.wondering import Environment, Knowledge, Wonderer, WonderingConfig, _yours  # noqa: E402

check = tw.check
T0 = 1_800_000_000.0


def wrig():
    kb = Knowledge()
    w = Wonderer(WonderingConfig.from_dict({"min_gap_s": 0.0}), kb, clock=lambda: T0)
    return w, kb


def ask(w, subject):
    w.raise_wonder("activity", subject, f"I keep seeing you in {subject}. What is it?", key=f"activity:{subject.lower()}",
                   data={"term": subject, "how": "activity", "links": ["named"]}, now=T0)
    q = w.next_question(Environment(idle_s=60.0, free_moment=True), T0)
    w.mark_asked(q.id, T0)
    return q


def main():
    print("\n=== S1. Said back to the person who said it ===")
    for said, back in (("im working on my first novel", "your first novel"), ("my personal weather app", "your personal weather app"),
                       ("just my notes for the MBU on your work", "your notes for the MBU on my work"),
                       ("the place I keep my drafts .. and other stuff", "the place you keep your drafts")):
        check(f"{said!r} -> {back!r}", _yours(said) == back, _yours(said))

    print("\n=== S2. Answers to 'what is it?' ===")
    w, kb = wrig()
    ask(w, "Branch")
    rep = w.hear("im working on my first novel", T0 + 5, addressed=True)
    check("the ack is turned round (26 Nov: 'Branch is im working on my first novel')", rep.say == "Got it -- Branch is your first novel.", rep.say)
    check("...and what is kept is the person's own words", "first novel" in kb.terms["branch"]["definition"], str(kb.terms))
    w, kb = wrig()
    ask(w, "Trello")
    rep = w.hear("I use it to track my uni tasks", T0 + 5, addressed=True)
    check("'I use it to ...' is said as a use", rep.say == "Got it -- you use Trello to track your uni tasks.", rep.say)
    for answer in ("A.S.H is you silly .. im just building you as i go", "it's you", "that's you, I'm building you", "you"):
        w, kb = wrig()
        ask(w, "A.S.H")
        rep = w.hear(answer, T0 + 5, addressed=True)
        check(f"{answer!r}: it is ASH itself", rep.say == "Oh -- that's me. Noted, and I won't ask about it again."
              and kb.terms["a s h"]["definition"].startswith("ASH itself"), rep.say)
    w, kb = wrig()
    ask(w, "Nimbus")
    rep = w.hear("it's your rival assistant from the other team", T0 + 5, addressed=True)
    check("'your ...' is not 'you'", "rival" in (rep.say or ""), rep.say)

    print("\n=== S3. A long spoken answer is read back first ===")
    w, kb = wrig()
    ask(w, "Custom TPU Prototyping")
    rep = w.hear("It's just my gym and I tab to ask about the MBU on your work.", T0 + 5, addressed=True, spoken=True)
    check("said back as a question, not kept yet", rep.consumed and rep.say.startswith("So Custom TPU Prototyping is your gym and you tab")
          and rep.say.endswith("-- right?") and not kb.term_known("Custom TPU Prototyping"), rep.say)
    rep = w.hear("no, it's my Gemini chat about the NPU", T0 + 10, addressed=True, spoken=True)
    check("'no, it's ...' keeps the correction", kb.term_known("Custom TPU Prototyping")
          and "Gemini" in kb.terms["custom tpu prototyping"]["definition"] and "your Gemini chat" in rep.say, f"{rep.say} {kb.terms}")
    w, kb = wrig()
    ask(w, "Custom TPU Prototyping")
    w.hear("It's just my gym and I tab to ask about the MBU on your work.", T0 + 5, addressed=True, spoken=True)
    rep = w.hear("yes", T0 + 10, addressed=True, spoken=True)
    check("a yes keeps it", kb.term_known("Custom TPU Prototyping") and rep.say.startswith("Great, noted"), rep.say)
    w, kb = wrig()
    ask(w, "Custom TPU Prototyping")
    rep = w.hear("It's my Gemini chat where I ask about the NPU on my laptop.", T0 + 5, addressed=True, spoken=False)
    check("typed: kept at once", kb.term_known("Custom TPU Prototyping") and rep.say.startswith("Got it"), rep.say)
    w, kb = wrig()
    ask(w, "Orbit")
    rep = w.hear("my side project", T0 + 5, addressed=True, spoken=True)
    check("a short spoken answer: kept at once", kb.term_known("Orbit"), rep.say)

    print("\n=== S4. In the runtime ===")
    for line in ("Ash.", "A.S.H.", "hey Ash", "Ash?"):
        rt = tm.rig()
        used, said = rt.say(line)
        check(f"{line!r} -> 'Yes?', no model", not used and said and said[-1] == "Yes?", f"{used} {said}")
    for line in ("Ash, are you there?", "you up?", "can you hear me"):
        rt = tm.rig()
        used, said = rt.say(line)
        check(f"{line!r} -> 'I'm here.'", not used and said and said[-1] == "I'm here.", f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("its you .. i'm building you as i go and use you to see what i need to improve")
    check("'I'm building you' is answered by rule (66 s on the model, 26 Nov)", not used and said and said[-1].startswith("I know -- I'm the code in A.S.H"),
          f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("I'm building a house")
    check("'building a house' is not about ASH", not (said and said[-1].startswith("I know -- I'm the code")), f"{used} {said}")
    rt = tm.rig()
    used, said = rt.say("What model are you?")
    check("'What model are you?' by rule", not used, f"{used} {said}")
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
