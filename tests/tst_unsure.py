"""tst_unsure.py -- a line the recogniser was unsure of is never kept as an answer or a statement (26 Nov: a garbled line became a 'fact').
Run:  PYTHONIOENCODING=utf-8 python tests/tst_unsure.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.senses.base import Modality, SensorEvent  # noqa: E402

check = tw.check


def say(rt, text, conf=None):
    meta = {"asr_conf": conf} if conf is not None else {}
    n = len(rt.calls)
    rt.submit(SensorEvent(source="microphone", modality=Modality.TEXT, text=text, addressed=True, salience=0.9, urgency=0.6, meta=meta))
    rt._tick()
    return len(rt.calls) > n


def main():
    print("\n=== U1. An answer to ASH's own question ===")
    rt = tm.rig()
    w = rt.wonder
    out = []
    sink = rt.response_sink
    rt.response_sink = lambda t, m: (out.append(t), sink(t, m))
    q = w.raise_wonder("term", "Custom TPU Prototyping", "What's Custom TPU Prototyping?", "term:ctp", data={"term": "Custom TPU Prototyping"})
    w.mark_asked(q.id, rt.clock())
    say(rt, "It's just my gym and I tab to ask about the MBU on your work.", conf=-1.0)
    check("unsure: asked again, not kept", out and out[-1] == "I'm not sure I heard that right -- could you say it again?", str(out[-1:]))
    check("the question is still open", w.awaiting(rt.clock()) is not None)
    say(rt, "It's my Gemini tab where I ask about the NPU for you.", conf=-0.2)
    check("sure: taken as the answer", w.awaiting(rt.clock()) is None and out and "Gemini" in " ".join(out[-2:]), str(out[-2:]))

    print("\n=== U2. A statement ===")
    rt = tm.rig()
    out = []
    sink = rt.response_sink
    rt.response_sink = lambda t, m: (out.append(t), sink(t, m))
    n = len(out)
    say(rt, "I'm just building my custom hardware chip", conf=-1.1)
    check("unsure: not acknowledged as something they told me", not any(x.startswith(("Nice --", "Got it --", "Good to know --")) for x in out[n:]), str(out[n:]))
    n = len(out)
    say(rt, "I'm just building my custom hardware chip", conf=-0.1)
    check("sure: acknowledged as before", any(("hardware chip" in x) for x in out[n:]), str(out[n:]))
    n = len(out)
    say(rt, "I'm just building my custom hardware chip")
    check("typed (no confidence known): as before", any(("hardware chip" in x) for x in out[n:]), str(out[n:]))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
