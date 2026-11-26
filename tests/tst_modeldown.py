"""tst_modeldown.py -- when the model times out on a question: it is SAID (with why), and a squeezed machine waits 90 s, not 300.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_modeldown.py"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon.ambient import AmbientRuntime, NO_MODEL_TEXT  # noqa: E402

check = tw.check


def main():
    print("\n=== M1. Why, and how long it waits ===")
    rt = types.SimpleNamespace(governor=types.SimpleNamespace(status=lambda: {"squeezed": False, "memory": 92.0, "cpu": 10.0, "why": ""}))
    check("high memory is named", AmbientRuntime._why_no_model(rt) == " Memory is at 92%, which is why it is slow.", AmbientRuntime._why_no_model(rt))
    rt.governor.status = lambda: {"squeezed": True, "memory": 92.0, "cpu": 10.0, "why": "memory is at 92%"}
    check("squeezed: the reason the governor gave", "under load (memory is at 92%)" in AmbientRuntime._why_no_model(rt))
    rt.governor.status = lambda: {"squeezed": False, "memory": 50.0, "cpu": 10.0, "why": ""}
    check("a calm machine: no excuse", AmbientRuntime._why_no_model(rt) == "")
    check("no governor: no excuse", AmbientRuntime._why_no_model(types.SimpleNamespace(governor=None)) == "")

    llm = types.SimpleNamespace(_timeout=300)
    box = types.SimpleNamespace(ash=types.SimpleNamespace(llm=llm))
    AmbientRuntime._llm_timeout_cap(box, True)
    check("squeezed: the model gets 90 s, not 300", llm._timeout == 90.0, str(llm._timeout))
    AmbientRuntime._llm_timeout_cap(box, True)
    AmbientRuntime._llm_timeout_cap(box, False)
    check("relaxed: back to what it was (also after a second squeeze)", llm._timeout == 300, str(llm._timeout))
    llm2 = types.SimpleNamespace(_timeout=45)
    box2 = types.SimpleNamespace(ash=types.SimpleNamespace(llm=llm2))
    AmbientRuntime._llm_timeout_cap(box2, True)
    check("a limit already shorter is left", llm2._timeout == 45)
    AmbientRuntime._llm_timeout_cap(types.SimpleNamespace(ash=None), True)

    print("\n=== M2. In the runtime: the reply, and that it is spoken ===")
    rt = tm.rig()
    sent = []
    rt.response_sink = lambda t, m: sent.append((t, m))
    rt.governor = types.SimpleNamespace(status=lambda: {"squeezed": False, "memory": 92.0, "cpu": 5.0, "why": ""}, tick=lambda now: None)
    ev = types.SimpleNamespace(addressed=True, source="console", text="can my pc run gta 6")
    dec = types.SimpleNamespace(likely=True, as_dict=lambda: {})
    rt._model_unavailable(ev, dec)
    t, m = sent[-1]
    check("said with the reason", t == "My brain's not answering right now. Memory is at 92%, which is why it is slow. Give it a minute and ask me again.", t)
    check("and marked to be spoken, since it was asked", m.get("degraded") and m.get("speak_degraded") is True, str(m))
    rt.governor = None
    rt._last_no_reply_notice = -1e9
    rt._model_unavailable(ev, dec)
    check("with no reason to give: the old line", sent[-1][0] == NO_MODEL_TEXT, sent[-1][0])

    import ashd
    spoken = []
    Act = lambda sink: (lambda x: (sink.append(x), {"ok": True})[1])  # noqa: E731

    class A:
        def __init__(self, f):
            self.state = types.SimpleNamespace(value="active")
            self.f = f

        def __call__(self, x):
            return self.f(x)

    class P:
        actuators = {"speak": A(lambda x: (spoken.append(x), {"ok": True})[1])}
    sink = ashd.build_response_sink(P(), prefer_speech=False)
    sink("owed", {"addressed": True, "degraded": True, "speak_degraded": True})
    sink("an apology nobody asked for", {"addressed": False, "degraded": True, "speak_degraded": True})
    sink("old style", {"addressed": True, "degraded": True})
    check("an owed answer is spoken; an unprompted apology and the old style are not", spoken == ["owed"], str(spoken))
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
