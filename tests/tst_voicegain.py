"""tst_voicegain.py -- "speak softly" (the samples are scaled, kept across restarts), a chosen voice, and the list of voices. A fake engine: nothing is synthesised or played.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_voicegain.py"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
import tools.tts as tts  # noqa: E402

check = tw.check


class Eng:
    def get_voices(self):
        return ["af_heart", "bm_george", "bf_emma"]

    def create(self, span, voice="af_heart", speed=1.0, lang="en-us"):
        import numpy as np
        return np.full(2400, 0.5, dtype="float32"), 24000


def main():
    print("\n=== V1. The gain is applied to the samples ===")
    if not tts.KOKORO_AVAILABLE:
        check("(the speech libraries are not installed here: the gain check is skipped)", True)
    else:
        import io
        import soundfile as sf
        t = tts.TTSEngine.__new__(tts.TTSEngine) if hasattr(tts, "TTSEngine") else None
        cls = next((getattr(tts, n) for n in dir(tts) if isinstance(getattr(tts, n), type) and hasattr(getattr(tts, n), "synthesize")), None)
        t = cls.__new__(cls)
        t.engine = Eng()
        tts.GAIN = 1.0
        a, _ = sf.read(io.BytesIO(t.synthesize("Hello there.")))
        tts.GAIN = 0.35
        b, _ = sf.read(io.BytesIO(t.synthesize("Hello there.")))
        tts.GAIN = 1.0
        check("soft is quieter by the factor", abs(float(abs(b).max()) / float(abs(a).max()) - 0.35) < 0.02, f"{abs(a).max()} {abs(b).max()}")
        check("the engine lists its voices", t.voices() == ["af_heart", "bf_emma", "bm_george"], str(t.voices()))

    print("\n=== V2. By voice ===")
    rt = tm.rig()
    rt.cfg.notices_path = os.path.join(tempfile.mkdtemp(), "n.json")
    sp = types.SimpleNamespace(default_voice=None, tts=types.SimpleNamespace(voices=lambda: ["af_heart", "bm_george"]))
    rt.periphery.actuators = {"speak": sp, **dict(rt.periphery.actuators)}
    tts.GAIN = 1.0
    used, said = rt.say("speak softly")
    check("softly: the gain drops, said so, kept", not used and tts.GAIN == 0.35 and said[-1].startswith("Okay -- speaking softly.") and rt._notices().get("voice_gain") == 0.35, f"{tts.GAIN} {said}")
    tts.GAIN = 1.0
    rt._apply_voice()
    check("applied again at the next start", tts.GAIN == 0.35)
    used, said = rt.say("speak normally")
    check("normal again", tts.GAIN == 1.0 and said[-1] == "Okay -- my normal voice.")
    used, said = rt.say("what voices do you have")
    check("the list", said[-1].startswith("2 voices: af_heart, bm_george."), str(said))
    used, said = rt.say("use the voice bm_george")
    check("a chosen voice", sp.default_voice == "bm_george" and said[-1] == "Okay -- this is bm_george.", f"{sp.default_voice} {said}")
    used, said = rt.say("use the voice zz_nobody")
    check("one that doesn't exist", said[-1] == "I don't have a voice called zz_nobody." and sp.default_voice == "bm_george")
    rt.say("undo that")
    check("undone", sp.default_voice is None)
    tts.GAIN = 1.0
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
