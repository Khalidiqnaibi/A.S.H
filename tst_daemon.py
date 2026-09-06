"""
tst_daemon.py -- smoke test for the always-on layer.

Runs the ambient loop against fake sensors and a stub ASH, so the whole
observe/respond pipeline is exercised without a microphone, a model, or an
LLM. Verifies the behaviors that actually matter for a 24h process:

  * direct address always gets a response, even under a spent budget
  * routine ambient events are observed, not answered
  * an urgent event breaks through
  * the hourly unsolicited budget is a hard stop
  * refractory period suppresses back-to-back remarks
  * mute keeps observing but stops responding
  * privacy pause halts sensitive sensors within a tick
  * redaction happens before anything is journaled
  * journal rollup turns thousands of events into a handful of episodes
  * a throwing sensor is backed off and disabled, not fatal
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

from src.daemon import (  # noqa: E402
    AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy,
)
from src.daemon.attention import Disposition  # noqa: E402
from src.senses.base import (  # noqa: E402
    Capability, Modality, PeripheryRegistry, Sensor, SensorEvent, State,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class ScriptedSensor(Sensor):
    """Emits a pre-baked list of events, one batch per poll."""

    capability = Capability(name="scripted", description="test sensor")
    default_interval = 0.01

    def __init__(self, script, privacy_sensitive=False, name="scripted"):
        self.capability = Capability(name=name, description="test",
                                     privacy_sensitive=privacy_sensitive)
        self.script = list(script)
        super().__init__(enabled=True)

    def read(self):
        return [self.script.pop(0)] if self.script else []


class BrokenSensor(Sensor):
    capability = Capability(name="broken", description="always throws")
    default_interval = 0.01

    def read(self):
        raise RuntimeError("sensor on fire")


class StubDrives:
    urgency = 0.3
    caution = 0.3
    curiosity = 0.4
    effort_budget = 0.5
    social = 0.5
    fatigue = 0.0

    def as_dict(self):
        return {k: getattr(self, k) for k in
                ("urgency", "caution", "curiosity", "effort_budget", "social", "fatigue")}


class StubTrace:
    def __init__(self):
        self.percepts = {"novelty": 0.5}
        self.notes = []


class StubBrain:
    def __init__(self):
        self.homeostasis = type("H", (), {"drives": StubDrives()})()
        self.observed = []
        self.slept = 0

    def observe(self, text="", image=None, source="ambient", salience=0.3):
        self.observed.append((source, text))
        return StubTrace()

    def sleep(self, seconds_idle=0):
        self.slept += 1
        return {"ok": True}

    def status(self):
        return {"cycles": len(self.observed)}


class StubResponse:
    def __init__(self, text):
        self.text = text
        self.pathway = type("P", (), {"value": "slow"})()


class StubASH:
    def __init__(self):
        self.brain = StubBrain()
        self.thought = []
        self.episodic_memory = StubEpisodic()

    def think(self, query, image=None):
        self.thought.append(query)
        return StubResponse(f"noted: {query[:40]}")

    def explain_last(self):
        return {}


class StubEpisodic:
    def __init__(self):
        self.episodes = []

    def add_episode(self, summary, event_type, related_entities=None, importance=0.5):
        self.episodes.append(summary)


def ev(text, salience=0.3, urgency=0.0, addressed=False, source="scripted"):
    return SensorEvent(source=source, modality=Modality.TEXT, text=text,
                       salience=salience, urgency=urgency, addressed=addressed)


# ----------------------------------------------------------------------
def main():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
        ok = ok and cond

    # ---------------------------------------------------------------
    print("\n=== 1. Attention gate: routing ===")
    gate = AttentionGate(base_threshold=0.62, max_unsolicited_per_hour=3,
                         refractory_seconds=120)
    d = StubDrives()

    r = gate.decide(ev("hey ash what time is it", addressed=True), d)
    check("direct address -> RESPOND", r.disposition == Disposition.RESPOND, r.reasons[0])

    r = gate.decide(ev("switched to Chrome", salience=0.25), d)
    check("routine event -> OBSERVE", r.disposition == Disposition.OBSERVE,
          f"score={r.score:.2f} thr={r.threshold:.2f}")

    r = gate.decide(ev("battery critical at 4%", salience=0.9, urgency=0.95), d)
    check("urgent event -> RESPOND", r.disposition == Disposition.RESPOND,
          f"score={r.score:.2f}")

    r = gate.decide(ev("something mildly interesting", salience=0.8, urgency=0.7), d)
    check("refractory suppresses the next remark", r.disposition == Disposition.OBSERVE,
          "; ".join(r.reasons))

    print("\n=== 2. Hourly budget is a hard stop ===")
    g2 = AttentionGate(base_threshold=0.1, max_unsolicited_per_hour=2,
                       refractory_seconds=0.001)
    fired = 0
    for i in range(8):
        r = g2.decide(ev(f"interesting thing {i}", salience=0.9, urgency=0.6), d,
                      now=time.time() + i)
        if r.disposition == Disposition.RESPOND:
            fired += 1
    check("budget caps unsolicited remarks", fired <= 2, f"{fired} fired, cap 2")
    r = g2.decide(ev("hey ash", addressed=True), d)
    check("address bypasses a spent budget", r.disposition == Disposition.RESPOND)

    print("\n=== 3. Mute observes but does not respond ===")
    g3 = AttentionGate(base_threshold=0.1)
    g3.mute(True)
    r = g3.decide(ev("very interesting", salience=0.95, urgency=0.9), d)
    check("muted -> OBSERVE", r.disposition == Disposition.OBSERVE)
    r = g3.decide(ev("hey ash", addressed=True), d)
    check("muted still answers direct address", r.disposition == Disposition.RESPOND)

    print("\n=== 4. Privacy: redaction before persistence ===")
    priv = PrivacyPolicy()
    e = ev("my api_key=sk-abcdef0123456789abcdef and mail me at bob@example.com")
    priv.scrub_event(e)
    check("secrets redacted", "sk-abcdef" not in e.text and "bob@example.com" not in e.text,
          e.text[:70])

    print("\n=== 5. Privacy pause halts sensitive sensors ===")
    sensitive = ScriptedSensor([ev("saw something")], privacy_sensitive=True, name="cam")
    benign = ScriptedSensor([ev("cpu fine")], privacy_sensitive=False, name="sys")
    check("pause blocks sensitive", (priv.pause(), not priv.allows(sensitive))[1])
    check("pause allows benign", priv.allows(benign))
    priv.resume()
    check("resume restores sensitive", priv.allows(sensitive))

    # ---------------------------------------------------------------
    print("\n=== 6. Ambient loop end to end ===")
    tmp = tempfile.mkdtemp()
    ash = StubASH()
    periph = PeripheryRegistry()

    script = [ev(f"switched to app {i}", salience=0.2) for i in range(6)]
    script.append(ev("hey ash, what am I working on", salience=0.8, addressed=True))
    script.append(ev("disk 95% full", salience=0.7, urgency=0.9))
    periph.add_sensor(ScriptedSensor(script, name="scripted"))
    periph.add_sensor(BrokenSensor(enabled=True))

    delivered = []
    rt = AmbientRuntime(
        ash=ash, periphery=periph, journal=Journal(tmp),
        gate=AttentionGate(base_threshold=0.62, max_unsolicited_per_hour=5),
        privacy=PrivacyPolicy(),
        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=999999),
        response_sink=lambda t, m: delivered.append((t, m)),
    )
    rt.start()
    time.sleep(1.2)
    rt.stop()

    check("all events consumed", rt.stats["events"] >= 8, str(rt.stats))
    check("most events observed silently", rt.stats["observed"] >= 5,
          f"observed={rt.stats['observed']} responded={rt.stats['responded']}")
    check("responded to address + urgent only", rt.stats["responded"] == 2,
          f"responded={rt.stats['responded']}, delivered={len(delivered)}")
    check("addressed prompt reached the brain verbatim",
          any("what am I working on" in q for q in ash.thought))
    check("unprompted remark was framed as unprompted",
          any("UNPROMPTED OBSERVATION" in q for q in ash.thought))

    broken = periph.sensors["broken"]
    check("broken sensor backed off, not fatal",
          broken.error_count > 0 and broken.interval > 0.01,
          f"errors={broken.error_count} interval={broken.interval:.2f}s")

    print("\n=== 7. Journal rollup ===")
    j = Journal(tempfile.mkdtemp())
    base = time.time() - 3 * 3600
    for i in range(400):
        e = ev(f"switched to editor {i % 5}", salience=0.2)
        e.ts = base + i * 8            # one dense block
        e.meta = {"process": "code.exe"}
        j.append(e)
    for i in range(30):
        e = ev("discussed the brain package", salience=0.6, source="microphone")
        e.source = "microphone"
        e.ts = base + 5000 + i * 20    # second block after a gap
        j.append(e)
    j.flush()

    from datetime import datetime
    day = datetime.fromtimestamp(base).strftime("%Y-%m-%d")
    epi = StubEpisodic()
    rep = j.rollup(day=day, episodic=epi)
    check("430 raw events -> few episodes",
          0 < rep.get("blocks", 0) <= 4 and len(epi.episodes) == rep["blocks"],
          f"{rep.get('events')} events -> {rep.get('blocks')} blocks")
    check("rollup is idempotent", j.rollup(day=day, episodic=epi).get("skipped") is not None)
    if epi.episodes:
        print(f"     e.g. {epi.episodes[0][:110]}")

    print("\n=== 8. Context block for prompts ===")
    ctx = j.context_block(seconds=99999, limit=5)
    check("context block renders", bool(ctx) and "\n" in ctx, f"{len(ctx)} chars")

    print("\n=== Attention counters ===")
    print("    ", rt.gate.status()["counts"])

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
