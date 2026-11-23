"""
tst_governor.py -- stepping aside when the machine is struggling, not only for a game.

Memory sat at 95% and one answer took 60-73 s while ASH held a language model, a speech model, an OCR engine and an encoder in RAM.
governor.py watches memory and CPU and asks game mode's machinery (lower priority, screen reading paused, models let go) to step aside, with
the microphone left on, and to come back after a few calm minutes.

  H1  The governor: how long, how high, hysteresis, no flapping
  H2  What squeeze() and relax() do on the game-mode machinery, and how they live with a game
  H3  On the runtime: wired from the config, in the status

Run:  PYTHONIOENCODING=utf-8 python tests/tst_governor.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_gamemode as tg  # noqa: E402
import tst_wondering as tw  # noqa: E402
from src.daemon.governor import Governor, GovernorConfig  # noqa: E402

check = tw.check
T0 = tg.T0


def gov(read, **cfg):
    gm = tg.mode()
    t = [T0]
    g = Governor(gm, GovernorConfig.from_dict(cfg), clock=lambda: t[0], probe=lambda: read[0])
    g.t = t
    return g, gm


def run(g, seconds, step=15.0):
    out = []
    end = g.t[0] + seconds
    while g.t[0] < end:
        g.t[0] += step
        r = g.tick(g.t[0])
        if r:
            out.append((round(g.t[0] - T0), r))
    return out


def h1_governor():
    print("\n=== H1. The governor ===")
    read = [(60.0, 20.0)]
    g, gm = gov(read)
    check("a calm machine: nothing happens", run(g, 600) == [] and not gm.squeezed)
    read[0] = (95.0, 30.0)
    ev = run(g, 100)
    check("memory high for less than two minutes: not yet", ev == [] and not gm.squeezed, str(ev))
    ev = run(g, 60)
    check("memory high for two minutes: it steps aside", gm.squeezed and ev and ev[0][1] == "squeezed", str(ev))
    check("...saying why", "memory at 95%" in gm.squeeze_why, gm.squeeze_why)
    read[0] = (80.0, 30.0)
    check("between the marks it stays aside (hysteresis)", run(g, 600) == [] and gm.squeezed)
    read[0] = (60.0, 20.0)
    ev = run(g, 120)
    check("calm for less than three minutes: not yet", not ev and gm.squeezed)
    ev = run(g, 120)
    check("calm for three minutes: back", not gm.squeezed and ev and ev[0][1] == "relaxed", str(ev))
    # a spike that comes and goes does not count
    read[0] = (95.0, 30.0)
    run(g, 90)
    read[0] = (60.0, 30.0)
    run(g, 30)
    read[0] = (95.0, 30.0)
    check("a spike that came and went resets the wait", run(g, 100) == [] and not gm.squeezed)
    # the CPU: pinned for a long while only
    g2, gm2 = gov([(50.0, 99.0)])
    check("the CPU pinned for 4 minutes: aside", run(g2, 200) == [] and run(g2, 60) and gm2.squeezed and "CPU" in gm2.squeeze_why)
    off, gmo = gov([(99.0, 99.0)], enabled=False)
    check("switched off it does nothing", run(off, 900) == [] and not gmo.squeezed)
    unread, gmu = gov([(None, None)])
    check("a machine it cannot read is left alone", run(unread, 900) == [] and not gmu.squeezed)
    check("the report says what it did", "keeping out of the way" in g.report() or "running normally" in g.report())


def h2_machinery():
    print("\n=== H2. squeeze() and relax() ===")
    gm = tg.mode()
    check("squeeze: it does", gm.squeeze("memory at 95%", T0))
    time.sleep(0.2)
    check("it dropped its priority, let go of the vision model, asked Ollama to unload", gm.calls["priority"] == [True] and gm.calls["release"] == 1
          and gm.calls["unload"] == 1, str(gm.calls))
    check("it paused the screen and NOT the microphone", gm.paused == ["screenshot"]
          and gm.periphery.sensors["microphone"].log == [] and gm.periphery.sensors["screenshot"].log == ["disable"], str(gm.paused))
    check("squeezing again does nothing more", gm.squeeze("again", T0) is False and gm.calls["priority"] == [True])
    check("relax: everything back", gm.relax(T0 + 300) and gm.calls["priority"] == [True, False]
          and gm.periphery.sensors["screenshot"].log == ["disable", "enable"] and not gm.squeezed)
    time.sleep(0.2)
    check("...the models are woken", gm.calls["restore"] == 1)
    check("relax with nothing to relax is nothing", gm.relax(T0 + 400) is False)
    # with a game
    gm = tg.mode()
    gm.look("League of Legends.exe", False, T0)
    gm.look("League of Legends.exe", False, T0 + 9)
    check("(a game is up)", gm.active)
    pri = list(gm.calls["priority"])
    check("squeeze during a game marks it, and does no second round of everything", gm.squeeze("memory", T0 + 20) and gm.calls["priority"] == pri)
    gm.look("Code.exe", False, T0 + 30)
    gm.look("Code.exe", False, T0 + 80)
    check("the game ends: the mic comes back, the machine is still struggling so ASH stays aside and keeps the screen paused",
          not gm.active and gm.squeezed and gm.periphery.sensors["microphone"].log[-1] == "enable"
          and gm.periphery.sensors["screenshot"].state.value == "disabled" and gm.calls["priority"] == pri, str((gm.periphery.sensors["microphone"].log, gm.calls)))
    check("and when the machine calms, then it all comes back", gm.relax(T0 + 400) and gm.calls["priority"][-1] is False
          and gm.periphery.sensors["screenshot"].state.value == "active")


def h3_runtime():
    print("\n=== H3. On the runtime ===")
    import tst_effects as te
    rt = te.rig(game_mode=True, governor=True, governor_cfg={"memory_for_s": 30, "sample_s": 1})
    check("the governor is built with game mode", rt.governor is not None and rt.game is not None)
    check("it is in the status", rt.status().get("governor", {}).get("enabled") is True, str(rt.status().get("governor")))
    off = te.rig(game_mode=True)
    check("off unless asked for (a bare runtime)", off.governor is None)
    nogame = te.rig(governor=True)
    check("and not without game mode (nothing to step aside with)", nogame.governor is None)
    rt.governor.probe = lambda: (95.0, 10.0)
    t = [T0]
    rt.governor.clock = lambda: t[0]
    for _ in range(5):
        t[0] += 10
        rt.governor.tick(t[0])
    check("a hot machine, for long enough, puts the runtime aside", rt.game.squeezed and rt.status()["governor"]["squeezed"], str(rt.status()["governor"]))


def h4_idle_unload():
    print("\n=== H4. The model is let go of after a long idle ===")
    import tst_effects as te
    rt = te.rig(unload_idle_s=600.0)
    calls = []
    rt._unload_model = lambda: calls.append(1)
    t0 = rt.clock()
    rt._unload_when_idle(t0 + 9999)
    time.sleep(0.2)
    check("before any question there is nothing to unload", calls == [])
    rt.say("what should I cook tonight")                 # a question to the model
    last = rt._last_model_at
    rt._unload_when_idle(last + 300)
    time.sleep(0.2)
    check("five minutes on: still loaded", calls == [])
    rt._unload_when_idle(last + 700)
    time.sleep(0.2)
    check("ten minutes without a question: Ollama is asked to unload", calls == [1], str(calls))
    rt._unload_when_idle(last + 900)
    time.sleep(0.2)
    check("once, not every tick", calls == [1])
    rt.say("and what about dessert")
    rt._unload_when_idle(rt._last_model_at + 700)
    time.sleep(0.2)
    check("after the next question it can be unloaded again", calls == [1, 1], str(calls))
    off = te.rig()
    c2 = []
    off._unload_model = lambda: c2.append(1)
    off.say("what should I cook tonight")
    off._unload_when_idle(off._last_model_at + 10 ** 6)
    time.sleep(0.2)
    check("switched off (0), never", c2 == [])


if __name__ == "__main__":
    h4_idle_unload()
    h1_governor()
    h2_machinery()
    h3_runtime()
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)
