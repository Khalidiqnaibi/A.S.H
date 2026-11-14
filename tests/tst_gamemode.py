"""
tst_gamemode.py -- staying out of a game's way.

With League of Legends running, ASH took 100% of the CPU and RAM: the screen sensor reading frames, the microphone
transcribing the game's sound, a vision model on every capture, two gigabytes held. gamemode.py notices a game (a known
one, or any program that fills the screen) and steps aside -- idle priority, screen and mic paused, models released --
and comes back by itself.

  G1  What is a game: known names, the launcher that is not, the unknown full-screen program, the ones that are not
  G2  The state machine: delays in and out, only what it paused, no flapping, the hooks it calls
  G3  A game running behind a window in front
  G4  On the runtime: real sensors paused and resumed, availability says gaming, nothing said, status and `ashctl game`
  G5  The pieces: the vision model lets go and comes back, the window sensor reports full-screen, Ollama on this machine only

Run:  PYTHONIOENCODING=utf-8 python tests/tst_gamemode.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_effects as te  # noqa: E402
import tst_wondering as tw  # noqa: E402

check = tw.check
T0 = 1_800_000_000.0


def g1_what_is_a_game():
    print("\n=== G1. What is a game ===")
    from src.daemon.gamemode import looks_like_game
    from src.daemon.occasion import activity_of
    for proc, fs, want in [
        ("League of Legends.exe", False, "game"), ("LeagueClientUx.exe", False, "game"),
        ("VALORANT-Win64-Shipping.exe", False, "game"), ("cs2.exe", False, "game"), ("eldenring.exe", True, "game"),
        ("steam.exe", False, ""), ("EpicGamesLauncher.exe", False, ""), ("RiotClientServices.exe", False, ""),
        ("somegame.exe", True, "fullscreen"), ("somegame.exe", False, ""),
        ("chrome.exe", True, ""), ("msedge.exe", True, ""), ("stremio-shell-ng.exe", True, ""), ("vlc.exe", True, ""),
        ("Code.exe", True, ""), ("POWERPNT.EXE", True, ""), ("explorer.exe", True, ""), ("WindowsTerminal.exe", True, ""),
        ("", True, ""),
    ]:
        got = looks_like_game(proc, fs)
        check(f"{proc or '(none)'}{' fullscreen' if fs else ''} -> {want or 'not a game'}", got == want, got)
    check("fullscreen_counts off: an unknown program is not guessed", looks_like_game("somegame.exe", True, False) == "")
    check("League of Legends counts as gaming for who may be spoken to", activity_of("League of Legends.exe") == "gaming")
    check("...and so does a launcher", activity_of("steam.exe") == "gaming")
    from src.daemon.availability import judge
    check("availability: League of Legends in front is 'gaming', never spoken to",
          judge("League of Legends.exe", "", 30.0).state == "gaming" and not judge("League of Legends.exe", "", 30.0).can_talk)
    check("...and game_up says so for a game it does not know by name",
          judge("somegame.exe", "", 30.0, game_up=True).state == "gaming")
    check("...and a free user is still free", judge("chrome.exe", "", 30.0).state != "gaming")


class Sensor:
    def __init__(self, name, state="active"):
        self.name = name
        self.state = type("S", (), {"value": state})()
        self.log = []

    def disable(self):
        self.state = type("S", (), {"value": "disabled"})()
        self.log.append("disable")

    def enable(self):
        self.state = type("S", (), {"value": "active"})()
        self.log.append("enable")


class Per:
    def __init__(self, **states):
        self.sensors = {n: Sensor(n, s) for n, s in states.items()}


def mode(per=None, running=None, **cfg):
    from src.daemon.gamemode import GameMode, GameModeConfig
    calls = {"priority": [], "release": 0, "restore": 0, "unload": 0}
    gm = GameMode(GameModeConfig.from_dict(cfg), clock=lambda: T0, periphery=per or Per(screenshot="active", microphone="active"),
                  release=lambda: calls.__setitem__("release", calls["release"] + 1),
                  restore=lambda: calls.__setitem__("restore", calls["restore"] + 1),
                  scan=lambda: list(running or []),
                  own_priority=lambda low: calls["priority"].append(low),
                  unload_model=lambda: calls.__setitem__("unload", calls["unload"] + 1))
    gm.calls = calls
    return gm


def g2_state_machine():
    print("\n=== G2. The state machine ===")
    gm = mode()
    gm.look("League of Legends.exe", False, T0)
    check("a game just appeared: not yet (a launcher is not the game)", not gm.active)
    gm.look("League of Legends.exe", False, T0 + 5)
    check("...still not", not gm.active)
    gm.look("League of Legends.exe", False, T0 + 9)
    check("a game in front for 8 seconds: game mode", gm.active and gm.why == "game", str(gm.status()))
    check("it dropped its priority", gm.calls["priority"] == [True])
    check("it paused the screen and the mic", sorted(gm.paused) == ["microphone", "screenshot"], str(gm.paused))
    check("...really: both sensors were disabled", all(s.log == ["disable"] for s in gm.periphery.sensors.values()))
    check("it let go of the models it holds", gm.calls["release"] == 1)
    import time
    time.sleep(0.2)
    check("it asked the language model to unload", gm.calls["unload"] == 1, str(gm.calls))
    gm.look("League of Legends.exe", False, T0 + 30)
    check("staying in the game does it all only once", gm.calls["priority"] == [True] and gm.calls["release"] == 1)

    gm.look("Code.exe", False, T0 + 40)
    gm.look("Code.exe", False, T0 + 70)
    check("alt-tabbing for half a minute does not end it", gm.active)
    gm.look("League of Legends.exe", False, T0 + 75)
    gm.look("Code.exe", False, T0 + 80)
    gm.look("Code.exe", False, T0 + 100)
    check("...and coming back resets the wait", gm.active)
    gm.look("Code.exe", False, T0 + 130)
    check("45 seconds without a game: back to normal", not gm.active and gm.stats == {"entered": 1, "left": 1}, str(gm.status()))
    check("priority restored", gm.calls["priority"] == [True, False])
    check("the sensors it paused are on again", all(s.log == ["disable", "enable"] for s in gm.periphery.sensors.values()))
    check("nothing paused remains", gm.paused == [])
    time.sleep(0.2)
    check("the models are woken", gm.calls["restore"] == 1)

    # What the user turned off stays off.
    per = Per(screenshot="disabled", microphone="active")
    gm = mode(per)
    gm.look("cs2.exe", False, T0)
    gm.look("cs2.exe", False, T0 + 9)
    check("a sensor the user had already turned off is not touched", per.sensors["screenshot"].log == []
          and gm.paused == ["microphone"], str(gm.paused))
    gm.look("Code.exe", False, T0 + 20)
    gm.look("Code.exe", False, T0 + 70)
    check("...and is not turned on after", per.sensors["screenshot"].log == [] and per.sensors["screenshot"].state.value == "disabled")

    # The switches
    for off, kept in (("pause_mic", ["screenshot"]), ("pause_screen", ["microphone"])):
        g = mode(**{off: False})
        g.look("cs2.exe", False, T0)
        g.look("cs2.exe", False, T0 + 9)
        check(f"{off} off: only {kept[0]} is paused", g.paused == kept, str(g.paused))
    g = mode(lower_priority=False, free_model=False)
    g.look("cs2.exe", False, T0)
    g.look("cs2.exe", False, T0 + 9)
    check("lower_priority and free_model off: neither happens", g.calls["priority"] == [] and g.calls["unload"] == 0)
    g = mode(enabled=False)
    g.look("cs2.exe", False, T0)
    g.look("cs2.exe", False, T0 + 99)
    check("disabled: nothing, ever", not g.active and g.calls["priority"] == [])
    g = mode()
    g.look("somegame.exe", True, T0)
    g.look("somegame.exe", True, T0 + 9)
    check("an unknown full-screen program counts", g.active and g.why == "fullscreen")
    g = mode()
    g.look("stremio-shell-ng.exe", True, T0)
    g.look("stremio-shell-ng.exe", True, T0 + 99)
    check("watching a film full screen does not", not g.active)
    g = mode()
    g.look("steam.exe", False, T0)
    g.look("steam.exe", False, T0 + 99)
    check("browsing the Steam store does not", not g.active)

    from src.daemon.gamemode import GameMode, GameModeConfig
    bad = GameMode(GameModeConfig(), clock=lambda: T0, periphery=Per(screenshot="active"), release=lambda: 1 / 0,
                   restore=lambda: 1 / 0, scan=lambda: [], own_priority=lambda low: 1 / 0, unload_model=lambda: 1 / 0)
    bad.look("cs2.exe", False, T0)
    bad.look("cs2.exe", False, T0 + 9)
    check("...it is on anyway, and the sensors were still paused", bad.active and bad.paused == ["screenshot"])
    check("it can say what it is doing", "idle priority" in bad.report() and "screenshot" in bad.report(), bad.report())
    check("...and when it is not", "running normally" in mode().report())


def g3_running_behind():
    print("\n=== G3. A game running behind ===")
    gm = mode(running=["league of legends"])
    gm.look("Code.exe", False, T0)
    check("a known game running while another window is in front: seen after the first look", gm._running == ["league of legends"])
    gm.look("Code.exe", False, T0 + 9)
    check("...game mode (the match is on, you are just on the wiki)", gm.active and gm.why == "running", str(gm.status()))
    gm._running = []
    gm._scan = lambda: []
    gm.look("Code.exe", False, T0 + 30)
    gm.look("Code.exe", False, T0 + 80)
    check("the game closed: it ends", not gm.active)
    n = []
    g2 = mode()
    g2._scan = lambda: n.append(1) or []
    for i in range(10):
        g2.look("Code.exe", False, T0 + i)
    check("...it ran once in ten seconds", len(n) == 1, str(n))
    from src.daemon.gamemode import running_games
    check("the real scan answers without raising", isinstance(running_games(), list))


class AW:
    """The window sensor, as far as game mode reads it."""
    name = "active_window"

    def __init__(self):
        self.last_process, self.last_fullscreen = "Code.exe", False

    def due(self, now):
        return False

    def status(self):
        return {"name": "active_window"}


def g4_runtime():
    print("\n=== G4. On the runtime ===")
    rt = te.rig(game_mode=True, game_cfg={"enter_after_s": 8, "leave_after_s": 45, "lower_priority": False,
                                          "free_model": False})
    aw = AW()
    rt.periphery.sensors["active_window"] = aw
    rt.game._scan = lambda: []
    try:
        rt.say("turn on the mic", source="console")
        rt.scr.enable() if rt.scr.state.value == "disabled" else None
        check("the mic is on before the game", rt.mic.state.value == "active", rt.mic.state.value)
        for _ in range(3):
            rt.t[0] += 3
            rt._tick()
        check("with an editor in front: no game mode", not rt.game.active)
        aw.last_process = "League of Legends.exe"
        for _ in range(5):
            rt.t[0] += 3
            rt._tick()
        check("League of Legends in front for 15 seconds: game mode", rt.game.active, str(rt.game.status()))
        check("the real mic sensor is paused (its capture thread stopped)", rt.mic.state.value == "disabled"
              and not te.alive(rt.mic), rt.mic.state.value)
        check("availability says gaming, so nothing is said unprompted", rt._avail is not None and rt._avail.state in
              ("gaming", "away", "unknown") and not rt._avail.can_talk, str(rt._avail))
        check("the status shows it", rt.status()["game_mode"]["active"] is True, str(rt.status().get("game_mode")))
        from src.daemon.service import ControlServer
        cs = ControlServer.__new__(ControlServer)
        cs.runtime = rt
        got = cs.dispatch("game")
        check("`ashctl game` says so", got.get("ok") and "A game is up" in got.get("report", ""), str(got))
        aw.last_process = "Code.exe"
        for _ in range(20):
            rt.t[0] += 3
            rt._tick()
        check("the game is gone: back to normal", not rt.game.active)
        check("the mic is listening again", rt.mic.state.value == "active" and te.alive(rt.mic), rt.mic.state.value)

        rt2 = te.rig()
        check("a bare runtime has no game mode (tests and embedders are not surprised)", rt2.game is None)
        check("...and its status says so", rt2.status().get("game_mode") is None)
    finally:
        te.stop_all(rt)


def g5_pieces():
    print("\n=== G5. The pieces ===")
    from src.brain.sensory import VisionExtractor
    v = VisionExtractor.__new__(VisionExtractor)
    v.model_name, v.target_dim, v._proj = "x", 384, None
    v._backend, v._model, v._processor, v._resting = "vit", object(), object(), False
    loads = []
    v._try_load = lambda: loads.append(1)
    v.rest()
    check("the vision model lets go and the descriptor stands in", v._backend == "descriptor" and v._model is None
          and v._resting is True)
    v.rest()
    v.wake()
    v.wake()
    check("waking reloads it once", loads == [1], str(loads))
    import numpy as np
    v._backend = "descriptor"
    p = v.extract(np.zeros((32, 32, 3), dtype=np.float32))
    check("and it still sees, with the descriptor", p is not None and p.meta["backend"] == "descriptor")

    from src.daemon import gamemode
    seen = []
    real = gamemode.urllib.request.urlopen
    gamemode.urllib.request.urlopen = lambda req, timeout=0: (seen.append(req.full_url), type("R", (), {"read": lambda s: b""})())[1]
    try:
        gamemode.unload_ollama("mistral:latest", "http://127.0.0.1:4444/api/chat")
        gamemode.unload_ollama("mistral:latest", "http://example.com/api/chat")
    finally:
        gamemode.urllib.request.urlopen = real
    check("unloading asks Ollama on this machine, by keep_alive 0", seen == ["http://127.0.0.1:4444/api/generate"], str(seen))
    check("...and never anything off this machine", not any("example.com" in s for s in seen))

    from src.senses.screen import ActiveWindowSensor, ScreenshotSensor
    a = ActiveWindowSensor(enabled=True)

    def cur():
        a.last_process, a.last_fullscreen = "somegame.exe", True
        return "Some Game", "somegame.exe"
    a.current = cur
    ev = a.read()
    check("the window sensor reports full-screen in the event", ev and ev[0].meta.get("fullscreen") is True,
          str(ev[0].meta if ev else ev))
    s = ScreenshotSensor(enabled=False)
    s._reader = object()
    s.release()
    check("the screen sensor drops its OCR engines", s._reader is None and s._look_reader is None)

    from src.daemon.service import DEFAULT_CONFIG
    gmc = DEFAULT_CONFIG.get("game_mode", {})
    check("the default config turns it on, with every part on", gmc.get("enabled") is True
          and all(gmc.get(k) is True for k in ("lower_priority", "pause_screen", "pause_mic", "free_model")), str(gmc))
    from src.daemon import manual
    ids = {e[0]: e[2] for e in manual.ENTRIES}
    check("the manual in core memory has an entry for it", "game" in ids and "idle priority" in ids["game"], str(list(ids)))
    check("...which fits", len(ids["game"]) < 760, str(len(ids.get("game", ""))))


if __name__ == "__main__":
    g1_what_is_a_game()
    g2_state_machine()
    g3_running_behind()
    g4_runtime()
    g5_pieces()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
