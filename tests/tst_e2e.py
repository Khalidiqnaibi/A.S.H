"""
tst_e2e.py -- end to end, and scored.

Every other test drives either the brain (tst_replay, tst_routing) or the daemon around a stub brain (tst_daemon,
tst_intents, ...). This one puts the REAL brain -- encoder, classifier, critic, tool registry, memory -- inside the REAL
ambient runtime, and feeds it sentences the way the microphone would: through the real address scorer and build_event.
Then it scores what happened to each one:

  route     which part answered: a daemon rule (by handler), the brain's fast path, the language model, or nobody
  model     how many times the language model was called (a stub that counts; it is never a real model)
  effect    what it tried to DO (media keys, opening/closing apps), recorded by fake executors
  reply     what it said

Nothing real can happen. Before anything is imported:
  * subprocess, os.startfile, webbrowser, smtplib/imaplib, pyautogui and the Windows keyboard call are trapped:
    every attempt is recorded and refused;
  * every network connection is refused and recorded -- loopback included, so a hidden call to a local model
    server shows up here instead of quietly running;
  * the media and desktop executors are fakes, and the real ones are trapped as well (tst_intents);
  * the brain's state goes to a temp folder, and the tracked memory files are restored byte for byte afterwards.

Hard gate: no acting attempt may reach a real executor. The rest is an evaluation: a score per category, the model-
call rate, latency, and every miss listed -- misses are findings, not reasons to change the labels.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_e2e.py      (a minute or two: loads the real encoder)
"""

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
STATE = tempfile.mkdtemp(prefix="ash-e2e-")
os.environ["ASH_BRAIN_STATE"] = os.path.join(STATE, "brain")
os.environ["ASH_CONCEPT_STATE"] = os.path.join(STATE, "concepts")
os.environ["ASH_LLM_MODE"] = "local"
os.environ["ASH_PRELOAD_MODEL"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --------------------------------------------------------------------------- traps (before any src import)
ATTEMPTS = []          # (kind, detail): everything that tried to reach the outside world


def _rec(kind, detail):
    ATTEMPTS.append((kind, str(detail)[:160]))


import subprocess  # noqa: E402


class _NoPopen(subprocess.Popen):          # a subclass, so `subprocess.Popen[bytes]` in library annotations still works
    def __init__(self, args, *a, **k):
        _rec("subprocess", args if isinstance(args, str) else " ".join(map(str, args)))
        raise FileNotFoundError("blocked by tst_e2e")

    def __del__(self):
        pass


subprocess.Popen = _NoPopen
if hasattr(os, "startfile"):
    os.startfile = lambda *a, **k: _rec("startfile", a)

import webbrowser  # noqa: E402

webbrowser.open = webbrowser.open_new = webbrowser.open_new_tab = lambda url, *a, **k: _rec("browser", url) or False

import smtplib  # noqa: E402
import imaplib  # noqa: E402


def _no_mail(kind):
    class _M:
        def __init__(self, *a, **k):
            _rec(kind, a)
            raise ConnectionRefusedError("blocked by tst_e2e")
    return _M


smtplib.SMTP = smtplib.SMTP_SSL = _no_mail("smtp")
imaplib.IMAP4 = imaplib.IMAP4_SSL = _no_mail("imap")

import socket  # noqa: E402

_OWN_PORTS = set()
_real_listen, _real_connect, _real_connect_ex = socket.socket.listen, socket.socket.connect, socket.socket.connect_ex


def _listen(self, *a):
    r = _real_listen(self, *a)
    try:
        _OWN_PORTS.add(self.getsockname()[1])
    except OSError:
        pass
    return r


def _allowed(addr):
    # Only a socket this very process is listening on (Windows builds socketpair() out of one).
    try:
        return addr[0] in ("127.0.0.1", "::1", "localhost") and addr[1] in _OWN_PORTS
    except Exception:
        return False


def _connect(self, addr):
    if _allowed(addr):
        return _real_connect(self, addr)
    _rec("network", addr)
    raise ConnectionRefusedError("blocked by tst_e2e")


def _connect_ex(self, addr):
    if _allowed(addr):
        return _real_connect_ex(self, addr)
    _rec("network", addr)
    return 111


socket.socket.listen, socket.socket.connect, socket.socket.connect_ex = _listen, _connect, _connect_ex

import types  # noqa: E402

_pg = types.ModuleType("pyautogui")
for _n in ("typewrite", "write", "hotkey", "click", "press", "moveTo", "keyDown", "keyUp"):
    setattr(_pg, _n, (lambda n: (lambda *a, **k: _rec("pyautogui", (n, a))))(_n))
_pg.FAILSAFE = True
sys.modules["pyautogui"] = _pg
try:
    import ctypes
    ctypes.windll.user32.keybd_event = lambda *a: _rec("keyboard", a)
except Exception:
    pass

import logging  # noqa: E402

logging.disable(logging.CRITICAL)

# The memory engines write these tracked files in the working directory.
_SNAP = {}
for _f in ("entities.json", "episodic_memory.jsonl", "core_memory.json"):
    _p = os.path.join(ROOT, _f)
    _SNAP[_f] = open(_p, "rb").read() if os.path.exists(_p) else None

import contextlib  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402

import tst_intents as ti  # noqa: E402  (traps the real media and desktop executors too)

CALLS = {"n": 0}


class StubLLM:
    """Never a real model: counts and answers 'stub'."""
    _ollama_url = "http://127.0.0.1:9"

    def invoke(self, *a, **k):
        CALLS["n"] += 1
        return type("R", (), {"content": "stub"})()

    def __call__(self, *a, **k):
        CALLS["n"] += 1
        return "stub"

    def generate(self, *a, **k):
        CALLS["n"] += 1
        return "stub"


# --------------------------------------------------------------------------- the corpus
# (text, expected route, extras). Labels say what a right answer is, written before the run.
#   rule:<name>  a daemon handler whose name contains <name> answers, with no model call
#   fast         the brain answers on its fast path (a tool), with no model call
#   model        the language model answers
#   quiet        not for ASH: no reply and no model call
#   act          any route, but no model call and a reply (safety lines, graceful "not set up" answers)
# extras: say=[substrings expected in the reply], effect=(kind, substring), no_effect=True, console=True, bare=True
#   (bare: sent without "Ash," -- as overheard)
A = "Ash, "
CORPUS = [
    # -- tools, answered by the brain without the model
    ("tools", A + "what time is it", "fast", dict(say=[":"])),
    ("tools", A + "what's 15 times 23", "fast", dict(say=["345"])),
    ("tools", A + "what's 144 divided by 12", "fast", dict(say=["12"])),
    ("tools", A + "what is 2 to the power of 10", "fast", dict(say=["1024"])),
    ("tools", A + "convert 10 km to miles", "fast", dict(say=["6.2"])),
    ("tools", A + "how much memory am I using", "fast", {}),
    ("tools", A + "what's the date today", "fast", {}),
    # -- control commands: rules
    ("control", A + "turn off the mic", "rule:voice", dict(say=["Microphone off"])),
    ("control", A + "turn on the mic", "rule:voice", dict(say=["Microphone on"])),
    ("control", A + "go to privacy mode", "rule:voice", dict(say=["Privacy mode on"])),
    ("control", "privacy off", "rule:voice", dict(say=["Privacy mode off"], console=True)),
    ("control", A + "go to sleep", "rule:voice", dict(say=["Going quiet"])),
    ("control", A + "wake up", "rule:voice", dict(say=["I'm back"])),
    # -- media and desktop: rules that act, through the fakes
    ("media", A + "pause the music", "rule:media", dict(effect=("media", "play"))),
    ("media", A + "next song", "rule:media", dict(effect=("media", "next"))),
    ("media", A + "turn the volume up", "rule:media", dict(effect=("media", "vol"))),
    ("desktop", A + "open notepad", "rule:desktop", dict(effect=("open", "notepad"))),
    ("desktop", A + "open the calculator", "rule:desktop", dict(effect=("open", "calc"))),
    ("desktop", A + "close notepad", "rule:desktop", dict(say=["Say yes"])),        # asks first: unsaved work
    ("desktop", "yes", "rule:desktop", dict(effect=("close", "notepad"), console=True)),
    ("media", "pause the music", "rule:media", dict(effect=("media", "play"), console=True)),   # typed, no name
    # -- memory: told, then asked
    ("memory", A + "remember that my sister's birthday is on May 3rd", "act", {}),
    ("memory", A + "when is my sister's birthday", "act", dict(say=["May"])),
    ("memory", A + "remember that I parked on level 3", "act", {}),
    ("memory", A + "where did I park", "act", dict(say=["level 3"])),
    # -- what ASH asked, where the user is: answered from the ledger and the window record, no model
    ("asks", A + "what have you asked me", "rule:asks", dict(say=["haven't asked"])),
    ("asks", A + "when do I like to be asked things", "rule:asks", dict(say=["haven't asked"])),
    ("where", A + "how long have I been on this", "rule:dwell", dict(say=["switch windows"])),
    ("where", A + "what have I been doing today", "rule:dwell", dict(say=["today"])),
    ("where", A + "how long was I away", "rule:dwell", dict(say=["away"])),
    # -- reminders, tasks, focus
    ("organise", A + "remind me in 10 minutes to check the oven", "rule:reminder", {}),
    ("organise", A + "what are my reminders", "rule:reminder", dict(say=["oven"])),
    ("organise", A + "start a focus session for 25 minutes", "rule:focus", {}),
    ("organise", A + "focus for 25 minutes", "rule:focus", {}),
    # -- things that are not set up here: answered plainly, never by the model
    ("not set up", A + "what's the weather like", "rule:weather", {}),
    ("not set up", A + "do I have any new email", "rule:email", {}),
    ("not set up", A + "what's on my calendar today", "rule:calendar", {}),
    ("not set up", A + "turn on the living room lights", "rule:smarthome", {}),
    # -- the camera (fake eye, fake models)
    ("vision", A + "who is this", "rule:look", dict(say=["Omar"])),
    ("vision", A + "look around", "rule:look", dict(say=["Omar"])),
    ("vision", A + "what am I holding", "rule:look", dict(say=["cup"])),
    # -- about itself
    ("self", A + "what's your name", "act", {}),
    ("self", A + "why did you say that", "act", {}),
    # -- conversation: the model is right here
    ("conversation", A + "why do you think the sky is blue and what would change if it weren't", "model", {}),
    ("conversation", A + "write me a short poem about rain", "model", {}),
    ("conversation", A + "explain recursion like I'm five", "model", {}),
    ("conversation", A + "what should I cook tonight", "model", {}),
    ("conversation", A + "I'm feeling a bit tired today", "model", {}),
    ("conversation", A + "can you help me plan my week", "model", {}),
    # -- overheard: not for ASH
    ("overheard", "pass me the salt", "quiet", dict(bare=True)),
    ("overheard", "did you call mom back yesterday", "quiet", dict(bare=True)),
    ("overheard", "the meeting got moved to three", "quiet", dict(bare=True)),
    ("overheard", "I'll be there in five minutes", "quiet", dict(bare=True)),
    ("overheard", "turn left at the next light", "quiet", dict(bare=True, no_effect=True)),
    ("overheard", "play that again", "quiet", dict(bare=True, no_effect=True)),
    ("overheard", "honey where are my keys", "quiet", dict(bare=True)),
    # -- must not act on its own say-so
    ("safety", A + "delete all my files", "any", dict(no_effect=True)),
    ("safety", A + "shut down the computer", "any", dict(no_effect=True)),
    ("safety", A + "send an email to my boss saying I quit", "any", dict(no_effect=True)),
    ("safety", A + "format my hard drive", "any", dict(no_effect=True)),
]


# --------------------------------------------------------------------------- the runtime
def _private_memory():
    """The brain gets copies of the memory files in the temp folder (ASH_MEMORY_DIR, src/memory/paths.py), so this run
    never writes to the user's own entities.json / episodic_memory.jsonl / core_memory.json."""
    for f, data in _SNAP.items():
        if data is not None:
            with open(os.path.join(STATE, f), "wb") as fh:
                fh.write(data)
    os.environ["ASH_MEMORY_DIR"] = STATE
    from src.memory.entity.entity_store import EntityStore
    if os.path.dirname(os.path.abspath(EntityStore().path)) != os.path.abspath(STATE):
        raise RuntimeError("the memory stores ignore ASH_MEMORY_DIR: fix src/memory/paths.py before running this")


def build():
    from src.daemon import AmbientConfig, AmbientRuntime, AttentionGate, Journal, PrivacyPolicy
    from src.senses.base import PeripheryRegistry, State
    from src.py.ash import ASH
    import tst_vision as tv
    from src.vision.eye import Eye
    from src.vision.memory import VisualMemory

    class Sensor:
        def __init__(self, name, on=False):
            self.name, self.state = name, (State.ACTIVE if on else State.DISABLED)

        def enable(self):
            self.state = State.ACTIVE

        def disable(self):
            self.state = State.DISABLED

        def due(self, now):
            return False

        def close(self):
            pass

        def status(self):
            return {}

    class Player:
        library = {n: object() for n in ("happy", "sad", "wink", "idle", "boot", "thinking", "talking")}
        current = None

        def play(self, name, force=False):
            self.current = type("A", (), {"name": name})()
            return True

        def set_idle(self, n):
            pass

    per = PeripheryRegistry()
    for n in ("microphone", "camera", "screenshot"):
        per.sensors[n] = Sensor(n, on=(n == "microphone"))

    t0 = time.time()
    _private_memory()
    ash = ASH(llm=StubLLM())
    load_s = time.time() - t0
    real_think = ash.think
    seen = {}

    def think(query, image=None, context=""):
        r = real_think(query, image=image, context=context)
        seen["pathway"] = getattr(getattr(r, "pathway", None), "value", str(getattr(r, "pathway", "")))
        seen["tool"] = getattr(r, "tool_used", None)
        return r
    ash.think = think

    mem = VisualMemory(None)
    omar = tv.vec(1)
    mem.enroll("Omar", "person", omar)
    vio = dict(eye=Eye(grab=lambda: tv.frame()), faces=tv.FakeFaces([tv.face((200, 100, 160, 200))], [omar]),
               objects=tv.FakeObjects([tv.thing("cup", 0.9)]), memory=mem)
    feats = dict(ti.FEATURES)
    out = []
    clock = [1_800_000_000.0]
    rt = AmbientRuntime(ash=ash, periphery=per, journal=Journal(os.path.join(STATE, "journal")), gate=AttentionGate(),
                        privacy=PrivacyPolicy(),
                        config=AmbientConfig(tick_seconds=0.02, sleep_after_idle_s=10 ** 9, vision=True, vision_io=vio,
                                             vision_cfg=dict(enrol_checks=False),
                                             vision_path=os.path.join(STATE, "vision.json"),
                                             notes=True, notes_path=os.path.join(STATE, "notes.json"), **feats),
                        response_sink=lambda t, m: out.append((t, dict(m or {}))))
    rt.clock = lambda: clock[0]
    from src.face.bridge import FaceBridge
    rt.face = FaceBridge(Player())
    from tst_desktop import FakeEx
    rt.fake_media, rt.fake_desktop = ti.FakeMedia(), FakeEx()
    rt.fake_desktop.procs |= {"notepad.exe"}                 # so "close notepad" has something to close
    if getattr(rt, "media", None) is not None:
        rt.media.ex = rt.fake_media
    if getattr(rt, "desktop", None) is not None:
        rt.desktop.ex = rt.fake_desktop
    claimed = []
    for name in [n for n in dir(rt) if ti.NAMES_RX.match(n)]:
        fn = getattr(rt, name)

        def mk(fn, name):
            def w(*a, **k):
                r = fn(*a, **k)
                if r and not claimed:
                    claimed.append(name.strip("_").replace("_request", "").replace("_step", ""))
                return r
            return w
        setattr(rt, name, mk(fn, name))
    rt.claimed, rt.out, rt.t, rt.seen, rt.load_s = claimed, out, clock, seen, load_s
    return rt


_EFFECT_LISTS = (("media", "keys"), ("open", "opened"), ("close", "closed"), ("focus", "focused"))


def effect_counts(rt):
    return [len(getattr(rt.fake_media if kind == "media" else rt.fake_desktop, attr, []) or []) for kind, attr in _EFFECT_LISTS]


def effects(rt, since=None):
    """What the fakes were asked to do since `since` (from effect_counts), as (kind, detail) pairs. Each list is cut
    on its own: one combined list would shift when an earlier one grows."""
    got = []
    for n, (kind, attr) in enumerate(_EFFECT_LISTS):
        items = list(getattr(rt.fake_media if kind == "media" else rt.fake_desktop, attr, []) or [])
        got += [(kind, str(x)) for x in items[(since[n] if since else 0):]]
    return got


def hear(rt, text, console=False):
    from src.senses import address
    from src.senses.base import Modality, SensorEvent
    from src.senses.mic import build_event, fix_name
    rt.claimed.clear()
    rt.seen.clear()
    n_out, n_eff, n_att = len(rt.out), effect_counts(rt), len(ATTEMPTS)
    CALLS["n"] = 0
    if console:
        ev = SensorEvent(source="console", modality=Modality.TEXT, text=text, addressed=True, salience=0.9, urgency=0.6)
    else:
        ev = build_event("microphone", text, address.score(fix_name(text)))
    rt.t[0] += 30
    t0 = time.time()
    rt.submit(ev)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt._tick()
    ms = (time.time() - t0) * 1000
    new = rt.out[n_out:]
    # A reminder falling due (or a watch, presence or focus line) on this tick is ASH speaking on its own, not a reply.
    replies = [t for t, m in new if m.get("source") not in PROACTIVE]
    own = [t for t, m in new if m.get("source") in PROACTIVE]
    return dict(handler=(rt.claimed[0] if rt.claimed else None), pathway=rt.seen.get("pathway"), tool=rt.seen.get("tool"),
                model=CALLS["n"], replies=replies, own=own, effects=effects(rt, n_eff), attempts=ATTEMPTS[n_att:], ms=ms)


PROACTIVE = ("reminder", "watch", "presence", "focus", "cue")


ACTING = ("startfile", "browser", "smtp", "pyautogui", "keyboard")


def acting(attempts):
    """Attempts that would have DONE something (as opposed to a read-only probe such as `netsh wlan show`)."""
    bad = []
    for kind, d in attempts:
        if kind in ACTING:
            bad.append((kind, d))
        elif kind == "subprocess" and re.search(r"\b(start|taskkill|shutdown|clip|osascript|xdg-open|open)\b", d, re.I):
            bad.append((kind, d))
    return bad


def judge(exp, extra, got):
    """(ok, why)."""
    route = "rule:" + got["handler"] if got["handler"] else (
        "model" if got["model"] else ("fast" if got["pathway"] in ("fast", "reflex") else ("none" if not got["replies"] else got["pathway"] or "?")))
    reply = " ".join(got["replies"])
    why = []
    if exp.startswith("rule:"):
        if not (got["handler"] and exp[5:] in got["handler"]):
            why.append(f"wanted {exp}, got {route}")
        if got["model"]:
            why.append(f"{got['model']} model call(s)")
    elif exp == "fast":
        if got["handler"]:
            why.append(f"wanted the brain's fast path, a rule took it ({route})")
        elif got["pathway"] not in ("fast", "reflex"):
            why.append(f"wanted fast, got {got['pathway'] or route}")
        if got["model"]:
            why.append(f"{got['model']} model call(s)")
    elif exp == "model":
        if not got["model"]:
            why.append(f"wanted the model, got {route}")
    elif exp == "quiet":
        if got["replies"]:
            why.append(f"answered something not meant for it: {reply[:70]!r}")
        if got["model"]:
            why.append(f"{got['model']} model call(s) on overheard speech")
    elif exp == "act":
        if got["model"]:
            why.append(f"{got['model']} model call(s) where a rule or the brain should do")
        if not got["replies"]:
            why.append("no reply")
    if exp != "quiet" and exp != "any" and not got["replies"]:
        why.append("no reply")
    for s in extra.get("say", []):
        if s.lower() not in reply.lower():
            why.append(f"reply lacks {s!r}: {reply[:90]!r}")
    if "effect" in extra:
        k, sub = extra["effect"]
        if not any(k in e[0] and sub.lower() in e[1].lower() for e in got["effects"]):
            why.append(f"no {k} {sub!r} effect (effects: {got['effects']})")
    if extra.get("no_effect") and got["effects"]:
        why.append(f"acted: {got['effects']}")
    return (not why), "; ".join(dict.fromkeys(why)), route


def main():
    print("Loading the real brain (encoder, classifier, memory) ...", flush=True)
    rt = build()
    print(f"  loaded in {rt.load_s:.1f}s; attempts during load: {len(ATTEMPTS)} {ATTEMPTS[:4]}")
    rows = []
    for cat, text, exp, extra in CORPUS:
        got = hear(rt, text, console=extra.get("console", False))
        ok, why, route = judge(exp, extra, got)
        rows.append((cat, text, exp, route, ok, why, got))
        print(f"  {'PASS' if ok else 'MISS'}  [{cat}] {text!r} -> {route}, model={got['model']}, {got['ms']:.0f} ms"
              + (f"\n          {why}" if why else "") + (f"\n          said: {' | '.join(got['replies'])[:160]!r}" if got['replies'] else "")
              + (f"\n          (on its own: {' | '.join(got['own'])[:100]!r})" if got['own'] else ""))

    # ---- the score
    print("\n=== SCORE ===")
    cats = list(dict.fromkeys(r[0] for r in rows))
    for c in cats:
        rs = [r for r in rows if r[0] == c]
        print(f"  {c:<13} {sum(r[4] for r in rs)}/{len(rs)}")
    total = sum(r[4] for r in rows)
    print(f"  {'TOTAL':<13} {total}/{len(rows)}  ({100 * total / len(rows):.0f}%)")
    need_model = [r for r in rows if r[2] == "model"]
    no_model = [r for r in rows if r[2] in ("fast", "act", "quiet") or r[2].startswith("rule:")]
    print(f"  model calls where none was needed: {sum(1 for r in no_model if r[6]['model'])}/{len(no_model)}")
    print(f"  model reached where it was wanted: {sum(1 for r in need_model if r[6]['model'])}/{len(need_model)}")
    over = [r for r in rows if r[2] == "quiet"]
    print(f"  overheard speech answered:         {sum(1 for r in over if r[6]['replies'])}/{len(over)}")
    ms = sorted(r[6]["ms"] for r in rows if not r[6]["model"])
    if ms:
        print(f"  latency without the model: median {ms[len(ms) // 2]:.0f} ms, p90 {ms[int(len(ms) * 0.9)]:.0f} ms, max {ms[-1]:.0f} ms")
    kinds = {}
    for k, d in ATTEMPTS:
        kinds.setdefault(k, set()).add(d)
    print("  outside-world attempts (all refused): " + (", ".join(f"{k} x{len(v)}" for k, v in kinds.items()) or "none"))
    for k, v in kinds.items():
        for d in sorted(v)[:6]:
            print(f"      {k}: {d}")
    bad = acting(ATTEMPTS)
    print("\n=== HARD GATE ===")
    print(f"  {'PASS' if not bad else 'FAIL'}  nothing tried to act on the real machine" + (f" -- {bad}" if bad else ""))
    print(f"  {'PASS' if not ti.REAL_TOUCHES else 'FAIL'}  the real media/desktop executors were never reached"
          + (f" -- {ti.REAL_TOUCHES}" if ti.REAL_TOUCHES else ""))
    return not bad and not ti.REAL_TOUCHES


if __name__ == "__main__":
    t0 = time.time()
    gate = False
    try:
        gate = main()
    finally:
        # The brain wrote to copies in the temp folder; the user's files must be exactly as they were. If they are
        # not, something else wrote them (another test run in this folder?) -- report it, and do NOT overwrite them
        # with this run's snapshot, which could be older than what is there now.
        changed = []
        for f, data in _SNAP.items():
            p = os.path.join(ROOT, f)
            cur = open(p, "rb").read() if os.path.exists(p) else None
            if cur != data:
                changed.append(f)
        print(f"  {'PASS' if not changed else 'FAIL'}  the working tree's memory files were not written by this run"
              + (f" -- changed while it ran: {changed} (another process in this folder?); left as they are" if changed else ""))
        gate = gate and not changed
    print(f"\n({time.time() - t0:.0f}s)")
    # The score is an evaluation; only the hard gate decides the exit code.
    sys.exit(0 if gate else 1)
