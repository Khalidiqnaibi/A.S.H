"""
tst_username.py -- the person's name is a setting (console settings -> "Your name"), not a constant in the code.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_username.py
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.daemon import identity  # noqa: E402
from src.ui import settings  # noqa: E402

check = tw.check


class FakeAsh(identity.NameMixin):
    def __init__(self, user="Immortal"):
        self.user = user
        self.user_aliases = set()
        self.brain = NS(user=user, user_aliases=set())


def u1_default():
    print("\n=== U1. The default ===")
    old = {k: os.environ.pop(k, None) for k in ("ASH_USER_NAME", "DEV_NAME")}
    try:
        check("no name set anywhere: a plain 'User'", identity.default_user() == "User")
        os.environ["DEV_NAME"] = '"IMMORTAL"'
        check("the developer's DEV_NAME is used (the owner's machine is unchanged)", identity.default_user() == "Immortal", identity.default_user())
        os.environ["ASH_USER_NAME"] = "sara"
        check("ASH_USER_NAME wins", identity.default_user() == "Sara")
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def u2_change():
    print("\n=== U2. Changing it ===")
    a = FakeAsh("Immortal")
    check("it changes in ASH and in the brain", a.set_user_name("Khalid") == "Khalid" and a.user == "Khalid" and a.brain.user == "Khalid")
    check("the old name is kept as an alias, on the brain too", a.user_aliases == {"Immortal"} and a.brain.user_aliases == {"Immortal"})
    a.set_user_name("   ")
    check("an empty name changes nothing", a.user == "Khalid")
    b = FakeAsh("User")
    b.set_user_name("Sara")
    check("'User' was never a name: not an alias", b.user_aliases == set())


def u3_settings():
    print("\n=== U3. In the settings menu ===")
    check("it is a setting, first in the list, in a group of its own", list(settings.SETTINGS)[0] == "identity.user_name"
          and settings.SETTINGS["identity.user_name"][2] == "You")
    for bad in ("", "  ", "x" * 41, "Khalid\nIqnaibi", "Rob<script>", "R2D2"):
        try:
            settings.parse("identity.user_name", bad)
            check(f"rejected: {bad[:12]!r}", False)
        except ValueError:
            check(f"rejected: {bad[:12]!r}", True)
    check("a name is cleaned of extra spaces", settings.parse("identity.user_name", "  Khalid   Iqnaibi ") == "Khalid Iqnaibi")
    check("an apostrophe, dot and hyphen are fine", settings.parse("identity.user_name", "Mary-Jane O'Neil Jr.") == "Mary-Jane O'Neil Jr.")
    d = tempfile.mkdtemp()
    path = os.path.join(d, "daemon.json")
    json.dump({"weather": {"city": "Amman"}}, open(path, "w"))
    rt = NS(ash=FakeAsh("Immortal"))
    res = settings.change(rt, path, "identity.user_name", "Khalid")
    check("the change is live and saved", res.get("ok") and res.get("live") and res.get("saved") and not res.get("restart"), str(res))
    saved = json.load(open(path))
    check("daemon.json has the name and the old one, and keeps the rest",
          saved["identity"]["user_name"] == "Khalid" and saved["identity"]["previous_names"] == ["Immortal"] and saved["weather"]["city"] == "Amman", str(saved))
    lst = {s["key"]: s for s in settings.listing(rt, path)}
    check("the panel shows the current name and says it is live", lst["identity.user_name"]["value"] == "Khalid" and lst["identity.user_name"]["live"])
    res = settings.change(rt, path, "identity.user_name", "bad<name>")
    check("a bad name is refused and nothing changes", not res["ok"] and rt.ash.user == "Khalid")


def u4_history():
    print("\n=== U4. What was said under the old name is still found ===")
    from src.daemon.ambient import AmbientRuntime
    rt = AmbientRuntime.__new__(AmbientRuntime)
    ep = lambda s: NS(summary=s)
    rt.ash = NS(user_aliases={"Immortal"}, episodic_memory=NS(store=NS(episodes={1: ep("Immortal: my dad is Afif"), 2: ep("Khalid: I like chess"),
                                                                                    3: ep("A.S.H: hello"), 4: ep("Sara: not mine")})),
                core_memory=None)
    rt.brain = NS(user="Khalid")
    got = AmbientRuntime._user_sentences(rt)
    check("old and new names both count, others do not", "my dad is Afif" in got and "I like chess" in got and "not mine" not in got and "hello" not in got, str(got))


if __name__ == "__main__":
    u1_default()
    u2_change()
    u3_settings()
    u4_history()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
