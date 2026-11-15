"""
tst_classify.py -- how much harm an action can do, guessed from its name.

classify_action() sorts a tool by its intent name when its shape file does not say. Review found it matched whole lowercase words,
so `deleteFile` (an MCP tool is registered under its own camelCase name) became "deletefile" and matched nothing, and
eraseDisk / terminate / uninstall / reset / restart / execute / unlock were not in the lists: all of them fell through to
"retrieval, reversible" -- the class that takes the fast path, runs as a reflex and is allowed as an autonomous plan step.

Run:  PYTHONIOENCODING=utf-8 python tests/tst_classify.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())

import tst_wondering as tw  # noqa: E402
from src.brain.system1 import ActionClass, classify_action  # noqa: E402

check = tw.check
I, P, M, R, C, V = (ActionClass.IRREVERSIBLE, ActionClass.PHYSICAL, ActionClass.MUTATION, ActionClass.RETRIEVAL,
                    ActionClass.COMPUTE, ActionClass.CONVERSATION)


def t1_harmful():
    print("\n=== Harmful, said in any style ===")
    for name in ("deleteFile", "delete_file", "DeleteFile", "eraseDisk", "erase_disk", "terminate_process", "terminateProcess",
                 "uninstall_app", "restartService", "resetPassword", "executeCommand", "unlockVault", "shutdownNow",
                 "sendEmail", "wipeDrive", "truncateTable", "revokeToken", "destroyBackup", "logoutAll"):
        cls, rev = classify_action(name)
        check(f"{name}: irreversible", cls == I and rev is False, f"{cls} {rev}")
    for name in ("openGarageDoor", "unlock_door", "motor_on", "sirenOn"):
        cls, rev = classify_action(name)
        check(f"{name}: not a retrieval (physical or irreversible, never autonomous)", cls in (P, I) and rev is False, f"{cls} {rev}")


def t2_mutation():
    print("\n=== Changes things, can be undone ===")
    for name in ("set_timer", "write_file", "openApp", "launchProgram", "run_command", "installPackage", "clearCache", "toggleLight",
                 "stop_music", "cancel_reminder", "uploadFile", "clickButton"):
        cls, rev = classify_action(name)
        check(f"{name}: a mutation", cls == M and rev is True, f"{cls} {rev}")


def t3_harmless():
    print("\n=== What must stay quick ===")
    for name, want in (("get_weather", R), ("read_file", R), ("list_directory", R), ("search_files", R), ("web_search", R),
                       ("get_time", R), ("file_info", R), ("system_stats", R), ("sentiment_analysis", R),
                       ("calculate", C), ("word_count", C), ("convert_units", C), ("conversation", V), ("", V),
                       ("presetLoader", R), ("assetManager", R), ("dataset_info", R), ("getStatsSummary", R),
                       ("listOpenTabs", R)):
        cls, rev = classify_action(name)
        check(f"{name or '(empty)'}: {want.value}", cls == want and rev is True, f"{cls} {rev}")


def t4_declared():
    print("\n=== A shape file's word is final ===")
    cls, rev = classify_action("deleteFile", {"action_class": "retrieval", "reversible": True})
    check("a shape that declares its class overrides the guess (the tool's author knows)", cls == R and rev is True)
    cls, rev = classify_action("get_weather", {"action_class": "physical"})
    check("...either way", cls == P and rev is False)


if __name__ == "__main__":
    t1_harmful()
    t2_mutation()
    t3_harmless()
    t4_declared()
    print("\nALL PASS" if tw.OK else "\nFAILED")
    sys.exit(0 if tw.OK else 1)
