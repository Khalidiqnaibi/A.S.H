"""tst_memnotice.py -- memory above 90% for twenty minutes is said once, with what holds the most; not again for three hours.
Run:  PYTHONIOENCODING=utf-8 python tests/tst_memnotice.py"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASH_BRAIN_STATE", tempfile.mkdtemp())
import tst_wondering as tw  # noqa: E402
import tst_modellast as tm  # noqa: E402
from src.daemon import machine  # noqa: E402

check = tw.check
GB = 1024 ** 3


def main():
    rt = tm.rig()
    said = []
    rt.response_sink = lambda t, m: said.append(t)
    rt.governor = types.SimpleNamespace(mem=92.0)
    rt._machine = machine.Machine(procs=lambda: [("opera.exe", 3 * GB, 1), ("opera.exe", int(2.1 * GB), 2), ("Code.exe", GB, 3), ("python.exe", 4 * GB, 9)],
                                  mem=lambda: (92.0, 0, 0), spec=lambda: {}, own_pid=9)
    t0 = rt.clock()
    rt._memory_notice(t0)
    rt._memory_notice(t0 + 600)
    check("not before twenty minutes", said == [], str(said))
    rt._memory_notice(t0 + 1260)
    check("then said once, naming the biggest (not ASH itself)", said == ["Memory has been above 90% for 21 minutes, mostly Opera (5.1 GB) -- that's why I'm slow. "
                                                                          "Say “what's using my memory” for the rest."], str(said))
    rt._memory_notice(t0 + 3000)
    check("not again for three hours", len(said) == 1)
    rt.governor.mem = 70.0
    rt._memory_notice(t0 + 3100)
    rt.governor.mem = 95.0
    rt._memory_notice(t0 + 3200)
    rt._memory_notice(t0 + 3200 + 1300)
    check("a dip resets the twenty minutes, and the three hours still hold", len(said) == 1)
    rt._memory_notice(t0 + 4 * 3600)
    check("after three hours, said again", len(said) == 2, str(said))
    rt2 = tm.rig()
    said2 = []
    rt2.response_sink = lambda t, m: said2.append(t)
    rt2.governor = types.SimpleNamespace(mem=95.0)
    rt2._machine = rt._machine
    rt2.gate.muted = True
    t1 = rt2.clock()
    rt2._memory_notice(t1)
    rt2._memory_notice(t1 + 1300)
    check("muted: nothing said", said2 == [])
    rt2.governor = types.SimpleNamespace(mem=None)
    rt2._memory_notice(t1 + 1400)
    check("nothing measured yet: nothing", said2 == [])
    print("\n" + ("ALL PASS" if tw.OK else "FAILED"))
    sys.exit(0 if tw.OK else 1)


if __name__ == "__main__":
    main()
