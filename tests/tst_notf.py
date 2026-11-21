"""
tst_notf.py -- ASH does not load TensorFlow.

transformers imports TensorFlow and Keras when it finds them, for models ASH never loads: about 250 MB and 14 s of one core at every start
(measured: 549 MB / 23.7 s with, 302 MB / 9.7 s without).

Run:  PYTHONIOENCODING=utf-8 python tests/tst_notf.py
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tst_wondering as tw  # noqa: E402

check = tw.check
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def n1_import():
    print("\n=== N1. Importing the classifier does not import TensorFlow ===")
    code = ("import os, sys; sys.path.insert(0, r'%s'); os.environ.pop('USE_TF', None); os.environ.pop('TRANSFORMERS_NO_TF', None); "
            "import tools.classification; print('TF=' + str('tensorflow' in sys.modules) + ' USE_TF=' + os.environ.get('USE_TF', '')) " % ROOT)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=240, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    last = [l for l in out.stdout.splitlines() if l.startswith("TF=")]
    check("tensorflow is not loaded, USE_TF is 0", bool(last) and last[-1] == "TF=False USE_TF=0", str(last or out.stderr[-300:]))
    env2 = {**os.environ, "USE_TF": "1", "PYTHONIOENCODING": "utf-8"}
    src = open(os.path.join(ROOT, "tools", "classification.py"), encoding="utf-8").read()
    check("a machine that sets USE_TF itself is not overridden (setdefault)", 'os.environ.setdefault("USE_TF", "0")' in src)


if __name__ == "__main__":
    n1_import()
    print("\n" + ("ALL PASS" if tw.OK else "FAILURES"))
    sys.exit(0 if tw.OK else 1)
