"""tst_ctx.py -- the context window asked of Ollama fits the call. Run:  PYTHONIOENCODING=utf-8 python tests/tst_ctx.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.LLM import context_for

OK = True
def check(n, c, d=""):
    global OK
    print(("  ok   " if c else "  FAIL ") + n + ("" if c else f"  {d}"))
    OK = OK and bool(c)

os.environ.pop("ASH_OLLAMA_NUM_CTX", None)
small = [{"role": "system", "content": "x" * 2900}, {"role": "user", "content": "y" * 1500}]
check("an ordinary call gets 3072, not 4096", context_for(small, 512) == 3072, str(context_for(small, 512)))
big = [{"role": "system", "content": "x" * 2900}, {"role": "user", "content": "y" * 8000}]
check("a longer prompt gets a window it fits in", context_for(big, 512) >= 4096 and (2900 + 8000) // 3 + 512 <= context_for(big, 512))
check("never smaller than the answer needs", context_for([], 3000) >= 3064)
os.environ["ASH_OLLAMA_NUM_CTX"] = "2048"
check("ASH_OLLAMA_NUM_CTX fixes it", context_for(big, 512) == 2048)
os.environ["ASH_OLLAMA_NUM_CTX"] = "abc"
check("nonsense is ignored", context_for(small, 512) == 3072)
print("\n" + ("ALL PASS" if OK else "FAILED"))
sys.exit(0 if OK else 1)
