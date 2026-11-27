"""
src/daemon/speedups.py

Things that make a slow machine feel quick, all on this computer, none needing the internet:

    AnswerCache   the same factual question asked again within a day gets the same answer at once -- no model. Only questions that do not lean on the moment
                  ("what is a transformer", not "what about now" / "what did I do today"), and only answers that were not a failure.
    more()        "more" / "tell me more" / "go on" right after an answer: rewritten into "Tell me more about: <the question>" so the model expands that answer
                  instead of guessing what "more" was.
    best_model()  of the models Ollama has, the one that fits this computer: the biggest at or under the size its memory can hold alongside everything else.
    why_slow()    "why was that slow?" -- from the last turn's real record: whether the model had to load, memory, CPU, how big the model is.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

MIN_ANSWER = 20
_LEANS = re.compile(r"\b(?:my|me|i|i'm|we|you|your|this|that|it|its|these|those|today|tonight|tomorrow|yesterday|now|current|currently|latest|last|next|here|there|"
                    r"again|before|earlier|just|recent|recently|ago|him|her|them|his|their|what\s+about|and\s+(?:what|how)|so)\b", re.I)


def _key(prompt: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (prompt or "").lower()))[:160]


def cacheable(prompt: str, answer: str) -> bool:
    p = (prompt or "").strip()
    if len(p.split()) < 4 or len(p) > 160 or len(answer or "") < MIN_ANSWER or _LEANS.search(p):
        return False
    return bool(re.match(r"^(?:what|who|why|how|when|where|which|explain|define|describe|tell\s+me)\b", p, re.I))


class AnswerCache:
    def __init__(self, path: Optional[str] = None, ttl_s: float = 86400.0, keep: int = 200, clock=time.time):
        self.path, self.ttl, self.keep, self.clock = path, ttl_s, keep, clock
        self.items: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        if path:
            try:
                from src.memory.atomic import load_json
                self.items = dict((load_json(path, {}) or {}).get("items") or {})
            except Exception:
                self.items = {}

    def get(self, prompt: str) -> Optional[str]:
        e = self.items.get(_key(prompt))
        if not e or self.clock() - float(e.get("t", 0)) > self.ttl:
            return None
        self.hits += 1
        return str(e["a"])

    def put(self, prompt: str, answer: str) -> bool:
        if not cacheable(prompt, answer):
            return False
        self.items[_key(prompt)] = {"a": answer, "t": self.clock()}
        if len(self.items) > self.keep:
            for k in sorted(self.items, key=lambda k: self.items[k]["t"])[: len(self.items) - self.keep]:
                self.items.pop(k, None)
        if self.path:
            try:
                from src.memory.atomic import write_json
                write_json(self.path, {"items": self.items})
            except Exception:
                pass
        return True

    def forget(self) -> int:
        n = len(self.items)
        self.items = {}
        if self.path:
            try:
                from src.memory.atomic import write_json
                write_json(self.path, {"items": {}})
            except Exception:
                pass
        return n


_MORE = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:ok(?:ay)?|and|so|yes|yeah)[,\s]+)?(?:(?:tell\s+me\s+)?more(?:\s+(?:about\s+(?:that|it)|please))?|go\s+on|continue|keep\s+going|"
                   r"elaborate|expand(?:\s+on\s+(?:that|it))?|say\s+more|(?:can|could)\s+you\s+(?:say|tell\s+me)\s+more)\W*$", re.I)


def more(text: str, last_prompt: str, last_at: float, now: float) -> Optional[str]:
    """The request "more" stands for, or None when it is not "more" (or nothing was just asked)."""
    if not _MORE.match((text or "").strip()) or not last_prompt or now - last_at > 600.0:
        return None
    return f"Tell me more about this, in a few sentences: {last_prompt}"


def params_b(name: str) -> float:
    m = re.search(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*b\b", (name or "").lower())
    if m:
        return float(m.group(1))
    base = (name or "").lower().split(":")[0]
    return 7.0 if base in ("mistral", "llama2", "llama3", "mixtral", "deepseek-r1", "gemma2", "qwen3", "phi4") else 3.0


def best_model(installed: List[str], ram_gb: float) -> Tuple[str, str]:
    """(model, why). A 4-bit model needs about 0.65 GB per billion parameters and the machine needs the rest for itself: allow a quarter of the memory."""
    usable = [n for n in installed if "embed" not in n.lower() and "rerank" not in n.lower()]
    if not usable:
        return "", "no language model is installed"
    budget_b = max(1.5, (ram_gb / 4.0) / 0.65)
    fit = [n for n in usable if params_b(n) <= budget_b]
    pick = max(fit, key=params_b) if fit else min(usable, key=params_b)
    return pick, f"{ram_gb:.0f} GB of memory holds about {budget_b:.0f}B parameters comfortably"


_WHY = re.compile(r"^\W*(?:(?:hey\s+)?ash[,\s]+)?(?:why\s+(?:was|is|did)\s+(?:that|it|the\s+(?:last\s+)?(?:answer|one))\s+(?:so\s+)?(?:slow|take\s+so\s+long|taking\s+so\s+long|late)|"
                  r"why\s+did\s+(?:that|it)\s+take\s+(?:so\s+long|that\s+long|forever)|why\s+so\s+slow|why\s+are\s+you\s+(?:so\s+)?slow)\W*$", re.I)


def asks_why_slow(text: str) -> bool:
    return bool(_WHY.match((text or "").strip()))


def why_slow(took_s: float, note: str, model: str, mem: Optional[float], cpu: Optional[float], pathway: str) -> str:
    if not took_s:
        return "I haven't had a slow one just now."
    if pathway not in ("slow", "escalated"):
        return f"That one was quick ({took_s:.0f} s) and by rule."
    bits = []
    if note:
        bits.append(note)
    if model and params_b(model) >= 6.0:
        bits.append(f"{model} is a big model for this computer")
    if mem is not None and mem >= 85 and not any("memory" in b for b in bits):
        bits.append(f"memory is at {mem:.0f}%")
    if cpu is not None and cpu >= 85:
        bits.append(f"the processor is at {cpu:.0f}%")
    why = "; ".join(bits) if bits else "the model simply had a lot to write"
    fix = " A smaller model (say “use the small model”), or closing a big program, would help." if took_s >= 30 else ""
    return f"The last one took {took_s:.0f} s because {why}.{fix}"
