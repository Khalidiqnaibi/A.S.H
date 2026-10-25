import re
import sys
import ast, operator as op
from dotenv import load_dotenv
from datetime import datetime
import logging

load_dotenv()

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}

_MAX_EXPONENT = 1000      # 9**9**9 would otherwise hang the whole assistant


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("that exponent is too large")
        return _ALLOWED_OPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")


_NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUM_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
                   "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100})

# Longest phrases first, so "to the power of" is not eaten by "to".
_PHRASES = [
    (r"\bto the power of\b", "**"), (r"\braised to\b", "**"), (r"\bmultiplied by\b", "*"),
    (r"\bdivided by\b", "/"), (r"\bmodulo\b", "%"), (r"\bplus\b", "+"), (r"\bminus\b", "-"),
    (r"\btimes\b", "*"), (r"\bover\b", "/"), (r"\bmod\b", "%"),
    (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"), (r"\^", "**"), (r"×", "*"), (r"÷", "/"),
    (r"(?<=\d)\s*x\s*(?=\d)", "*"),
]
_LEAD = re.compile(
    r"^\s*(?:(?:hey|hi|ok|okay|ash)[,\s]+)?(?:please[,\s]+)?(?:can you\s+|could you\s+)?(?:please\s+)?"
    r"(?:what\s+is|what's|whats|how\s+much\s+is|how\s+much\s+are|calculate|compute|"
    r"work\s+out|evaluate|solve|figure\s+out|tell\s+me)\b[:\s]*(?:the\s+)?", re.I)


def to_expression(query: str) -> str:
    """Turn 'what is 15% of 240' / '12 times 7' into '15/100*240' / '12*7'.

    Anything that is not arithmetic once the phrasing is stripped is rejected
    rather than guessed at: the result is later evaluated.
    """
    s = (query or "").lower().strip()
    for _ in range(3):                       # "hey, can you please calculate ..."
        t = _LEAD.sub("", s)
        if t == s:
            break
        s = t
    s = s.replace("?", " ").replace(",", "") if re.search(r"\d,\d{3}", s) else s.replace("?", " ")
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*", r"(\1/100)*", s)
    # "divide 81 by 9", "multiply 6 by 7", "add 2 and 3", "subtract 3 from 10"
    s = re.sub(r"\bdivide\s+(\d[\d.]*)\s+by\s+(?=\d)", r"\1 / ", s)
    s = re.sub(r"\bmultiply\s+(\d[\d.]*)\s+(?:by|with)\s+(?=\d)", r"\1 * ", s)
    s = re.sub(r"\badd\s+(\d[\d.]*)\s+(?:and|to)\s+(?=\d)", r"\1 + ", s)
    s = re.sub(r"\bsubtract\s+(\d[\d.]*)\s+from\s+(\d[\d.]*)", r"\2 - \1", s)
    # "square root of 144" / "cube root of 27" -> a power the evaluator allows
    s = re.sub(r"\bsquare\s+root\s+of\s+(\d+(?:\.\d+)?)", r"(\1)**0.5", s)
    s = re.sub(r"\bcube\s+root\s+of\s+(\d+(?:\.\d+)?)", r"(\1)**(1/3)", s)
    for pat, rep in _PHRASES:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[a-z]+", lambda m: str(_NUM_WORDS[m.group()]) if m.group() in _NUM_WORDS else m.group(), s)
    s = s.strip().rstrip(".")
    if not re.fullmatch(r"[\d\s.+\-*/()%]+", s) or not re.search(r"\d", s):
        raise ValueError("I couldn't find a sum in that")
    return s


_DATEISH = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")


def looks_like_arithmetic(query: str) -> bool:
    """True when the whole request is a sum: it parses to an expression that
    has an operator in it. A deterministic signal the embedding classifier
    lacks -- "what is 12 times 7" scored 0.16 for calculate and lost to chat."""
    q = query or ""
    if _DATEISH.search(q):
        return False
    try:
        expr = to_expression(q)
        ast.parse(expr, mode="eval")
    except (ValueError, SyntaxError):
        return False
    return bool(re.search(r"(?<=[\d)])\s*(?:\*\*|[+\-*/%])\s*(?=[\d(])", expr))


def _pretty(v):
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return int(v)
        return float(f"{v:.10g}")
    return v


def safe_calc(expr: str):
    tree = ast.parse(expr, mode='eval')
    return _pretty(_safe_eval(tree.body))

def calculator_tool(expression: str) -> dict:
    """Perform safe arithmetic calculations."""
    logging.info('Function calculator_tool called')
    print(f"[TOOL] calculator_tool called with: {expression}", flush=True, file=sys.stderr)
    try:
        try:
            result = safe_calc(expression)
        except (SyntaxError, ValueError):
            # Not a bare expression: it is a sentence ("what is 12 times 7").
            result = safe_calc(to_expression(expression))
        print(f"[TOOL] calculator_tool result: {result}", flush=True, file=sys.stderr)
        return {"ok": True, "result": result}
    except ZeroDivisionError:
        return {"ok": False, "error": "couldn't calculate that: you can't divide by zero"}
    except Exception as e:
        print(f"[TOOL] calculator_tool error: {e}", flush=True, file=sys.stderr)
        # Returned as a dict so ToolEntry.run doesn't wrap the message as
        # {"ok": True, "result": "Error..."} -- which recorded every failed
        # calculation as a success and taught the brain to pick it more.
        return {"ok": False, "error": f"couldn't calculate that: {e}"}

def date_time_tool(_query: str = "") -> str:
    """Returns the current date and time, phrased the way a person
    would say it out loud rather than as a raw timestamp -- this gets
    handed to the LLM as a fact it is told to use verbatim, so whatever
    shape it is in here is close to what ends up spoken. Built by hand
    instead of strftime's %-I/%-d because those no-leading-zero codes
    are glibc-only and silently break on Windows.
    """
    logging.info('Function date_time_tool called')
    print("date_time_tool called", flush=True , file=sys.stderr)
    now = datetime.now()
    hour12 = now.hour % 12 or 12
    ampm = "AM" if now.hour < 12 else "PM"
    val = (f"{hour12}:{now.minute:02d} {ampm} on "
           f"{now.strftime('%A, %B')} {now.day}, {now.year}")
    print(f"date_time_tool returns {val}", flush=True , file=sys.stderr)
    return val


if __name__ == "__main__":

    # print(retrieve_tool("What is observer pattern?"))
    print(calculator_tool("12 / (2.3 + 0.7) * 4 - 3"))
    print(date_time_tool())