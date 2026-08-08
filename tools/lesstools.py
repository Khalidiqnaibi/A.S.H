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

def _safe_eval(node):
    if isinstance(node, ast.Num):  
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError("Unsupported expression")

def safe_calc(expr: str):
    tree = ast.parse(expr, mode='eval')
    return _safe_eval(tree.body)

def calculator_tool(expression: str) -> str:
    """Perform safe arithmetic calculations."""
    logging.info('Function calculator_tool called')
    print(f"[TOOL] calculator_tool called with: {expression}", flush=True, file=sys.stderr)
    try:
        result = safe_calc(expression)
        print(f"[TOOL] calculator_tool result: {result}", flush=True, file=sys.stderr)
        return f"Result: {result}"
    except Exception as e:
        print(f"[TOOL] calculator_tool error: {e}", flush=True, file=sys.stderr)
        return f"Error in calculation: {str(e)}"

def date_time_tool(_query: str = "") -> str:
    """Returns the current date and time now."""
    logging.info('Function date_time_tool called')
    print("date_time_tool called", flush=True , file=sys.stderr)
    val = datetime.now().isoformat()
    print(f"date_time_tool returns {val}", flush=True , file=sys.stderr)
    return val


if __name__ == "__main__":

    # print(retrieve_tool("What is observer pattern?"))
    print(calculator_tool("12 / (2.3 + 0.7) * 4 - 3"))
    print(date_time_tool())