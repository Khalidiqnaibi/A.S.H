import os
import sys
from langchain.tools import tool  
import ast, operator as op
from AgentSystem import Retriever, ChromaVDB, mistral
from langchain_classic.chains.query_constructor.schema import AttributeInfo
import requests
from dotenv import load_dotenv
from datetime import datetime
from flask import  session
import logging

load_dotenv()

LLM = mistral.MistralLLM(mode="openrouter",openrouter_key=os.getenv("OPENROUTER_API_KEY"), temperature=0.7)

CHROMA = ChromaVDB(llm=LLM)
CHROMA.init_cloud_db_client()


METADATA_FIELDS = [
    AttributeInfo(
        name="source",
        description="Origin of the knowledge item (e.g., blog, article, manual)",
        type="string"
    ),
    AttributeInfo(
        name="type",
        description="Type of content (text, image, video, audio)",
        type="string"
    ),
    AttributeInfo(
        name="description",
        description="Short explanation about the content",
        type="string"
    ),
    AttributeInfo(
        name="summary",
        description="Optional summarized version of the content",
        type="string"
    ),
    AttributeInfo(
        name="keywords",
        description="Comma-separated keywords for search and retrieval",
        type="string"
    ),
    AttributeInfo(
        name="topic",
        description="Category or topic of the knowledge item",
        type="string"
    )
]

# Save metadata schema for search
CHROMA.save_metadata_map(metadata={f.name: {
    "type": f.type,
    "description": f.description
} for f in METADATA_FIELDS})

def retrieve_tool(query: str,top=5, llm=None) -> str:
    """Retrieve relevant knowledge from the vector database using semantic + metadata search."""
    logging.info('Function retrieve_tool called')
    print(f"[TOOL] retrieve_tool called with: {query}", flush=True, file=sys.stderr)
    if llm is None:
        llm = LLM
    R=Retriever(
        "Knowledge",
        vdb=CHROMA,
        llm=llm,
        metadata_fields=METADATA_FIELDS
    )
    return R.retrieve(query, top_k=top)

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

def date_time_tool() -> str:
    """Returns the current date and time now."""
    logging.info('Function date_time_tool called')
    print("date_time_tool called", flush=True , file=sys.stderr)
    val = datetime.now().isoformat()
    print(f"date_time_tool returns {val}", flush=True , file=sys.stderr)
    return val


if __name__ == "__main__":

    print(retrieve_tool("What is observer pattern?"))
    print(calculator_tool("12 / (2.3 + 0.7) * 4 - 3"))
    print(date_time_tool())
    
