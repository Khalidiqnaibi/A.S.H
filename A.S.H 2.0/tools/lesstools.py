from langchain.tools import tool  
import ast, operator as op
from AgentsSystem import Retriever, ChromaVDB, mistral
from langchain.chains.query_constructor.schema import AttributeInfo
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

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

class RetrieverFactory:
    def __init__(self, host: str = None, port: int = None):
        self.chroma = ChromaVDB(meta_api_url="http://localhost:2020")

        if host and port:
            self.chroma.init_cloud_db_client(host=host, port=port)

        # Save metadata schema for search
        self.chroma.save_metadata_map(metadata={f.name: {
            "type": f.type,
            "description": f.description
        } for f in METADATA_FIELDS})

    def build_retriever(self, llm=None, description: str = None):
        if llm is None:
            llm = mistral.MistralLLM(mode="openrouter", temperature=0.7)

        retriever = Retriever(
            "Financial Knowledge",
            vdb=self.chroma,
            llm=llm,
            metadata_fields=METADATA_FIELDS
        )
        retriever.create_selfquery_retriever(doc_description=description)
        return retriever


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

@tool
def calculator_tool(expression: str) -> str:
    """Perform safe arithmetic calculations."""
    try:
        result = safe_calc(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool
def date_time_tool(x: str) -> str:
    """Returns the current date and time now."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def stock_market_tool(symbol: str) -> str:
    """
    Retrieve the latest stock market information for a given ticker symbol.
    Example: 'AAPL' for Apple, 'TSLA' for Tesla.
    Returns: Current price, change, and other relevant data.
    """

    try:
        # Example API (Finnhub). Replace with your provider if needed
        API_KEY = FINHUB_API_KEY  
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"

        response = requests.get(url)
        if response.status_code != 200:
            return f"Error fetching stock data: {response.text}"

        data = response.json()
        if "c" not in data:
            return "Invalid response from stock API."

        current_price = data.get("c", "N/A")
        high = data.get("h", "N/A")
        low = data.get("l", "N/A")
        open_price = data.get("o", "N/A")
        prev_close = data.get("pc", "N/A")

        return (
            f"Stock: {symbol}\n"
            f"Current Price: {current_price}\n"
            f"High: {high}\n"
            f"Low: {low}\n"
            f"Open: {open_price}\n"
            f"Previous Close: {prev_close}\n"
        )

    except Exception as e:
        return f"Error: {str(e)}"
    
def make_retriever_tool(retriever, tool_name="knowledge_retrieval_tool", description=None):
    """
    Dynamically wraps a retriever instance into a LangChain tool.
    """
    @tool(tool_name)
    def _retriever_tool(query: str) -> str:
        """
        Retrieve relevant knowledge from the vector database using semantic + metadata search.
        Input: natural language query
        Output: concise structured knowledge text
        """
        results = retriever.retrieve_with_selfquery(query)
        if not results:
            return "No relevant knowledge found."
        
        formatted = "\n".join(
            [f"- {doc.page_content} (source: {doc.metadata.get('source', 'unknown')})"
             for doc in results]
        )
        return formatted

    if description:
        _retriever_tool.__doc__ = description

    return _retriever_tool

factory = RetrieverFactory(
   host="localhost",
   port=4444
)

if __name__ == "__main__":

    retriever = factory.build_retriever(
        llm=mistral.MistralLLM(mode="openrouter", temperature=0.7),
        description="the type of observer design pattern"
    )

    retrieval_tool = make_retriever_tool(retriever)

    print(retrieval_tool("What is observer pattern?"))
