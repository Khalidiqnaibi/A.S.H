from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any
import json

def generate_semantic_metadata(llm, chunk: str) -> Dict[str, Any]:
    """
    Use an LLM to generate semantic metadata for a text chunk,
    ensuring all metadata values are compatible with Chroma.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts semantic metadata from text chunks."),
        ("human", "Generate metadata as JSON with keys: 'summary', 'keywords', 'topic'.\n"
                  "Make sure keywords are a comma-separated string, not a list.\n\nText:\n{chunk}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"chunk": chunk})
    # print(response)
    # LLM returns text; parse safely
    try:
        metadata = json.loads(response.content)  # try to parse JSON
    except Exception:
        # fallback: just put text into summary
        metadata = {
            "summary": response.content[:200],
            "keywords": "",
            "topic": "unknown"
        }
    
    # sanitize metadata for Chroma (no lists/dicts)
    safe_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            safe_metadata[k] = ", ".join(str(i) for i in v)  # convert list -> string
        elif isinstance(v, dict):
            safe_metadata[k] = json.dumps(v)  # convert dict -> string
        else:
            safe_metadata[k] = v
    return safe_metadata
