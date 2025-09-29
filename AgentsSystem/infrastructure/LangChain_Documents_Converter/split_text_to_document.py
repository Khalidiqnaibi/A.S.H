from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from .metadata_generators.generate_semantic_metadata import generate_semantic_metadata
from pydantic import SecretStr

def split_text_to_docs(
    text: str,
    base_metadata: Dict[str, Any],
    llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0,
    api_key= SecretStr("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"),
    base_url="https://openrouter.ai/api/v1"
)
,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Split text into chunks and enrich each with semantic metadata from LLM.
    """
    if not isinstance(text, str):
        try:
            text = str(text, "utf-8", errors="ignore")
        except Exception:
            text = str(text)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    
    documents = []
    for chunk in chunks:
        semantic_meta = generate_semantic_metadata(llm, chunk)
        # merge base metadata (like source, author) with chunk-specific metadata
        doc_metadata = {**base_metadata, **semantic_meta}
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    
    return documents
