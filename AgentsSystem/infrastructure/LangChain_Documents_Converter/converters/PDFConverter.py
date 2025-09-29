from ....infrastructure.LangChain_Documents_Converter.DocumentConverter import DocumentConverter
from ....infrastructure.LangChain_Documents_Converter.split_text_to_document import split_text_to_docs
from langchain.schema import Document
from typing import List
import PyPDF2


class PDFConverter(DocumentConverter):
    def convert(self, source_path: str, *, chunk_size: int = 1000, chunk_overlap: int = 100, base_metadata = [{"source": "pdf"}]) -> List[Document]:
        text_parts = []
        with open(source_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for i, page in enumerate(reader.pages):
                # extract_text may return None
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    text_parts.append(f"--- PAGE {i+1} ---\n{txt}")
        text = "\n\n".join(text_parts)
        
        return split_text_to_docs(text, base_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
