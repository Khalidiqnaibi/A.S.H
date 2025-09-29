from ....infrastructure.LangChain_Documents_Converter.DocumentConverter import DocumentConverter
from ....infrastructure.LangChain_Documents_Converter.split_text_to_document import split_text_to_docs
from typing import List
from langchain.schema import Document
import docx

class DocxConverter(DocumentConverter):
    def convert(self, source_path: str, *, chunk_size: int = 1000, chunk_overlap: int = 100, base_metadata = [{"Source": "docx"}]) -> List[Document]:
        doc = docx.Document(source_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paragraphs)
        return split_text_to_docs(text, base_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
