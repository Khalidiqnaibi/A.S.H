from typing import List, Optional
from langchain.retrievers import SelfQueryRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from ..vdb.Chroma.Chroma import ChromaVDB
from ...domain.interfaces.IRetriever import IRetriever
from ...domain.interfaces.IVectorDB import IVectorDB
from ..LangChain_Documents_Converter.LoaderFactory import LoaderFactory
from ..LangChain_Documents_Converter.converters import DocxConverter, PDFConverter, TextConverter, CSVConverter

class Retriever(IRetriever):
    """
    A unified retriever class that supports:
      1. Normal metadata-based retrieval
      2. SelfQueryRetriever (LLM-driven metadata filtering)
      3. Compression retriever (contextual compression with LLM)
    """

    def __init__(self,
                _documents_content: str,
                vdb: IVectorDB = ChromaVDB(),
                llm=None, 
                metadata_fields: Optional[List[AttributeInfo]] = None
                ):
        self._documents_content = _documents_content
        self.vdb = vdb
        self.llm = llm
        self.metadata_fields = metadata_fields or []
        self._metadata_map = self.vdb.get_metadata_map()
        LoaderFactory.register("csv", CSVConverter)
        LoaderFactory.register("txt", TextConverter)
        LoaderFactory.register("pdf", PDFConverter)
        LoaderFactory.register(".docx", DocxConverter)

    def retrieve(self, query: str, top_k: int = 5):
        return self.vdb.search(query, top_k=top_k)

    def create_selfquery_retriever(self, doc_description: str = "Document chunks"):
        if not self.llm:
            raise ValueError("LLM must be provided for SelfQueryRetriever")

        # document_texts = []
        # for schema in self._metadata_map.values():
        #     text = schema.get("text") or schema.get("content") or schema.get("page_content")
        #     if text:
        #         document_texts.append(text)


        # print(f"Creating SelfQueryRetriever with {len(document_texts)} documents.")


        #NOTE: here vectorstore must expose ⁠ .store ⁠ like Chroma/Faiss wrapper
        self._selfquery = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vdb.get_store(),
            document_content_description=doc_description,
            document_contents = self._documents_content,# make sure it doesnt break
            metadata_field_info=self.metadata_fields,
        )
        return self._selfquery

    def retrieve_with_selfquery(self, query: str, top_k: int = 5):
        if not self._selfquery:
            self.create_selfquery_retriever()
        return self._selfquery.get_relevant_documents(query)

    def create_compression_retriever(self, base_retriever=None, max_docs=6, max_chars_per_doc=1500):
        if not self.llm:
            raise ValueError("LLM must be provided for CompressionRetriever")

        if base_retriever is None:
            if not self._selfquery:
                self.create_selfquery_retriever()
            base = self._selfquery
        else:
            base = base_retriever

        compressor = LLMChainFilter.from_llm(self.llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base
        )