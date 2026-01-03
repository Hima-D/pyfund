# src/pyfundlib/ml/rag.py
from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from pyfundlib.config import settings
from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)


class QuantRAG:
    """
    Grounds LLM insights in pyfundlib computations.
    Prevents hallucinations by retrieving exact numbers from a vector store.
    """

    def __init__(self, index_path: Optional[str] = None):
        self.embeddings = OpenAIEmbeddings(api_key=settings.api_key)
        self.index_path = index_path
        self.vector_store = None
        
        if index_path and os.path.exists(index_path):
            self.vector_store = FAISS.load_local(index_path, self.embeddings)

    def index_data(self, documents: List[str], metadata: Optional[List[dict]] = None):
        """Index quant reports or computed results"""
        logger.info("indexing_quant_data", count=len(documents))
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(documents, self.embeddings, metadatas=metadata)
        else:
            self.vector_store.add_texts(documents, metadatas=metadata)
            
        if self.index_path:
            self.vector_store.save_local(self.index_path)

    def query(self, user_query: str, k: int = 3) -> str:
        """Query the grounded quant knowledge base"""
        if self.vector_store is None:
            return "Knowledge base empty. Please index data first."
            
        docs = self.vector_store.similarity_search(user_query, k=k)
        context = "\n---\n".join([d.page_content for d in docs])
        
        logger.info("rag_query_performed", query=user_query)
        return context
