"""Retrieval-augmented question answering over your own documents, with verified citations."""

from .assistant import Answer, Citation, RAGAssistant
from .chunker import Chunk, Segment, chunk_document
from .embedders import HashingEmbedder, OpenAICompatibleEmbedder, make_embedder
from .ingest import ingest_path
from .llm import AnthropicClient, LLMError
from .retriever import Retriever
from .store import Store

__version__ = "0.1.0"

__all__ = [
    "Answer", "AnthropicClient", "Chunk", "Citation", "HashingEmbedder", "LLMError",
    "OpenAICompatibleEmbedder", "RAGAssistant", "Retriever", "Segment", "Store",
    "chunk_document", "ingest_path", "make_embedder",
]
