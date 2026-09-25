"""T3: retrieval package — evidence acquisition for claim verification.

Modules:
  chunk.py     document chunking (plain text; pypdf/python-docx parsing helpers)
  index.py     disk-persisted hybrid index (BM25 + FAISS dense + RRF fusion)
  web_search.py Brave-backed web search (LLM Context + Web snippets, Tavily fallback)
  rerank.py    lazy cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
"""

from .brave_answers import BraveAnswers  # noqa: F401
from .chunk import chunk_text, extract_text_from_bytes  # noqa: F401
from .index import RetrievalIndex  # noqa: F401
from .web_search import WebSearch  # noqa: F401
