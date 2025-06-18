""" 
MarketPulse AI – Backend package.

Exposes:
    * start_pipeline():  boots the Pathway ingestion thread.
    * answer_query_sync():  RAG pipeline – embed → retrieve → GPT-4 answer.
"""

from .ingestion import start_pipeline           # noqa: F401
from .llm import answer_query_sync              # noqa: F401