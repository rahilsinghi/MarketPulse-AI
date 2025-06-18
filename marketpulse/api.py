""" 
FastAPI endpoints for MarketPulse AI.

Provides:
    * /ask  – answer questions about news articles
    * /docs – auto-generated OpenAPI docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import answer_query_sync, start_pipeline

# --------------------------------------------------------------------------- #
# 📝  Data models                                                              #
# --------------------------------------------------------------------------- #
class Question(BaseModel):
    """Request body for /ask endpoint."""
    text: str


class Answer(BaseModel):
    """Response body for /ask endpoint."""
    answer: str


# --------------------------------------------------------------------------- #
# 🚀  App & endpoints                                                          #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="MarketPulse AI",
    description="Real-time financial news analysis with GPT-4.",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    """Start news ingestion pipeline."""
    from .config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    start_pipeline()


@app.get("/")
def root():
    """Redirect to API documentation."""
    return {"message": "Welcome to MarketPulse AI", "docs": "/docs"}


@app.post("/ask", response_model=Answer)
def ask(question: Question) -> Answer:
    """Answer a question about recent news articles."""
    try:
        answer = answer_query_sync(question.text)
        return Answer(answer=answer)
    except Exception as exc:                    # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))