"""
FastAPI chat backend.

Endpoints:
  POST /chat           — full pipeline + LLM synthesis
  POST /pipeline/raw   — pipeline only (no LLM), returns structured JSON
  GET  /health         — liveness check
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

app = FastAPI(
    title="Manufacturing Chatbot API",
    description="Multimodal anomaly detection + LLM explanation pipeline",
    version="0.1.0",
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    machine_id: str | None = None
    query_time: str | None = None   # ISO 8601 string
    # Conversation memory for follow-up parameter queries. The client echoes back
    # the `param_context` from the previous response so a follow-up ("now stay in
    # 1.5 diameter and more than 21 voltage") inherits the earlier material/
    # thickness and accumulated pinned parameters. Stateless on the server.
    param_context: dict | None = None


class ChatResponse(BaseModel):
    answer: str
    payload: dict


class PipelineResponse(BaseModel):
    payload: dict


# ── Routing helper ────────────────────────────────────────────────────────────

def _route_and_run(question: str, machine_id: str | None, query_time,
                   param_context: dict | None = None):
    """Route a question to the param / knowledge / anomaly pipeline and run it."""
    from src.api.pipeline import (
        route_question,
        run_pipeline,
        run_param_pipeline,
        run_knowledge_pipeline,
        run_general_pipeline,
    )

    route = route_question(question)
    if route == "param":
        return run_param_pipeline(question, context=param_context)
    if route == "knowledge":
        return run_knowledge_pipeline(question)
    if route == "general":
        return run_general_pipeline(question)
    return run_pipeline(question=question, machine_id=machine_id, query_time=query_time)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Run the full pipeline and return a plain-language LLM answer + raw payload.
    """
    from src.chat.synthesize import synthesize

    from src.chat.intent import OutOfScopeError

    try:
        query_time = None
        if req.query_time:
            from datetime import datetime
            query_time = datetime.fromisoformat(req.query_time)

        payload = _route_and_run(req.question, req.machine_id, query_time,
                                 param_context=req.param_context)
        answer = synthesize(payload)
        return ChatResponse(answer=answer, payload=payload)

    except OutOfScopeError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "out_of_scope", "message": str(exc)},
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model or data not found: {exc}. Run train_anomaly.py first.",
        )
    except Exception as exc:
        logging.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/pipeline/raw", response_model=PipelineResponse)
def pipeline_raw(req: ChatRequest):
    """Return structured pipeline output without LLM synthesis."""
    from src.chat.intent import OutOfScopeError

    try:
        query_time = None
        if req.query_time:
            from datetime import datetime
            query_time = datetime.fromisoformat(req.query_time)

        payload = _route_and_run(req.question, req.machine_id, query_time,
                                 param_context=req.param_context)
        return PipelineResponse(payload=payload)

    except OutOfScopeError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": "out_of_scope", "message": str(exc)},
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model or data not found: {exc}. Run train_anomaly.py first.",
        )
    except Exception as exc:
        logging.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))
