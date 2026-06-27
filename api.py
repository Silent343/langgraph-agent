"""
api.py
======
FastAPI HTTP server that exposes the LangGraph agent as a REST API.

This is the entry point Railway runs in production (see Procfile).
Endpoints:
  POST /chat     — send a message, get a response
  GET  /threads  — list all thread IDs stored in the database
  GET  /health   — Railway health check

Run locally:
    uvicorn api:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"thread_id": "test-1", "message": "How much is 200 USD in soles?"}'
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import DATABASE_URL, SQLITE_PATH, build_graph, chat


# ---------------------------------------------------------------------------
# Module-level singletons — shared across all requests
# ---------------------------------------------------------------------------
_checkpointer = None
_app = None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    FastAPI lifespan: opens the checkpointer on startup, closes on shutdown.

    Uses a manual enter/exit instead of `with` so we can hold the
    checkpointer open for the entire server lifetime (not just one request).
    """
    global _checkpointer, _app

    if DATABASE_URL:
        # ── Production: PostgreSQL (Railway injects DATABASE_URL) ───────────
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=10,
            kwargs={"autocommit": True},
        )
        checkpointer = PostgresSaver(pool)
        # Creates the checkpoints table if it doesn't exist (idempotent).
        checkpointer.setup()
        _checkpointer = checkpointer
        _app = build_graph(_checkpointer)
        print(f"✅ Agent ready — PostgreSQL checkpointer")

        yield  # server runs here

        pool.close()
        print("🔴 PostgreSQL pool closed.")

    else:
        # ── Local: SQLite ───────────────────────────────────────────────────
        from langgraph.checkpoint.sqlite import SqliteSaver

        # SqliteSaver is a sync context manager — use `with` normally.
        with SqliteSaver.from_conn_string(SQLITE_PATH) as checkpointer:
            _checkpointer = checkpointer
            _app = build_graph(_checkpointer)
            print(f"✅ Agent ready — SQLite checkpointer ({SQLITE_PATH})")

            yield  # server runs here

        print("🔴 SQLite connection closed.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LangGraph Agent API",
    description="Groq + LangGraph conversational agent with persistent memory.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """
    POST /chat request body.

    Attributes:
        thread_id: Conversation ID. Reuse the same ID to keep memory.
        message:   User's input text.
    """
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    """
    POST /chat response body.

    Attributes:
        thread_id: Echoed back from the request.
        response:  Agent's final answer.
    """
    thread_id: str
    response: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Railway health check — returns 200 OK when the server is running."""
    return {"status": "ok", "agent": "langgraph-groq"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Send a message to the agent and receive a response.

    The agent loads conversation history from the DB using thread_id,
    so it remembers previous turns within the same thread.

    Example:
        POST /chat
        {"thread_id": "gabriel", "message": "How much is 500 USD in soles?"}
        → {"thread_id": "gabriel", "response": "500 USD = 1875.00 PEN"}
    """
    if _app is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    try:
        response = chat(_app, request.thread_id, request.message)
        return ChatResponse(thread_id=request.thread_id, response=response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/threads")
def list_threads():
    """
    List all conversation thread IDs saved in the database.

    Useful for debugging or building a thread-picker UI.
    """
    if _checkpointer is None:
        return {"threads": []}
    try:
        namespaces = list(_checkpointer.list_namespaces(()))
        thread_ids = sorted({ns[0] for ns in namespaces if ns})
        return {"threads": thread_ids}
    except Exception:
        return {"threads": [], "note": "Thread listing not available."}
