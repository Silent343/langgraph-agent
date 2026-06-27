"""
api.py
======
FastAPI HTTP server that exposes the LangGraph agent as a REST API.

This is the entry point Railway runs in production (see Procfile).
It wraps the agent from agent.py in two endpoints:

  POST /chat          — send a message, get a response
  GET  /threads       — list all thread IDs stored in the database
  GET  /health        — Railway health check

The checkpointer (SQLite locally, PostgreSQL on Railway) is created once
at startup and shared across all requests via a module-level variable.
This avoids opening a new DB connection for every HTTP request.

Run locally:
    uvicorn api:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"thread_id": "test-1", "message": "How much is 200 USD in soles?"}'
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import build_graph, chat, get_checkpointer


# ---------------------------------------------------------------------------
# Shared state — checkpointer and app live for the lifetime of the server
# ---------------------------------------------------------------------------
# These are set once during startup (lifespan) and reused by every request.
_checkpointer_ctx = None
_checkpointer = None
_app = None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    FastAPI lifespan handler — runs startup and shutdown logic.

    On startup : opens the checkpointer (SQLite or Postgres) and builds
                 the compiled LangGraph app.
    On shutdown: closes the checkpointer connection cleanly.

    Using a lifespan instead of global startup/shutdown events is the
    recommended approach in FastAPI 0.93+.
    """
    global _checkpointer_ctx, _checkpointer, _app

    # Enter the checkpointer context (opens DB connection / pool)
    _checkpointer_ctx = get_checkpointer()
    _checkpointer = _checkpointer_ctx.__enter__(None)  # type: ignore[attr-defined]

    # Build the compiled graph once — reused for all requests
    _app = build_graph(_checkpointer)

    print("✅ Agent ready. Checkpointer:", type(_checkpointer).__name__)

    yield  # server is running

    # Shutdown: close DB connection / pool
    if _checkpointer_ctx is not None:
        _checkpointer_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
    print("🔴 Agent shutdown complete.")


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
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """
    Request body for POST /chat.

    Attributes:
        thread_id: Conversation identifier. Use the same ID across requests
                   to maintain memory. Use a new ID to start fresh.
        message:   The user's input text.

    Example:
        {"thread_id": "user-gabriel", "message": "How much is 100 USD in soles?"}
    """

    thread_id: str
    message: str


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.

    Attributes:
        thread_id: Echoed back so the client knows which thread was used.
        response:  The agent's final answer.
    """

    thread_id: str
    response: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Railway health check endpoint.

    Railway pings this URL periodically to verify the service is alive.
    Returns 200 OK with a simple JSON body.
    """
    return {"status": "ok", "agent": "langgraph-groq"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Send a message to the agent and receive a response.

    The agent uses the thread_id to load conversation history from the
    database, so it remembers previous turns within the same thread.

    Args:
        request: JSON body with thread_id and message fields.

    Returns:
        JSON with thread_id and the agent's response string.

    Raises:
        HTTPException 500: If the agent or LLM raises an unexpected error.

    Example request:
        POST /chat
        {"thread_id": "session-abc", "message": "How much is 500 USD in soles?"}

    Example response:
        {"thread_id": "session-abc", "response": "500 USD equals 1875.00 PEN."}
    """
    if _app is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet.")
    try:
        response = chat(_app, request.thread_id, request.message)
        return ChatResponse(thread_id=request.thread_id, response=response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/threads")
def list_threads():
    """
    List all conversation thread IDs stored in the database.

    Useful for debugging and for building a thread-picker UI.

    Returns:
        JSON with a list of thread ID strings.

    Example response:
        {"threads": ["user-gabriel", "demo-thread-a", "session-20260610"]}
    """
    if _checkpointer is None:
        return {"threads": []}

    try:
        # Both SqliteSaver and PostgresSaver expose .list_namespaces()
        # to enumerate stored thread IDs.
        namespaces = list(_checkpointer.list_namespaces(()))  # type: ignore[attr-defined]
        thread_ids = list({ns[0] for ns in namespaces if ns})
        return {"threads": sorted(thread_ids)}
    except Exception:
        # Graceful fallback — not all checkpointer versions expose this method
        return {"threads": [], "note": "Thread listing not available."}
