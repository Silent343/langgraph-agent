"""
agent.py
========
LangGraph + Groq (Llama 3) agent with persistent memory.

Checkpointer strategy:
  - LOCAL  (default): SqliteSaver  → writes to memory.db on disk.
  - RAILWAY (prod):   PostgresSaver → reads DATABASE_URL env var injected
                      automatically by Railway when you add a Postgres plugin.

The rest of the code (State, tools, graph) is identical in both environments.
The entry point (if __name__ == "__main__") runs a demo with multi-turn memory.
For the HTTP API used by Railway, see api.py.

Flow:
    START → agent → (tool call?) → tools → agent → ... → END

Setup (local):
    1. pip install -r requirements.txt
    2. cp .env.example .env  →  fill in GROQ_API_KEY
    3. python agent.py

Setup (Railway):
    See README.md → Deploy to Railway section.
"""

import os
from contextlib import contextmanager
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv()

# ---------------------------------------------------------------------------
# Config — read from environment variables
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.\n"
        "Local : add it to your .env file.\n"
        "Railway: add it in the Railway dashboard → Variables tab."
    )

# DATABASE_URL is injected automatically by Railway when you add a
# Postgres plugin to your project. Locally it is not set, so we fall
# back to SQLite.
DATABASE_URL = os.environ.get("DATABASE_URL")

# Local SQLite fallback path
SQLITE_PATH = "memory.db"


# ---------------------------------------------------------------------------
# Checkpointer factory
# ---------------------------------------------------------------------------
@contextmanager
def get_checkpointer():
    """
    Context manager that returns the right checkpointer for the environment.

    - If DATABASE_URL is set  → PostgresSaver (Railway / production).
    - Otherwise               → SqliteSaver   (local development).

    Usage:
        with get_checkpointer() as checkpointer:
            app = build_graph(checkpointer)
            ...

    Why a context manager?
        Both SqliteSaver and PostgresSaver need to open/close a connection.
        Wrapping them in a context manager guarantees the connection is always
        released, even if an exception is raised.
    """
    if DATABASE_URL:
        # ── Production: PostgreSQL via Railway ──────────────────────────────
        # PostgresSaver stores checkpoints in a `checkpoints` table that it
        # creates automatically on first run (checkpointer.setup()).
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        # ConnectionPool keeps a set of reusable DB connections open so we
        # don't pay the connection-setup cost on every agent invocation.
        pool = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=10,         # max concurrent connections
            kwargs={"autocommit": True},
        )
        checkpointer = PostgresSaver(pool)
        # Creates the `checkpoints` table if it doesn't exist yet.
        # Safe to call on every startup — it's idempotent.
        checkpointer.setup()
        try:
            yield checkpointer
        finally:
            pool.close()
    else:
        # ── Local: SQLite ───────────────────────────────────────────────────
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(SQLITE_PATH) as checkpointer:
            yield checkpointer


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    """
    Shared state that flows through every node in the graph.

    Attributes:
        messages: Full conversation history for the current thread.
                  The `add_messages` reducer appends incoming messages
                  instead of overwriting the list, preserving history.
    """

    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def calculate(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression and return the result.

    Supported operators: +, -, *, /, ** (power), // (floor division), % (mod).
    Do NOT pass variable names or function calls — numbers and operators only.

    Args:
        expression: Math expression string, e.g. '150 * 3.75' or '1000 / 4'.

    Returns:
        Numeric result formatted to 2 decimal places, or an error message.

    Examples:
        calculate('200 * 3.75') → '750.00'
        calculate('1000 / 4')   → '250.00'
    """
    # Restrict builtins to prevent arbitrary code execution via eval.
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"{result:.2f}"
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def exchange_rate(currency: str) -> str:
    """
    Return the approximate exchange rate of a currency vs the Peruvian Sol (PEN).

    Supported currencies: USD, EUR, GBP, JPY, BTC.

    Args:
        currency: ISO currency code (case-insensitive), e.g. 'USD', 'eur'.

    Returns:
        Rate string like '1 USD = 3.75 PEN', or an error listing valid codes.

    Examples:
        exchange_rate('USD') → '1 USD = 3.75 PEN'
        exchange_rate('btc') → '1 BTC = 340000.00 PEN'
    """
    rates: dict[str, float] = {
        "USD": 3.75,
        "EUR": 4.10,
        "GBP": 4.80,
        "JPY": 0.025,
        "BTC": 340_000.0,
    }
    code = currency.upper().strip()
    if code in rates:
        return f"1 {code} = {rates[code]:.2f} PEN"
    return f"Currency '{code}' not supported. Available: {', '.join(rates)}"


# All tools exposed to the LLM.
tools = [calculate, exchange_rate]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Llama 3.1 8B via Groq — fast, free tier, no regional quota issues.
# Swap `model` to upgrade:
#   "llama-3.1-8b-instant"    → fastest / cheapest  (dev)
#   "llama-3.3-70b-versatile" → smarter / slower    (prod)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,
)

# IMPORTANT — parallel_tool_calls=False:
# Llama on Groq does NOT support calling multiple tools simultaneously.
# Without this flag the model emits parallel tool_calls and Groq returns:
#   400 "Failed to call a function. Please adjust your prompt."
# This flag forces one tool call per loop iteration — slightly slower
# but 100% reliable.
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def agent_node(state: State) -> dict:
    """
    Primary node: calls the LLM with the full message history.

    The LLM either:
      (a) emits tool_calls → graph routes to tool_node
      (b) produces a final text answer → graph routes to END
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ToolNode reads tool_calls from the last AIMessage, runs each function,
# and appends ToolMessage results back to the state automatically.
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------
def should_continue(state: State) -> str:
    """
    Decide the next node after agent_node.

    Returns:
        'tools' — LLM wants to call a tool.
        'end'   — LLM produced a final answer; terminate the graph.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph(checkpointer) -> object:
    """
    Build and compile the LangGraph StateGraph with a given checkpointer.

    Graph topology:
        START → [agent] → (tool calls?) → [tools] → [agent] → ... → END

    Args:
        checkpointer: SqliteSaver (local) or PostgresSaver (Railway).
                      Persists every node execution so memory survives restarts.

    Returns:
        Compiled LangGraph app ready to invoke.
    """
    graph = StateGraph(State)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------
def chat(app, thread_id: str, user_message: str) -> str:
    """
    Send a message and return the agent's final response (silent).

    Args:
        app:          Compiled LangGraph app.
        thread_id:    Conversation ID. Same ID = agent remembers prior turns.
        user_message: User input text.

    Returns:
        Agent's final response string.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content


def chat_verbose(app, thread_id: str, user_message: str) -> str:
    """
    Same as chat() but prints tool calls and results to stdout.
    Useful for local development and debugging.

    Args:
        app:          Compiled LangGraph app.
        thread_id:    Conversation ID.
        user_message: User input text.

    Returns:
        Agent's final response string.
    """
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'=' * 55}")
    print(f"  Thread : {thread_id}")
    print(f"  User   : {user_message}")
    print(f"{'=' * 55}")

    result = app.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )

    for msg in result["messages"]:
        if type(msg).__name__ == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  🔧 Tool call  : {tc['name']}({tc['args']})")
        elif type(msg).__name__ == "ToolMessage":
            print(f"  📊 Tool result: {msg.content}")

    final = result["messages"][-1].content
    print(f"  🤖 Agent      : {final}")
    return final


# ---------------------------------------------------------------------------
# Entry point — local demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Runs a multi-turn memory demo using the appropriate checkpointer.

    Observe:
      - Turn 2 references the amount from Turn 1 without repeating it.
      - Thread B has no memory of Thread A (isolated conversations).
    """
    with get_checkpointer() as checkpointer:
        app = build_graph(checkpointer)

        THREAD_A = "demo-thread-a"
        THREAD_B = "demo-thread-b"

        print("\n=== THREAD A — Turn 1 ===")
        chat_verbose(app, THREAD_A, "How much is 500 USD in Peruvian soles?")

        print("\n=== THREAD A — Turn 2 (agent remembers 500 USD) ===")
        chat_verbose(app, THREAD_A, "And if I convert my budget to EUR instead? Same amount.")

        print("\n=== THREAD A — Turn 3 (still remembers context) ===")
        chat_verbose(app, THREAD_A, "Which conversion gave me more local currency?")

        print("\n=== THREAD B — Fresh session, no memory of Thread A ===")
        chat_verbose(app, THREAD_B, "What was the amount I was converting earlier?")
