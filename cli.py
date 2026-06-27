"""
cli.py
======
Interactive command-line interface for the LangGraph + Gemini agent.

Features:
- Start or resume any named conversation thread.
- Type messages and receive agent responses in a REPL loop.
- List all threads stored in the SQLite memory database.
- Special commands to switch threads or exit.

Usage:
    python cli.py

Commands (during a session):
    /switch <thread_id>  — Switch to a different (or new) conversation thread.
    /threads             — List all threads saved in memory.db.
    /clear               — Start a new thread with a generated ID.
    /quit  or  /exit     — Exit the CLI.
"""

import sqlite3
import uuid
from datetime import datetime

from agent import DB_PATH, build_graph, chat_verbose
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------------------------
# Thread utilities
# ---------------------------------------------------------------------------
def list_threads(db_path: str) -> list[str]:
    """
    Return all distinct thread IDs stored in the SQLite checkpoint database.

    The SqliteSaver stores checkpoints in a table called `checkpoints`.
    Each row has a `thread_id` column that identifies the conversation.

    Args:
        db_path: Path to the SQLite file (e.g. 'memory.db').

    Returns:
        Sorted list of thread ID strings, or an empty list if none exist.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except sqlite3.OperationalError:
        # Table doesn't exist yet — no conversations stored.
        return []


def generate_thread_id() -> str:
    """
    Generate a unique thread ID based on the current timestamp.

    Returns:
        A string like 'session-20250610-143022'.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"session-{timestamp}"


# ---------------------------------------------------------------------------
# CLI REPL
# ---------------------------------------------------------------------------
def run_cli() -> None:
    """
    Start the interactive agent CLI.

    Opens (or creates) the SQLite memory database, builds the agent graph
    with the persistent checkpointer, then enters a read-eval-print loop
    where the user can chat with the agent.
    """
    print("\n" + "=" * 55)
    print("  LangGraph + Gemini Agent  |  Persistent Memory CLI")
    print("=" * 55)
    print("  Commands:")
    print("    /switch <id>  — switch/create thread")
    print("    /threads      — list saved threads")
    print("    /clear        — start fresh thread")
    print("    /quit         — exit")
    print("=" * 55)

    # Reuse one SqliteSaver for the entire CLI session.
    # `with` ensures the SQLite connection is released on exit.
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = build_graph(checkpointer)

        # Default thread for the session
        current_thread = generate_thread_id()
        print(f"\n  ▶ New thread started: {current_thread}\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                # Graceful exit on Ctrl+C or Ctrl+D
                print("\n  Goodbye!")
                break

            if not user_input:
                continue

            # ── Special commands ────────────────────────────────────────────
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd   = parts[0].lower()
                arg   = parts[1] if len(parts) > 1 else ""

                if cmd == "/quit" or cmd == "/exit":
                    print("  Goodbye!")
                    break

                elif cmd == "/threads":
                    saved = list_threads(DB_PATH)
                    if saved:
                        print(f"\n  Saved threads ({len(saved)}):")
                        for t in saved:
                            marker = " ◀ current" if t == current_thread else ""
                            print(f"    • {t}{marker}")
                    else:
                        print("  No threads saved yet.")
                    print()

                elif cmd == "/switch":
                    if not arg:
                        print("  Usage: /switch <thread_id>\n")
                    else:
                        current_thread = arg
                        print(f"  ▶ Switched to thread: {current_thread}\n")

                elif cmd == "/clear":
                    current_thread = generate_thread_id()
                    print(f"  ▶ New thread started: {current_thread}\n")

                else:
                    print(f"  Unknown command: {cmd}\n")

                continue

            # ── Normal message — send to agent ──────────────────────────────
            chat_verbose(app, current_thread, user_input)
            print()  # blank line for readability


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_cli()
