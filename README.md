# LangGraph + Groq Agent with Persistent Memory

Conversational agent built with **LangGraph** + **Groq (Llama 3)**, featuring
persistent memory and a **FastAPI REST API** ready to deploy on **Railway**.

---

## Architecture

```
HTTP Client
    │
    ▼
FastAPI (api.py)          ← Railway runs this
    │
    ▼
LangGraph Graph (agent.py)
    │  ├── agent node  →  Groq / Llama 3.1
    │  └── tools node  →  calculate(), exchange_rate()
    │
    ▼
Checkpointer
    ├── LOCAL   → SQLite  (memory.db)
    └── RAILWAY → PostgreSQL  (DATABASE_URL)
```

---

## Project structure

```
langgraph-gemini-agent/
├── agent.py          # Core: state, tools, graph, checkpointer factory
├── api.py            # FastAPI HTTP server (Railway entry point)
├── cli.py            # Interactive REPL for local testing
├── Procfile          # Railway start command
├── runtime.txt       # Python version pin for Railway
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Local setup

```bash
# 1. Create virtualenv
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY from https://console.groq.com

# 4a. Run the demo script (multi-turn memory test)
python agent.py

# 4b. Run the interactive CLI
python cli.py

# 4c. Run the HTTP API locally
uvicorn api:app --reload --port 8000
```

---

## Deploy to Railway

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "feat: langgraph agent with persistent memory"
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Create Railway project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Select **Deploy from GitHub repo** → choose your repo
3. Railway auto-detects the `Procfile` and starts the build

### Step 3 — Add PostgreSQL plugin

1. In your Railway project → click **+ New**
2. Select **Database → Add PostgreSQL**
3. Railway automatically injects `DATABASE_URL` into your service's environment
4. The agent switches from SQLite to PostgreSQL automatically on next deploy

### Step 4 — Add environment variables

In Railway dashboard → your service → **Variables** tab:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | your key from console.groq.com |

`DATABASE_URL` is injected automatically — do NOT add it manually.

### Step 5 — Done

Railway builds and deploys. You get a public URL like:
```
https://your-agent.up.railway.app
```

---

## API endpoints

### `GET /health`
Railway health check. Returns `{"status": "ok"}`.

### `POST /chat`
Send a message to the agent.

```bash
curl -X POST https://your-agent.up.railway.app/chat \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "user-gabriel", "message": "How much is 500 USD in soles?"}'
```

Response:
```json
{
  "thread_id": "user-gabriel",
  "response": "500 USD is equivalent to 1875.00 Peruvian soles."
}
```

Send a follow-up with the **same thread_id** — the agent remembers:
```bash
curl -X POST https://your-agent.up.railway.app/chat \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "user-gabriel", "message": "What about EUR? Same amount."}'
```

### `GET /threads`
List all saved conversation threads.

```bash
curl https://your-agent.up.railway.app/threads
```

---

## Memory: how thread_id works

```
Same thread_id  →  agent loads history from DB  →  remembers context
New thread_id   →  fresh conversation, no prior memory
```

Locally stored in `memory.db` (SQLite).
On Railway stored in PostgreSQL — survives redeploys and restarts.

---

## Adding new tools

```python
# In agent.py — add a new @tool function and register it
@tool
def get_sunat_rate(date: str) -> str:
    """Fetch the official SUNAT exchange rate for a given date."""
    # call SUNAT API here
    ...

tools = [calculate, exchange_rate, get_sunat_rate]  # add here
```

Gemini/Llama will automatically discover and use the new tool.

---

## Tech stack

| Component | Library |
|---|---|
| Agent graph | `langgraph` |
| LLM | `langchain-groq` (Llama 3.1 8B) |
| Memory (local) | `langgraph-checkpoint-sqlite` |
| Memory (prod) | `langgraph-checkpoint-postgres` |
| HTTP API | `fastapi` + `uvicorn` |
| DB driver | `psycopg` + `psycopg-pool` |
