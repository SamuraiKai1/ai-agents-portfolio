# AI Agents Portfolio — Mustafa Sadree

Five production AI agents built from scratch using Claude + FastAPI.

Live portfolio: https://mustafa-ai-agents.netlify.app

## Agents

### 1. Web search agent
Answers questions using real-time web search via tool calling.
- Live: https://web-search-agent-s1ez.onrender.com

### 2. RAG document agent
Answers questions grounded in private documents via vector retrieval.
- Live: https://rag-agent-31ns.onrender.com

### 3. Memory agent
Remembers context across conversation turns, combined with RAG.
- Live: https://memory-agent-ngxi.onrender.com

### 4. Customer support agent
Connected to a live Supabase database. Looks up customers, manages tickets, writes changes back to the database in real time. Accepts CSV uploads.
- Live: https://support-agent-mprw.onrender.com
- Demo page: https://mustafa-ai-agents.netlify.app/support-agent.html

### 5. Bring-your-own-data agent
Upload any document (PDF, TXT, CSV) and ask questions grounded in it. Includes a built-in eval endpoint that scores answer accuracy against expected keywords.
- Live: https://byod-agent.onrender.com
- Eval diagram: https://mustafa-ai-agents.netlify.app/eval-diagram.html

## Stack
- LLM: Claude Sonnet 4.6
- Embeddings: ChromaDB default (all-MiniLM-L6-v2)
- Vector DB: ChromaDB
- Database: Supabase (PostgreSQL)
- API: FastAPI + uvicorn
- Rate limiting: slowapi
- Deployment: Render
- Frontend: Netlify

## Run any agent locally

cd into the agent folder, then:

pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
python3 <agent_file>.py