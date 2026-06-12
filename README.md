# AI Agents Portfolio — Mustafa Sadree

Six production AI agents built from scratch using Claude + FastAPI.

Live portfolio: https://samuraiKai1.github.io/ai-agents-portfolio

## Agents

### 1. Web search agent
Answers questions using real-time web search via tool calling.
- Live: https://web-search-agent-s1ez.onrender.com
- Demo: https://samuraiKai1.github.io/ai-agents-portfolio

### 2. RAG document agent
Answers questions grounded in private documents via vector retrieval.
- Live: https://rag-agent-31ns.onrender.com
- Demo: https://samuraiKai1.github.io/ai-agents-portfolio

### 3. Memory agent
Remembers context across conversation turns, combined with RAG.
- Live: https://memory-agent-ngxi.onrender.com
- Demo: https://samuraiKai1.github.io/ai-agents-portfolio

### 4. Customer support agent
Connected to a live Supabase database. Looks up customers, manages tickets, writes changes back to the database in real time. Accepts CSV uploads.
- Live: https://support-agent-mprw.onrender.com
- Demo page: https://samuraikai1.github.io/ai-agents-portfolio/support-agent.html

### 5. Bring-your-own-data agent
Upload any document (PDF, TXT, CSV) and ask questions grounded in it. Includes a built-in eval endpoint that scores answer accuracy against expected keywords.
- Live: https://byod-agent.onrender.com
- Demo: https://samuraikai1.github.io/ai-agents-portfolio/byod-agent.html
- Eval diagram: https://samuraikai1.github.io/ai-agents-portfolio/eval-diagram.html

### 6. Multi-agent system
Dynamic orchestration. Manager agent decomposes tasks, assembles specialist teams, delegates work, synthesizes reports.
- Live: https://multi-agent-4pr9.onrender.com
- Demo page: https://samuraiKai1.github.io/ai-agents-portfolio/multi-agent.html

## Stack
- LLM: Claude Sonnet / Haiku (Anthropic)
- Vector DB: Pinecone (serverless, hybrid search)
- Embeddings: Pinecone hosted multilingual-e5-large
- Keyword search: BM25 (rank-bm25)
- Re-ranking: Pinecone bge-reranker-v2-m3
- Observability: Langfuse v4
- Evals: RAGAS
- Orchestration: LangChain + LangGraph
- Workflow automation: n8n
- MCP: FastMCP (Streamable HTTP)
- Database: Supabase (PostgreSQL)
- API: FastAPI + uvicorn
- Deployment: Render
- Frontend: GitHub Pages

## Run any agent locally

cd into the agent folder, then:

pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
python3 <agent_file>.py