# AI Agents Portfolio

Three production AI agents built from scratch using Claude + FastAPI.

## Agents

### 1. Web Search Agent
An agent that answers questions using real-time web search.
- Tool calling, agent loop, runaway protection
- Live demo: https://web-search-agent-woa6.onrender.com

### 2. RAG Document Agent
An agent that answers questions grounded in private documents.
- Chunking, embeddings, ChromaDB vector search
- Live demo: https://rag-agent-31ns.onrender.com

### 3. Memory Agent
An agent that remembers context across conversation turns.
- Conversation memory, combined memory + retrieval
- Live demo: coming soon

## Stack
- LLM: Anthropic Claude Sonnet 4.6
- Embeddings: ChromaDB default (all-MiniLM-L6-v2 via ONNX)
- Vector DB: ChromaDB
- API: FastAPI + uvicorn
- Deployment: Render

## Run any agent locally

cd into the agent folder, then:

pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
export TAVILY_API_KEY=your-key  # web-search-agent only
python3 <agent_file>.py