# BYOD Agent — Bring Your Own Data RAG Agent

A production-grade RAG agent. Upload any document, ask questions, get answers grounded strictly in that document.

**Live:** https://byod-agent.onrender.com

---

## Architecture

### Indexing (upload time)
1. Extract text from PDF, TXT, or CSV
2. Chunk into overlapping word-count pieces (150 words, 30 word overlap)
3. Embed using Pinecone hosted multilingual-e5-large (1024 dims)
4. Store in Pinecone with session namespace for multi-tenant isolation
5. Build BM25 index from same chunks for keyword search

### Retrieval (query time)
1. Embed question using same Pinecone model
2. Vector search returns top 12 candidates by cosine similarity
3. BM25 scores same chunks by keyword match
4. Scores combined 60/40 (vector/BM25)
5. Pinecone bge-reranker-v2-m3 re-ranks top 8 candidates
6. Top 4 chunks passed to Claude as context

### Generation
- Claude Sonnet generates answer from retrieved context only
- Two-layer grounding check: string match then self-verification LLM call
- Conversation history loaded from Supabase, compressed after 6 turns

### Observability
- Langfuse traces every Claude call with tokens, latency, cost
- RAGAS offline eval: faithfulness, answer relevancy, context precision, context recall

---

## Stack

| Layer | Tool |
|---|---|
| LLM | Claude Sonnet + Haiku |
| Vector DB | Pinecone serverless |
| Embeddings | Pinecone multilingual-e5-large |
| Keyword search | BM25 via rank-bm25 |
| Re-ranking | Pinecone bge-reranker-v2-m3 |
| Observability | Langfuse v4 |
| Evals | RAGAS 0.1.21 |
| Memory | Supabase PostgreSQL |
| API | FastAPI + uvicorn |
| Deployment | Render free tier |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /upload | Upload PDF, TXT, or CSV |
| POST | /ask | Ask a question about uploaded document |
| POST | /eval | Run keyword eval against document |
| GET | /eval-questions/{session_id} | Get generated eval questions |
| GET | /status | Health check |

---

## Key Design Decisions

**Pinecone over ChromaDB:** ChromaDB is in-memory and dies on restart. Pinecone persists across restarts and keeps users isolated via namespaces.

**Hybrid search:** Vector search misses exact keyword matches. BM25 catches proper nouns, IDs, contract clauses. Combined they cover both semantic and precise queries.

**Re-ranking:** Cosine similarity ranks by topic similarity not answer quality. A cross-encoder re-ranker directly compares question and chunk for better precision.

**Pinecone hosted embeddings:** Local sentence-transformers need 400MB RAM. Render free tier has 512MB total. Offloading to Pinecone keeps memory under 200MB.

**History compression:** After 6 turns, older history summarized by Claude Haiku. Only summary plus last 2 turns passed as context. Keeps token costs predictable.

**Self-verification:** After generation, a second Haiku call checks if the answer is grounded in context. Only fires when the cheap string match passes. Catches hallucinations that string matching misses.

---

## Run Locally

```bash
git clone https://github.com/SamuraiKai1/ai-agents-portfolio
cd byod-agent
pip install -r requirements.txt
cp .env.example .env
uvicorn 08_byod_agent:app --port 8005 --reload
```

Required env vars: ANTHROPIC_API_KEY, PINECONE_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL, SUPABASE_URL, SUPABASE_SERVICE_KEY
