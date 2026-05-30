import anthropic
import os
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

anthropic_client = anthropic.Anthropic()
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.create_collection(name="company_docs", embedding_function=embedding_fn)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

COMPANY_DOCS = """
ACME Corp Employee FAQ

Q: What is the vacation policy?
A: Full-time employees receive 20 days of paid vacation per year. Vacation days reset on January 1st. Unused days cannot be carried over. Requests must be submitted at least 2 weeks in advance.

Q: How do I submit an expense report?
A: Expense reports must be submitted within 30 days. Use the Expensify app. Reports over $500 require manager approval. Reimbursements are processed within 2 weeks.

Q: What is the remote work policy?
A: Employees may work remotely up to 3 days per week. Core hours are 10am-3pm local time. Must be available on Slack during core hours.

Q: What health insurance plans are available?
A: Three plans: Basic (70% coverage), Standard (85% + dental), Premium (95% + dental + vision). Open enrollment every November.

Q: How does the performance review process work?
A: Reviews happen twice a year in June and December. Employees submit self-assessment two weeks before. Managers rate 1-5. Ratings of 4+ qualify for merit increase.

Q: What is the parental leave policy?
A: Primary caregivers get 16 weeks fully paid. Secondary caregivers get 6 weeks. Must be taken within 12 months of birth or adoption.
"""

def index_documents():
    words = COMPANY_DOCS.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + 80
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - 20
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
    print(f"Indexed {len(chunks)} chunks into ChromaDB")

def retrieve_context(query, n_results=2):
    results = collection.query(query_texts=[query], n_results=n_results)
    return "\n\n".join(results["documents"][0])

conversation_histories: Dict[str, List[dict]] = {}

def run_memory_agent(session_id: str, user_message: str) -> str:
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    history = conversation_histories[session_id]
    context = retrieve_context(user_message)
    system_prompt = f"""You are a helpful HR assistant for ACME Corp.
You have access to company policy documents. Use them to answer questions accurately.
If the answer is not in the documents, say so honestly.
Remember the full conversation history and refer back to it when relevant.

Relevant company policies:
{context}"""
    history.append({"role": "user", "content": user_message})
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=history
    )
    assistant_message = response.content[0].text
    history.append({"role": "assistant", "content": assistant_message})
    conversation_histories[session_id] = history
    return assistant_message

print("Indexing documents...")
index_documents()
print("Ready!")

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    turn_count: int

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    response = run_memory_agent(body.session_id, body.message)
    turn_count = len(conversation_histories.get(body.session_id, [])) // 2
    return ChatResponse(response=response, turn_count=turn_count)

@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    if session_id in conversation_histories:
        del conversation_histories[session_id]
    return {"status": "session cleared"}

@app.get("/")
async def root(request: Request):
    return {"status": "Memory agent is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)