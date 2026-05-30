import anthropic
import os
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

anthropic_client = anthropic.Anthropic()
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.create_collection(
    name="company_docs",
    embedding_function=embedding_fn
)
app = FastAPI()

def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

def load_and_index_document(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks from {filepath}")
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB")

def retrieve_relevant_chunks(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]

def run_rag_agent(user_question):
    relevant_chunks = retrieve_relevant_chunks(user_question)
    context = "\n\n".join(relevant_chunks)
    messages = [
        {
            "role": "user",
            "content": f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {user_question}"""
        }
    ]
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=messages
    )
    return {
        "answer": response.content[0].text,
        "source_chunks": relevant_chunks
    }

print("Loading and indexing documents...")
load_and_index_document("company_faq.txt")
print("Ready!")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QuestionRequest):
    result = run_rag_agent(request.question)
    return result

@app.get("/")
def root():
    return {"status": "RAG agent is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
