import anthropic
import os
import io
import json
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import Optional
import uvicorn

try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

anthropic_client = anthropic.Anthropic()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

document_collections = {}

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def extract_text(file_content: bytes, filename: str) -> str:
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        if not PDF_SUPPORT:
            return "PDF support not available."
        reader = pypdf.PdfReader(io.BytesIO(file_content))
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    if filename_lower.endswith(".csv"):
        text = file_content.decode("utf-8", errors="ignore")
        lines = text.strip().split("\n")
        if not lines:
            return ""
        headers = lines[0].split(",")
        readable = []
        for line in lines[1:]:
            values = line.split(",")
            row_text = " | ".join([
                f"{h.strip()}: {v.strip()}"
                for h, v in zip(headers, values)
                if v.strip()
            ])
            if row_text:
                readable.append(row_text)
        return "\n".join(readable)

    return file_content.decode("utf-8", errors="ignore")

def index_document(session_id: str, text: str) -> int:
    collection_name = f"doc_{session_id}"

    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    chunks = chunk_text(text)
    if not chunks:
        return 0

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    document_collections[session_id] = collection
    return len(chunks)

def retrieve_chunks(session_id: str, query: str, n_results: int = 4) -> list:
    collection = document_collections.get(session_id)
    if not collection:
        return []
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]

def answer_question(session_id: str, question: str) -> dict:
    chunks = retrieve_chunks(session_id, question)
    if not chunks:
        return {
            "answer": "No document loaded for this session. Please upload a document first.",
            "source_chunks": [],
            "grounded": False
        }

    context = "\n\n---\n\n".join(chunks)

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I don't have that information in the uploaded document."
Do not use any outside knowledge.

Context:
{context}

Question: {question}"""
            }
        ]
    )

    answer = response.content[0].text
    grounded = "don't have that information" not in answer.lower()

    return {
        "answer": answer,
        "source_chunks": chunks,
        "grounded": grounded
    }

def run_eval(session_id: str, eval_questions: list) -> dict:
    results = []
    correct = 0

    for item in eval_questions:
        question = item.get("question", "")
        expected_keywords = item.get("expected_keywords", [])

        result = answer_question(session_id, question)
        answer = result["answer"].lower()

        keyword_hits = [kw for kw in expected_keywords if kw.lower() in answer]
        score = len(keyword_hits) / len(expected_keywords) if expected_keywords else 0
        passed = score >= 0.5

        if passed:
            correct += 1

        results.append({
            "question": question,
            "answer": result["answer"],
            "expected_keywords": expected_keywords,
            "keyword_hits": keyword_hits,
            "score": round(score, 2),
            "passed": passed,
            "grounded": result["grounded"]
        })

    accuracy = round(correct / len(eval_questions), 2) if eval_questions else 0

    return {
        "accuracy": accuracy,
        "passed": correct,
        "total": len(eval_questions),
        "results": results
    }

document_store = {}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = extract_text(content, file.filename)

        if not text.strip():
            return {"status": "error", "message": "Could not extract text from file."}

        import hashlib
        session_id = hashlib.md5(file.filename.encode()).hexdigest()[:8]
        chunk_count = index_document(session_id, text)

        document_store[session_id] = {
            "filename": file.filename,
            "char_count": len(text),
            "chunk_count": chunk_count,
            "preview": text[:300]
        }

        return {
            "status": "success",
            "session_id": session_id,
            "filename": file.filename,
            "chunks_indexed": chunk_count,
            "preview": text[:300],
            "message": f"Document indexed into {chunk_count} chunks. Ready for questions."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

class QuestionRequest(BaseModel):
    session_id: str
    question: str

class EvalRequest(BaseModel):
    session_id: str
    questions: list

@app.post("/ask")
@limiter.limit("20/minute")
async def ask(request: Request, body: QuestionRequest):
    result = answer_question(body.session_id, body.question)
    return result

@app.post("/eval")
@limiter.limit("5/minute")
async def eval_endpoint(request: Request, body: EvalRequest):
    if not body.questions:
        return {"status": "error", "message": "No eval questions provided."}
    result = run_eval(body.session_id, body.questions)
    return result

@app.get("/sessions")
async def sessions():
    return {"sessions": document_store}

@app.get("/status")
async def status():
    return {
        "status": "online",
        "model": "claude-sonnet-4-6",
        "active_documents": len(document_collections),
        "pdf_support": PDF_SUPPORT
    }

@app.get("/")
async def root():
    return {"status": "BYOD agent is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)