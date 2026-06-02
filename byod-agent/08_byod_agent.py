import anthropic
from dotenv import load_dotenv
from langfuse import observe, get_client

load_dotenv()

langfuse = get_client()
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
document_store = {}
eval_question_store = {}

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
                for h, v in zip(headers, values) if v.strip()
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

@observe()
def generate_eval_questions(text_sample: str, all_chunks: list) -> list:
    prompt = f"""You are creating an evaluation set for a RAG agent.

Read this document excerpt and generate exactly 5 factual questions that can be answered from it.
For each question, provide 2-3 keywords that MUST appear in a correct answer.

Rules for keywords:
- Keywords must be exact words or short phrases that appear verbatim in the document
- Do not invent keywords — only use words actually present in the text
- Keywords should be specific facts, numbers, names, or terms

Return ONLY a JSON array, no other text:
[
  {{"question": "...", "expected_keywords": ["keyword1", "keyword2"]}},
  ...
]

Document excerpt:
{text_sample[:2000]}"""

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="eval-question-generation",
        model="claude-sonnet-4-6",
    ) as generation:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        generation.update(
            input=prompt,
            output=response.content[0].text,
            usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens}
        )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    questions = json.loads(raw.strip())

    verified = []
    all_text = " ".join(all_chunks).lower()
    for item in questions:
        verified_keywords = [
            kw for kw in item.get("expected_keywords", [])
            if kw.lower() in all_text
        ]
        if verified_keywords:
            verified.append({
                "question": item["question"],
                "expected_keywords": verified_keywords
            })

    return verified[:5]

@observe()
def answer_question(session_id: str, question: str) -> dict:
    chunks = retrieve_chunks(session_id, question)
    if not chunks:
        return {
            "answer": "No document loaded for this session. Please upload a document first.",
            "source_chunks": [],
            "grounded": False
        }

    context = "\n\n---\n\n".join(chunks)

    prompt_text = f"""You are a document assistant. Answer the question using ONLY the text in the context below.

Rules:
- Use only information explicitly stated in the context
- Do not infer, elaborate, or add outside knowledge
- Keep your answer concise and direct
- If the answer is not in the context, say: "I don't have that information in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="answer-question",
        model="claude-sonnet-4-6",
    ) as generation:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt_text}]
        )
        generation.update(
            input=prompt_text,
            output=response.content[0].text,
            usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens}
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

        not_found_phrases = [
            "i don't have that information",
            "not mentioned",
            "not in the context",
            "no information",
            "cannot find"
        ]
        answered = not any(phrase in answer for phrase in not_found_phrases)

        if expected_keywords:
            keyword_hits = [kw for kw in expected_keywords if kw.lower() in answer]
            score = len(keyword_hits) / len(expected_keywords)
            passed = score >= 0.5
        else:
            score = 1.0 if answered else 0.0
            passed = answered
            keyword_hits = []

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

class QuestionRequest(BaseModel):
    session_id: str
    question: str

class EvalRequest(BaseModel):
    session_id: str
    questions: Optional[list] = None

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

        all_chunks = list(document_collections[session_id].get()["documents"])

        print(f"Generating eval questions for session {session_id}...")
        try:
            eval_questions = generate_eval_questions(text, all_chunks)
            eval_question_store[session_id] = eval_questions
            print(f"Generated {len(eval_questions)} eval questions")
        except Exception as e:
            print(f"Eval generation failed: {e}")
            eval_question_store[session_id] = []

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
            "eval_questions_generated": len(eval_question_store.get(session_id, [])),
            "message": f"Document indexed into {chunk_count} chunks. Ready for questions."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ask")
@limiter.limit("20/minute")
async def ask(request: Request, body: QuestionRequest):
    result = answer_question(body.session_id, body.question)
    langfuse.flush()
    return result

@app.post("/eval")
@limiter.limit("5/minute")
async def eval_endpoint(request: Request, body: EvalRequest):
    if body.questions:
        questions = body.questions
    else:
        questions = eval_question_store.get(body.session_id, [])

    if not questions:
        return {"status": "error", "message": "No eval questions available. Upload a document first."}

    result = run_eval(body.session_id, questions)
    return result

@app.get("/eval-questions/{session_id}")
async def get_eval_questions(session_id: str):
    questions = eval_question_store.get(session_id, [])
    return {"session_id": session_id, "questions": questions}

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