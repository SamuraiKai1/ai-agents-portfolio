import anthropic
from dotenv import load_dotenv
from langfuse import observe, get_client
import os
import time
import random
from supabase import create_client

load_dotenv()

langfuse = get_client()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
import io
import json
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
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

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
pinecone_index = pc.Index('byod-agent')
document_namespaces = {}
bm25_store = {}  # stores bm25 index per session
chunks_store = {}  # stores raw chunks per session for bm25 retrieval
document_store = {}
eval_question_store = {}


def with_backoff(fn, max_retries=3):
    """calls fn with exponential backoff on failure"""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # out of retries, raise the error
            wait = (2 ** attempt) + random.uniform(0, 1)  # 1s, 2s, 4s with jitter
            print(f"[backoff] attempt {attempt + 1} failed: {e}. retrying in {wait:.1f}s")
            time.sleep(wait)

def save_memory(session_id: str, role: str, content: str):
    """persists a message to supabase so it survives server restarts"""
    try:
        supabase.table("agent_memory").insert({
            "session_id": session_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        print(f"[memory] failed to save: {e}")

def load_memory(session_id: str) -> list:
    """loads conversation history from supabase for this session"""
    try:
        result = supabase.table("agent_memory")\
            .select("role, content")\
            .eq("session_id", session_id)\
            .order("created_at")\
            .execute()
        return result.data
    except Exception as e:
        print(f"[memory] failed to load: {e}")
        return []


def summarize_history(session_id: str, history: list) -> list:
    """when conversation history exceeds 6 turns, summarize older turns
    and keep only the summary plus the last 2 turns"""
    if len(history) <= 6:
        return history
    
    # split: old turns to summarize, recent turns to keep
    old_turns = history[:-2]
    recent_turns = history[-2:]
    
    # format old turns for summarization
    history_text = "\n".join([
        f"{turn['role'].upper()}: {turn['content']}"
        for turn in old_turns
    ])
    
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Summarize this conversation in 2-3 sentences, capturing key facts discussed:\n\n{history_text}"
        }]
    )
    
    summary = response.content[0].text
    
    # return summary as a system message plus recent turns
    return [
        {"role": "user", "content": f"[Previous conversation summary: {summary}]"},
        {"role": "assistant", "content": "Understood. I have context from our previous conversation."}
    ] + recent_turns


def verify_answer(question: str, context: str, answer: str) -> bool:
    """second llm call to verify the answer is grounded in the context
    only runs when the cheap string match passes"""
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5",  # cheap model for verification
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"""Does this answer contain ONLY information from the context below?
Reply with YES or NO only.

Context: {context}

Question: {question}

Answer: {answer}

Grounded (YES/NO):"""
        }]
    )
    verdict = response.content[0].text.strip().upper()
    return verdict.startswith("YES")

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
    chunks = chunk_text(text)
    if not chunks:
        return 0
    # embed all chunks into vectors using local sentence-transformers model
    embedding_response = pc.inference.embed(
        model='multilingual-e5-large',
        inputs=chunks,
        parameters={'input_type': 'passage'}
    )
    embeddings = [e['values'] for e in embedding_response.data]
    # build pinecone upsert payload: each vector needs an id, the embedding, and metadata
    vectors = [
        {
            "id": f"{session_id}_chunk_{i}",
            "values": embeddings[i],
            "metadata": {"text": chunks[i], "session_id": session_id}
        }
        for i in range(len(chunks))
    ]
    # upsert into pinecone under a namespace per session so sessions don't mix
    with_backoff(lambda: pinecone_index.upsert(vectors=vectors, namespace=session_id))
    document_namespaces[session_id] = True
    # build bm25 index from same chunks for keyword search
    # tokenize each chunk into words so bm25 can count term frequencies
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25_store[session_id] = BM25Okapi(tokenized_chunks)
    chunks_store[session_id] = chunks  # keep raw chunks for retrieval
    return len(chunks)

def retrieve_chunks(session_id: str, query: str, n_results: int = 4) -> list:
    if session_id not in document_namespaces:
        return []

    # --- vector search ---
    query_embedding_response = pc.inference.embed(
        model='multilingual-e5-large',
        inputs=[query],
        parameters={'input_type': 'query'}
    )
    query_embedding = query_embedding_response.data[0]['values']
    vector_results = with_backoff(lambda: pinecone_index.query(
        vector=query_embedding,
        top_k=n_results * 3,
        namespace=session_id,
        include_metadata=True
    ))
    # build a dict of chunk_text -> vector score
    vector_scores = {
        match["metadata"]["text"]: match["score"]
        for match in vector_results["matches"]
    }

    # --- bm25 keyword search ---
    bm25 = bm25_store.get(session_id)
    all_chunks = chunks_store.get(session_id, [])
    tokenized_query = query.lower().split()
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    # normalize bm25 scores to 0-1 range so they are comparable to cosine scores
    bm25_max = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_scores = {
        all_chunks[i]: bm25_scores_raw[i] / bm25_max
        for i in range(len(all_chunks))
    }

    # --- combine scores: 60% vector, 40% bm25 ---
    all_chunk_texts = set(vector_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for chunk in all_chunk_texts:
        v_score = vector_scores.get(chunk, 0)
        b_score = bm25_scores.get(chunk, 0)
        combined[chunk] = 0.6 * v_score + 0.4 * b_score

    # sort by combined score and take top candidates for re-ranking
    top_candidates = sorted(combined, key=combined.get, reverse=True)[:n_results * 2]

    if not top_candidates:
        return []

    # --- re-rank the combined candidates ---
    reranked = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=top_candidates,
        top_n=n_results,
        return_documents=True
    )
    return [item.document['text'] for item in reranked.data]

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

    # load and compress conversation history from supabase
    raw_history = load_memory(session_id)
    history = summarize_history(session_id, raw_history)

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
            messages=history + [{"role": "user", "content": prompt_text}]
        )
        generation.update(
            input=prompt_text,
            output=response.content[0].text,
            usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens}
        )

    answer = response.content[0].text
    grounded = "don't have that information" not in answer.lower()
    # only run expensive self-verification if cheap check passes
    if grounded:
        grounded = verify_answer(question, context, answer)

    # persist this turn to supabase so memory survives restarts
    save_memory(session_id, "user", question)
    save_memory(session_id, "assistant", answer)

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

        all_chunks = chunk_text(text)

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
        "active_documents": len(document_namespaces),
        "pdf_support": PDF_SUPPORT
    }

@app.get("/")
async def root():
    return {"status": "BYOD agent is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)