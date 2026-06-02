import anthropic
import os
import json
from tavily import TavilyClient
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

anthropic_client = anthropic.Anthropic()
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_SPECIALISTS = 3

def search_web(query: str) -> str:
    try:
        result = tavily_client.search(query=query, max_results=3)
        return "\n\n".join([
            f"{r['title']}\n{r['content']}"
            for r in result["results"]
        ])
    except Exception as e:
        return f"Search failed: {str(e)}"

def manager_plan(task: str) -> List[dict]:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""You are a manager agent. Decompose this research task into 2-3 specialist roles.

Task: {task}

Return ONLY a JSON array, no other text:
[
  {{
    "role": "Short role name",
    "goal": "Specific focused goal in one sentence",
    "search_queries": ["query 1", "query 2"]
  }}
]

Rules:
- Maximum {MAX_SPECIALISTS} specialists
- Roles must be complementary, not overlapping
- Each role covers a distinct angle of the task
- search_queries must be specific and targeted, 2 queries per specialist"""
        }]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def run_specialist(role: str, goal: str, search_queries: List[str], task: str) -> dict:
    search_results = []
    tool_calls_log = []

    for query in search_queries[:2]:
        result = search_web(query)
        search_results.append(f"Search: {query}\n\n{result}")
        tool_calls_log.append({
            "type": "tool_call",
            "tool": "web_search",
            "input": {"query": query},
            "result": result[:500]
        })

    combined_research = "\n\n---\n\n".join(search_results)

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""You are a specialist agent with the role: {role}

Your goal: {goal}

Overall task: {task}

Here is your research gathered from web searches:

{combined_research}

Based on this research, provide:
1. Key findings (bullet points, specific facts and data)
2. Your analysis and conclusions
3. Confidence score 1-10 for your findings
4. Knowledge gaps (what you could not determine)

Be direct and specific. Only use information from the research above."""
        }]
    )

    conclusion = response.content[0].text

    return {
        "role": role,
        "goal": goal,
        "findings": tool_calls_log,
        "conclusion": conclusion
    }

def manager_synthesize(task: str, specialist_outputs: List[dict]) -> dict:
    specialist_summaries = "\n\n".join([
        f"## {s['role']}\nGoal: {s['goal']}\nFindings:\n{s['conclusion']}"
        for s in specialist_outputs
    ])

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""You are a manager agent synthesizing research from your specialist team.

Original task: {task}

Specialist outputs:
{specialist_summaries}

Write a comprehensive final report with:
1. Executive summary (2-3 sentences)
2. Key findings (one section per specialist with their role as heading)
3. Overall assessment
4. Knowledge gaps (what could not be determined)

Be direct, factual, and attribute findings to the specialist that discovered them.
Format with clear markdown headings."""
        }]
    )

    return {
        "report": response.content[0].text,
        "specialist_count": len(specialist_outputs),
        "roles": [s["role"] for s in specialist_outputs]
    }

def run_multi_agent(task: str, log_callback=None):
    def log(message: str, type: str = "info"):
        if log_callback:
            log_callback({"message": message, "type": type})

    log(f"Manager received task: '{task}'", "manager")
    log("Manager reasoning about task decomposition...", "manager")

    plan = manager_plan(task)
    plan = plan[:MAX_SPECIALISTS]

    log(f"Manager created plan with {len(plan)} specialist agents", "manager")
    for p in plan:
        log(f"Specialist role: {p['role']} — {p['goal']}", "plan")

    specialist_outputs = []

    for specialist in plan:
        role = specialist["role"]
        goal = specialist["goal"]
        queries = specialist.get("search_queries", [])

        log(f"Spawning specialist: {role}", "spawn")
        log(f"Goal: {goal}", "detail")

        for q in queries[:2]:
            log(f"{role} searching: {q}", "tool")

        output = run_specialist(role, goal, queries, task)

        log(f"{role} complete.", "complete")
        specialist_outputs.append(output)

    log("All specialists complete. Manager synthesizing final report...", "manager")
    result = manager_synthesize(task, specialist_outputs)
    log("Report ready.", "done")

    return {
        "task": task,
        "plan": plan,
        "specialist_outputs": specialist_outputs,
        "report": result["report"],
        "roles": result["roles"]
    }

session_results = {}

class TaskRequest(BaseModel):
    task: str
    session_id: Optional[str] = None

class FollowUpRequest(BaseModel):
    session_id: str
    question: str

@app.post("/run")
@limiter.limit("3/minute")
async def run(request: Request, body: TaskRequest):
    import uuid
    session_id = body.session_id or str(uuid.uuid4())[:8]
    logs = []

    def collect_log(entry):
        logs.append(entry)

    result = run_multi_agent(body.task, log_callback=collect_log)
    result["session_id"] = session_id
    result["logs"] = logs

    session_results[session_id] = {
        "task": body.task,
        "report": result["report"],
        "specialist_outputs": result["specialist_outputs"],
        "logs": logs
    }

    return result

@app.post("/followup")
@limiter.limit("10/minute")
async def followup(request: Request, body: FollowUpRequest):
    session = session_results.get(body.session_id)
    if not session:
        return {"error": "Session not found. Run a task first."}

    context = f"Original task: {session['task']}\n\nResearch report:\n{session['report']}"

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"{context}\n\nFollow-up question: {body.question}\n\nAnswer based only on the research above."
        }]
    )

    return {
        "answer": response.content[0].text,
        "session_id": body.session_id
    }

@app.get("/status")
async def status():
    return {
        "status": "online",
        "model": "claude-sonnet-4-6",
        "max_specialists": MAX_SPECIALISTS,
        "available_tools": ["web_search"],
        "active_sessions": len(session_results)
    }

@app.get("/")
async def root():
    return {"status": "Multi-agent system is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)