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
from dotenv import load_dotenv

load_dotenv()

# LANGFUSE: this gives observability into every manager and specialist
# call below. observe() wraps a function as one traced span. get_client()
# gives access to update_current_span for adding custom metadata, like
# marking a span as failed with a reason, which is what makes failures
# visible in the Langfuse dashboard instead of only in our own logs.
from langfuse import observe, get_client

anthropic_client = anthropic.Anthropic()
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
langfuse = get_client()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_SPECIALISTS = 3


# FIX 1: search_web previously returned a plain string on failure,
# identical in type to a successful result. Nothing downstream could
# tell the difference without checking for the literal phrase "Search failed".
# Now it returns a dict with an explicit status field, so callers can
# check status programmatically instead of pattern matching text.
# LANGFUSE: @observe() traces every call to this function as its own span,
# so a failed search shows up clearly in the dashboard, not just in our logs.
@observe()
def search_web(query: str) -> dict:
    try:
        result = tavily_client.search(query=query, max_results=3)
        text = "\n\n".join([
            f"{r['title']}\n{r['content']}"
            for r in result["results"]
        ])
        return {"status": "success", "text": text}
    except Exception as e:
        langfuse.update_current_span(metadata={"status": "error", "error": str(e)})
        return {"status": "error", "text": f"Search failed: {str(e)}"}


# FIX 2: manager_plan had no error handling around json.loads. If Claude's
# response was not clean JSON, this would throw an unhandled exception and
# crash the entire request before any specialist had run, wasting nothing
# in cost terms but giving the user a 500 error with no explanation.
# Now it catches parse failures and raises a clear, named exception instead,
# so the caller in run_multi_agent can fail fast with a readable message.
class PlanningError(Exception):
    pass


@observe()
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

    try:
        plan = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        langfuse.update_current_span(metadata={"status": "error", "error": str(e), "raw_response": raw[:300]})
        raise PlanningError(f"Manager returned invalid JSON, could not parse plan: {str(e)}")

    if not isinstance(plan, list) or len(plan) == 0:
        langfuse.update_current_span(metadata={"status": "error", "error": "empty or malformed plan"})
        raise PlanningError("Manager returned an empty or malformed plan")

    return plan


# FIX 3: run_specialist had no error handling around its own LLM call, and
# run_multi_agent's loop had no error handling around run_specialist either.
# A single specialist failure used to crash the entire run, discarding any
# specialists that had already succeeded before it.
# Now each specialist gets one retry on failure. If it still fails after
# the retry, that failure is recorded as plain data, not raised as an
# exception, so the loop continues and the manager finds out about the
# gap explicitly later, at synthesis time.
@observe()
def run_specialist(role: str, goal: str, search_queries: List[str], task: str) -> dict:
    search_results = []
    tool_calls_log = []

    for query in search_queries[:2]:
        result = search_web(query)
        search_results.append(f"Search: {query}\n\n{result['text']}")
        tool_calls_log.append({
            "type": "tool_call",
            "tool": "web_search",
            "input": {"query": query},
            "status": result["status"],
            "result": result["text"][:500]
        })

    combined_research = "\n\n---\n\n".join(search_results)

    attempt = 0
    last_error = None
    while attempt < 2:
        try:
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
            return {
                "role": role,
                "goal": goal,
                "findings": tool_calls_log,
                "conclusion": response.content[0].text,
                "status": "success"
            }
        except Exception as e:
            last_error = str(e)
            attempt += 1

    # Both attempts failed. Return this as data, not an exception,
    # so the loop in run_multi_agent keeps going to the next specialist.
    langfuse.update_current_span(metadata={"status": "failed", "error": last_error, "role": role})
    return {
        "role": role,
        "goal": goal,
        "findings": tool_calls_log,
        "conclusion": None,
        "status": "failed",
        "error": last_error
    }


# FIX 4: manager_synthesize had no error handling around its own LLM call.
# Now it retries once on a transient failure (network issue, rate limit),
# and it now always runs regardless of whether any specialists failed.
# It receives explicit success/failure status for every specialist and is
# instructed to write around any gaps and name what is missing, rather than
# silently treating a failed specialist's empty conclusion as real data.
@observe()
def manager_synthesize(task: str, specialist_outputs: List[dict]) -> dict:
    specialist_summaries = "\n\n".join([
        f"## {s['role']} ({'SUCCESS' if s['status'] == 'success' else 'FAILED, no data available'})\n"
        f"Goal: {s['goal']}\n"
        f"Findings:\n{s['conclusion'] if s['status'] == 'success' else 'This specialist failed and produced no findings. Do not invent data for this role.'}"
        for s in specialist_outputs
    ])

    attempt = 0
    last_error = None
    while attempt < 2:
        try:
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
2. Key findings (one section per successful specialist with their role as heading)
3. Overall assessment
4. Knowledge gaps (what could not be determined). If any specialist failed,
   explicitly say which role failed and what information is therefore missing.

Be direct, factual, and attribute findings to the specialist that discovered them.
Format with clear markdown headings."""
                }]
            )
            return {
                "report": response.content[0].text,
                "specialist_count": len(specialist_outputs),
                "roles": [s["role"] for s in specialist_outputs],
                "failed_roles": [s["role"] for s in specialist_outputs if s["status"] == "failed"]
            }
        except Exception as e:
            last_error = str(e)
            attempt += 1

    langfuse.update_current_span(metadata={"status": "error", "error": last_error})
    raise RuntimeError(f"Manager synthesis failed after retry: {last_error}")


@observe()
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

        if output["status"] == "success":
            log(f"{role} complete.", "complete")
        else:
            log(f"{role} failed after retry: {output['error']}", "error")

        specialist_outputs.append(output)

    log("All specialists complete. Manager synthesizing final report...", "manager")
    result = manager_synthesize(task, specialist_outputs)

    if result["failed_roles"]:
        log(f"Report ready, with gaps noted for: {', '.join(result['failed_roles'])}", "done")
    else:
        log("Report ready.", "done")

    return {
        "task": task,
        "plan": plan,
        "specialist_outputs": specialist_outputs,
        "report": result["report"],
        "roles": result["roles"],
        "failed_roles": result["failed_roles"]
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

    # FIX 2 continued: manager_plan can now raise PlanningError if Claude's
    # plan was malformed. Catch it here and fail fast with a clear message,
    # before any specialist cost has been incurred.
    try:
        result = run_multi_agent(body.task, log_callback=collect_log)
    except PlanningError as e:
        return {"error": f"Planning failed: {str(e)}", "session_id": session_id, "logs": logs}
    except RuntimeError as e:
        return {"error": str(e), "session_id": session_id, "logs": logs}

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
