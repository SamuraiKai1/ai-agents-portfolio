import anthropic
import os
import json
from tavily import TavilyClient
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio

anthropic_client = anthropic.Anthropic()
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_SPECIALISTS = 3
MAX_TOOL_CALLS = 3

AVAILABLE_TOOLS = {
    "web_search": {
        "definition": {
            "name": "web_search",
            "description": "Search the web for current information on any topic.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        },
        "executor": lambda args: search_web(args["query"])
    },
    "extract_facts": {
        "definition": {
            "name": "extract_facts",
            "description": "Extract key facts and data points from a body of text.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to extract facts from"},
                    "focus": {"type": "string", "description": "What type of facts to focus on"}
                },
                "required": ["text", "focus"]
            }
        },
        "executor": lambda args: extract_facts(args["text"], args["focus"])
    },
    "score_fit": {
        "definition": {
            "name": "score_fit",
            "description": "Score how well something fits a set of criteria on a scale of 1-10.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "What is being scored"},
                    "criteria": {"type": "string", "description": "The criteria to score against"},
                    "evidence": {"type": "string", "description": "Evidence gathered so far"}
                },
                "required": ["subject", "criteria", "evidence"]
            }
        },
        "executor": lambda args: score_fit(args["subject"], args["criteria"], args["evidence"])
    },
    "summarize": {
        "definition": {
            "name": "summarize",
            "description": "Summarize a body of research into key points.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to summarize"},
                    "max_points": {"type": "integer", "description": "Maximum number of key points"}
                },
                "required": ["content"]
            }
        },
        "executor": lambda args: summarize_content(args["content"], args.get("max_points", 5))
    }
}

def search_web(query: str) -> str:
    try:
        result = tavily_client.search(query=query, max_results=3)
        return "\n\n".join([
            f"{r['title']}\n{r['content']}"
            for r in result["results"]
        ])
    except Exception as e:
        return f"Search failed: {str(e)}"

def extract_facts(text: str, focus: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Extract the most important facts about '{focus}' from this text. Return as a numbered list.\n\nText:\n{text[:2000]}"
        }]
    )
    return response.content[0].text

def score_fit(subject: str, criteria: str, evidence: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Score '{subject}' against these criteria: {criteria}\n\nEvidence:\n{evidence[:1500]}\n\nReturn a score from 1-10 with a one paragraph justification."
        }]
    )
    return response.content[0].text

def summarize_content(content: str, max_points: int = 5) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Summarize this into {max_points} key points:\n\n{content[:2000]}"
        }]
    )
    return response.content[0].text

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name in AVAILABLE_TOOLS:
        return AVAILABLE_TOOLS[tool_name]["executor"](tool_input)
    return f"Unknown tool: {tool_name}"

def manager_plan(task: str) -> List[dict]:
    tool_descriptions = "\n".join([
        f"- {name}: {info['definition']['description']}"
        for name, info in AVAILABLE_TOOLS.items()
    ])

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""You are a manager agent. Your job is to decompose a research task into specialist roles.

Available tools:
{tool_descriptions}

Task: {task}

Create a plan of exactly 2-3 specialist agents to complete this task.
Each specialist should have a focused role with 1-2 tools from the available list.

Return ONLY a JSON array, no other text:
[
  {{
    "role": "Short role name (e.g. Market researcher)",
    "goal": "Specific goal for this specialist in one sentence",
    "tools": ["tool_name_1", "tool_name_2"],
    "search_queries": ["query 1", "query 2"]
  }}
]

Rules:
- Maximum {MAX_SPECIALISTS} specialists
- Each specialist uses only tools from the available list
- search_queries are only needed for specialists using web_search
- Make roles complementary, not overlapping
- Think about what a hiring manager would find impressive"""
        }]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def run_specialist(role: str, goal: str, tools: List[str], search_queries: List[str], task: str) -> dict:
    tool_defs = [AVAILABLE_TOOLS[t]["definition"] for t in tools if t in AVAILABLE_TOOLS]

    initial_content = f"You are a specialist agent with the role: {role}\n\nYour goal: {goal}\n\nOverall task: {task}"
    if search_queries:
        initial_content += f"\n\nSuggested search queries to start with: {', '.join(search_queries)}"
    initial_content += "\n\nComplete your goal using the available tools. Be thorough but focused."

    messages = [{"role": "user", "content": initial_content}]

    tool_calls_made = 0
    findings = []

    while tool_calls_made < MAX_TOOL_CALLS:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tool_defs,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    findings.append({"type": "conclusion", "content": block.text})
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_made += 1
                    result = run_tool(block.name, block.input)
                    findings.append({
                        "type": "tool_call",
                        "tool": block.name,
                        "input": block.input,
                        "result": result[:500]
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

    conclusion = next((f["content"] for f in findings if f["type"] == "conclusion"), "No conclusion reached.")

    return {
        "role": role,
        "goal": goal,
        "findings": findings,
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
2. Key findings (one section per specialist, with their role as the heading)
3. Overall assessment
4. Knowledge gaps (what we could not determine from available sources)

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
        tools = specialist.get("tools", ["web_search"])
        queries = specialist.get("search_queries", [])

        log(f"Spawning specialist: {role}", "spawn")
        log(f"Goal: {goal}", "detail")

        output = run_specialist(role, goal, tools, queries, task)

        for finding in output["findings"]:
            if finding["type"] == "tool_call":
                log(f"{role} called {finding['tool']}: {str(finding['input'])[:80]}...", "tool")

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
        "max_tool_calls_per_specialist": MAX_TOOL_CALLS,
        "available_tools": list(AVAILABLE_TOOLS.keys()),
        "active_sessions": len(session_results)
    }

@app.get("/")
async def root():
    return {"status": "Multi-agent system is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)