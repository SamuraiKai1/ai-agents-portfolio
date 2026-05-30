import anthropic
import os
from tavily import TavilyClient
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import uvicorn

anthropic_client = anthropic.Anthropic()
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information on any topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
]

def run_agent(user_question: str) -> str:
    messages = [{"role": "user", "content": user_question}]
    max_iterations = 10
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Calling tool: {block.name} with input: {block.input}")
                    search_result = tavily_client.search(
                        query=block.input["query"],
                        max_results=3
                    )
                    result_text = "\n\n".join(
                        [f"{r['title']}\n{r['content']}" for r in search_result["results"]]
                    )
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": block.id, "content": result_text}]
                    })
    return "Agent could not complete the task within the allowed iterations."

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, body: QuestionRequest):
    answer = run_agent(body.question)
    return {"answer": answer}

@app.get("/")
async def root(request: Request):
    return {"status": "Agent is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)