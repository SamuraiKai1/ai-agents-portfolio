import anthropic
import os
import csv
import io
from supabase import create_client
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import uvicorn

anthropic_client = anthropic.Anthropic()
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

tools = [
    {
        "name": "get_customer",
        "description": "Look up a customer by email address.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer email address"}
            },
            "required": ["email"]
        }
    },
    {
        "name": "get_orders",
        "description": "Get all orders for a customer by their customer ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer UUID"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_tickets",
        "description": "Get all support tickets for a customer by their customer ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer UUID"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "update_ticket_status",
        "description": "Update the status of a support ticket. Valid statuses: open, in_progress, resolved, escalated.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "The ticket UUID"},
                "status": {"type": "string", "description": "New status: open, in_progress, resolved, escalated"},
                "notes": {"type": "string", "description": "Optional notes about the status change"}
            },
            "required": ["ticket_id", "status"]
        }
    },
    {
        "name": "create_ticket",
        "description": "Create a new support ticket for a customer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer UUID"},
                "order_id": {"type": "string", "description": "Optional order UUID"},
                "subject": {"type": "string", "description": "Brief description of the issue"},
                "priority": {"type": "string", "description": "Priority: low, normal, high"}
            },
            "required": ["customer_id", "subject"]
        }
    }
]

def write_log(session_id: str, tool_name: str, tool_input: dict, result: str):
    try:
        supabase.table("logs").insert({
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": str(tool_input),
            "result": result[:500],
            "created_at": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        print(f"Log write failed: {e}")

def run_tool(tool_name: str, tool_input: dict, session_id: str = "system") -> str:
    try:
        if tool_name == "get_customer":
            result = supabase.table("customers").select("*").eq("email", tool_input["email"]).execute()
            if not result.data:
                output = "No customer found with that email."
            else:
                output = str(result.data[0])

        elif tool_name == "get_orders":
            result = supabase.table("orders").select("*").eq("customer_id", tool_input["customer_id"]).execute()
            output = "No orders found." if not result.data else str(result.data)

        elif tool_name == "get_tickets":
            result = supabase.table("tickets").select("*").eq("customer_id", tool_input["customer_id"]).execute()
            output = "No tickets found." if not result.data else str(result.data)

        elif tool_name == "update_ticket_status":
            update_data = {"status": tool_input["status"]}
            if "notes" in tool_input:
                update_data["notes"] = tool_input["notes"]
            supabase.table("tickets").update(update_data).eq("id", tool_input["ticket_id"]).execute()
            output = f"Ticket updated to status: {tool_input['status']}"

        elif tool_name == "create_ticket":
            ticket_data = {
                "customer_id": tool_input["customer_id"],
                "subject": tool_input["subject"],
                "priority": tool_input.get("priority", "normal"),
                "status": "open"
            }
            if "order_id" in tool_input:
                ticket_data["order_id"] = tool_input["order_id"]
            result = supabase.table("tickets").insert(ticket_data).execute()
            output = f"Ticket created with ID: {result.data[0]['id']}"

        else:
            output = f"Unknown tool: {tool_name}"

    except Exception as e:
        output = f"Database error: {str(e)}"

    write_log(session_id, tool_name, tool_input, output)
    return output

conversation_histories: Dict[str, List[dict]] = {}

def run_support_agent(session_id: str, user_message: str) -> str:
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    history = conversation_histories[session_id]
    history.append({"role": "user", "content": user_message})

    system_prompt = """You are a helpful customer support agent for a SaaS company.
You have access to the customer database. You can look up customers, view their orders and tickets, update ticket statuses, and create new tickets.

When a customer contacts you:
1. Ask for their email if you don't have it
2. Look up their account
3. Help them with their issue using the available tools
4. Always confirm before updating or creating records

Be professional, empathetic, and efficient. Never show one customer another customer's data."""

    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=history
        )

        if response.stop_reason == "end_turn":
            assistant_message = response.content[0].text
            history.append({"role": "assistant", "content": assistant_message})
            conversation_histories[session_id] = history
            return assistant_message

        if response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Calling tool: {block.name} with input: {block.input}")
                    result = run_tool(block.name, block.input, session_id)
                    print(f"Tool result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            history.append({"role": "user", "content": tool_results})
            conversation_histories[session_id] = history

    return "I was unable to complete your request. Please try again."

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    turn_count: int

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    response = run_support_agent(body.session_id, body.message)
    turn_count = len(conversation_histories.get(body.session_id, [])) // 2
    return ChatResponse(response=response, turn_count=turn_count)

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        decoded = contents.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))

        created_customers = 0
        created_tickets = 0

        for row in reader:
            email = row.get("email", "").strip().lower()
            name = row.get("name", "Unknown").strip()
            issue = row.get("issue", "General inquiry").strip()
            priority = row.get("priority", "normal").strip().lower()

            if not email:
                continue

            existing = supabase.table("customers").select("id").eq("email", email).execute()
            if existing.data:
                customer_id = existing.data[0]["id"]
            else:
                new_customer = supabase.table("customers").insert({
                    "name": name,
                    "email": email,
                    "plan": "free"
                }).execute()
                customer_id = new_customer.data[0]["id"]
                created_customers += 1

            supabase.table("tickets").insert({
                "customer_id": customer_id,
                "subject": issue,
                "priority": priority,
                "status": "open"
            }).execute()
            created_tickets += 1

        return {
            "status": "success",
            "customers_created": created_customers,
            "tickets_created": created_tickets,
            "message": f"Loaded {created_tickets} tickets from CSV. Try asking the agent about any of the uploaded emails."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/reset")
async def reset_demo():
    try:
        supabase.table("logs").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        supabase.table("tickets").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        supabase.table("orders").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        supabase.table("customers").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        conversation_histories.clear()
        return {"status": "success", "message": "All demo data cleared. Ready for fresh data."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/logs")
async def get_logs():
    try:
        result = supabase.table("logs").select("*").order("created_at", desc=True).limit(50).execute()
        return {"logs": result.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/stats")
async def get_stats():
    try:
        customers = supabase.table("customers").select("id", count="exact").execute()
        open_tickets = supabase.table("tickets").select("id", count="exact").eq("status", "open").execute()
        escalated = supabase.table("tickets").select("id", count="exact").eq("status", "escalated").execute()
        resolved = supabase.table("tickets").select("id", count="exact").eq("status", "resolved").execute()
        in_progress = supabase.table("tickets").select("id", count="exact").eq("status", "in_progress").execute()
        return {
            "total_customers": customers.count,
            "open_tickets": open_tickets.count,
            "escalated_tickets": escalated.count,
            "resolved_tickets": resolved.count,
            "in_progress_tickets": in_progress.count,
            "active_sessions": len(conversation_histories)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/status")
async def get_status():
    try:
        supabase.table("customers").select("id").limit(1).execute()
        db_connected = True
    except:
        db_connected = False
    return {
        "status": "online",
        "database": "connected" if db_connected else "disconnected",
        "active_sessions": len(conversation_histories),
        "model": "claude-sonnet-4-6"
    }

@app.get("/")
async def root(request: Request):
    return {"status": "Support agent is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)