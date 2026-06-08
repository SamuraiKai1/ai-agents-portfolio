import httpx
from mcp.server.fastmcp import FastMCP

# FastMCP is the modern high-level interface for building MCP servers
# it handles all protocol details automatically
mcp = FastMCP("byod-mcp-server", host="127.0.0.1", port=8080)

BYOD_URL = "https://byod-agent.onrender.com"

@mcp.tool()
async def ask_document(session_id: str, question: str) -> str:
    """Ask a question about a previously uploaded document.
    
    Args:
        session_id: The session ID returned when the document was uploaded
        question: The question to ask about the document
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{BYOD_URL}/ask",
            json={"session_id": session_id, "question": question}
        )
        data = response.json()
        answer = data.get("answer", "No answer returned")
        grounded = data.get("grounded", False)
        return f"Answer: {answer}\nGrounded: {grounded}"

@mcp.tool()
async def upload_document(filename: str, content: str) -> str:
    """Upload a text document to make it queryable.
    
    Args:
        filename: Name of the document file
        content: Text content of the document
    """
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{BYOD_URL}/upload",
            files={"file": (filename, content.encode(), "text/plain")}
        )
        data = response.json()
        session_id = data.get("session_id", "unknown")
        chunks = data.get("chunks_indexed", 0)
        return f"Document uploaded. Session ID: {session_id}. Chunks indexed: {chunks}"

if __name__ == "__main__":
    print("Starting BYOD MCP server on http://localhost:8080")
    print("Tools: ask_document, upload_document")
    mcp.run(transport="streamable-http")
