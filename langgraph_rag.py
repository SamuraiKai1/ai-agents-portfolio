import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pinecone import Pinecone

load_dotenv("byod-agent/.env")

llm = ChatAnthropic(
    model="claude-haiku-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("byod-agent")

# --- STATE ---
# this is what gets passed between every node in the graph
# every node reads from state and writes back to state
class RAGState(TypedDict):
    question: str
    session_id: str
    chunks: List[str]
    answer: str
    grounded: bool
    attempts: int  # tracks how many times we have tried retrieval

# --- NODE 1: RETRIEVE ---
def retrieve_node(state: RAGState) -> RAGState:
    print(f"[retrieve] attempt {state['attempts'] + 1}")
    query = state["question"]

    query_embedding_response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    query_embedding = query_embedding_response.data[0]["values"]

    results = pinecone_index.query(
        vector=query_embedding,
        top_k=12,
        namespace=state["session_id"],
        include_metadata=True
    )
    candidates = [match["metadata"]["text"] for match in results["matches"]]

    reranked = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=candidates,
        top_n=4,
        return_documents=True
    )
    chunks = [item.document["text"] for item in reranked.data]

    # write retrieved chunks back to state
    return {**state, "chunks": chunks, "attempts": state["attempts"] + 1}

# --- NODE 2: GENERATE ---
def generate_node(state: RAGState) -> RAGState:
    print("[generate] calling Claude...")
    context = "\n\n---\n\n".join(state["chunks"])

    prompt = ChatPromptTemplate.from_template("""
You are a document assistant. Answer using ONLY the context below.
If the answer is not in the context, say exactly: "I don't have that information."

Context:
{context}

Question: {question}

Answer:""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})

    grounded = "i don't have that information" not in answer.lower()

    # write answer and grounded flag back to state
    return {**state, "answer": answer, "grounded": grounded}

# --- NODE 3: CHECK (conditional logic) ---
def check_node(state: RAGState) -> str:
    # this node returns a string that tells the graph where to go next
    if state["grounded"]:
        print("[check] answer is grounded, finishing")
        return "end"
    elif state["attempts"] >= 2:
        print("[check] max attempts reached, finishing anyway")
        return "end"
    else:
        print("[check] answer not grounded, retrying retrieval")
        return "retrieve"  # loop back to retrieve node

# --- BUILD THE GRAPH ---
graph = StateGraph(RAGState)

# add nodes
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

# add edges
graph.set_entry_point("retrieve")          # start here
graph.add_edge("retrieve", "generate")     # always go retrieve -> generate
graph.add_conditional_edges(              # after generate, decide where to go
    "generate",
    check_node,
    {
        "end": END,
        "retrieve": "retrieve"
    }
)

# compile the graph
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "question": "who founded Anthropic?",
        "session_id": "398e1f14",
        "chunks": [],
        "answer": "",
        "grounded": False,
        "attempts": 0
    })

    print(f"\nFinal Answer: {result['answer']}")
    print(f"Grounded: {result['grounded']}")
    print(f"Attempts: {result['attempts']}")
