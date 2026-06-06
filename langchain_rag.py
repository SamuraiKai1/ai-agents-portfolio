import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

load_dotenv("byod-agent/.env")

# --- 1. CHAT MODEL ---
# swap this one line to change LLM provider
llm = ChatAnthropic(
    model="claude-haiku-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# --- 2. PROMPT TEMPLATE ---
# variables in curly braces get filled at runtime
prompt = ChatPromptTemplate.from_template("""
You are a document assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""")

# --- 3. OUTPUT PARSER ---
# extracts plain text from the LLM response object
parser = StrOutputParser()

# --- 4. RETRIEVER (manual, using our Pinecone + BM25 setup) ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("byod-agent")

def retrieve(query: str, session_id: str, n_results: int = 4) -> str:
    # embed query using pinecone hosted model
    query_embedding_response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    query_embedding = query_embedding_response.data[0]["values"]

    # vector search
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=n_results * 3,
        namespace=session_id,
        include_metadata=True
    )
    candidates = [match["metadata"]["text"] for match in results["matches"]]

    if not candidates:
        return "No document found for this session."

    # rerank
    reranked = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=candidates,
        top_n=n_results,
        return_documents=True
    )
    chunks = [item.document["text"] for item in reranked.data]
    return "\n\n---\n\n".join(chunks)

# --- 5. CHAIN ---
# this is the langchain way: pipe components together with |
# RunnablePassthrough passes the question through unchanged
def build_chain(session_id: str):
    return (
        {
            "context": lambda x: retrieve(x["question"], session_id),
            "question": RunnablePassthrough() | (lambda x: x["question"])
        }
        | prompt
        | llm
        | parser
    )

if __name__ == "__main__":
    session_id = "398e1f14"
    question = "who founded Anthropic?"

    print(f"Question: {question}")
    print("Building chain...")
    chain = build_chain(session_id)

    print("Running chain...")
    answer = chain.invoke({"question": question})
    print(f"\nAnswer: {answer}")
