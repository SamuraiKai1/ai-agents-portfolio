import os
import json
import requests
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_anthropic import ChatAnthropic

load_dotenv("byod-agent/.env")

# point ragas at claude instead of openai
llm = ChatAnthropic(
    model="claude-haiku-4-5",  # haiku is cheapest for eval
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# override ragas default llm
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

BASE_URL = "https://byod-agent.onrender.com"

def run_ragas_eval():
    print("Step 1: Uploading test document...")
    with open("byod-agent/test_doc.txt", "rb") as f:
        upload_response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("test_doc.txt", f, "text/plain")}
        )
    upload_data = upload_response.json()
    session_id = upload_data["session_id"]
    print(f"Session ID: {session_id}")

    print("Step 2: Getting eval questions...")
    questions_response = requests.get(f"{BASE_URL}/eval-questions/{session_id}")
    eval_questions = questions_response.json()["questions"]
    print(f"Found {len(eval_questions)} eval questions")

    print("Step 3: Asking each question and collecting answers + chunks...")
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in eval_questions[:3]:  # limit to 3 to save credits
        question = item["question"]
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"session_id": session_id, "question": question}
        )
        data = response.json()
        questions.append(question)
        answers.append(data["answer"])
        contexts.append(data["source_chunks"])
        ground_truths.append(" ".join(item["expected_keywords"]))
        print(f"  Q: {question[:60]}...")

    print("Step 4: Running RAGAS evaluation...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print("\n=== RAGAS EVAL RESULTS ===")
    print(f"Faithfulness:      {results['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['context_precision']:.3f}")
    print(f"Context Recall:    {results['context_recall']:.3f}")
    print("==========================")

if __name__ == "__main__":
    run_ragas_eval()
