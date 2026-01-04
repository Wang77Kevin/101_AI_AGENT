import os
import sys
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load env
load_dotenv()

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import get_retriever
from agent import build_agent

def run_evaluation():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found. Cannot run evaluation.")
        return

    print("🚀 Starting Evaluation...")

    # Initialize Gemini for Ragas
    # Ragas needs an LLM and Embeddings to evaluate the answers
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 1. Define Test Data
    # Questions based on our knowledge base (mcp_overview.md, ragas_overview.md)
    questions = [
        "What is the Model Context Protocol (MCP)?",
        "What are the main metrics in Ragas?",
        "How does MCP connect AI models to data?"
    ]
    
    ground_truths = [
        ["MCP is an open standard that enables connections between AI models and data sources, replacing fragmented integrations with a universal protocol."],
        ["The main metrics in Ragas include Faithfulness, Answer Relevancy, Context Precision, and Context Recall."],
        ["MCP uses a client-host-server architecture where hosts (like IDEs) connect to servers (data sources) via a standard protocol."]
    ]

    # 2. Collect Answers and Contexts
    answers = []
    contexts = []
    
    agent = build_agent()
    retriever = get_retriever()
    
    print("🤖 Generating answers...")
    for q in questions:
        # Get Answer from Agent
        inputs = {"messages": [("user", q)]}
        response = agent.invoke(inputs)
        answer = response["messages"][-1].content
        answers.append(answer)
        
        # Get Contexts directly from Retriever (to see what the agent *should* have seen)
        # Note: The agent uses the tool, but for Ragas 'context_recall' we want to see what the retriever yields directly
        docs = retriever.invoke(q)
        ctx = [d.page_content for d in docs]
        contexts.append(ctx)

    # 3. Create Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # 4. Run Ragas Evaluation
    # We need to explicitly pass the LLM/Embeddings to Ragas if not default
    
    print("📊 Calculating metrics (this uses Gemini API)...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n✅ Evaluation Results:")
    print(results)
    
    # Save to CSV
    df = results.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("💾 Detailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()
