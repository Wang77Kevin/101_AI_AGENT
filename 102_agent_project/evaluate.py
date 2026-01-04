import os
import sys
from dotenv import load_dotenv
from datasets import Dataset
from langsmith import Client
from langchain_core.tracers.context import collect_runs
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load env
if not load_dotenv():
    load_dotenv("../.env")

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
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 1. Define Test Data
    # Questions based on our knowledge base (mcp_overview.md, ragas_overview.md)
    questions = [
        "What is the Model Context Protocol (MCP)?",
        "What are the main metrics in Ragas?",
        "How does MCP connect AI models to data?"
    ]
    
    ground_truths = [
        "MCP is an open standard that enables connections between AI models and data sources, replacing fragmented integrations with a universal protocol.",
        "The main metrics in Ragas include Faithfulness, Answer Relevancy, Context Precision, and Context Recall.",
        "MCP uses a client-host-server architecture where hosts (like IDEs) connect to servers (data sources) via a standard protocol."
    ]

    # 2. Collect Answers and Contexts
    answers = []
    contexts = []
    run_ids = []
    
    agent = build_agent()
    retriever = get_retriever()
    
    print("🤖 Generating answers...")
    for q in questions:
        # Get Answer from Agent and capture Run ID
        inputs = {"messages": [("user", q)]}
        with collect_runs() as cb:
            response = agent.invoke(inputs)
            run_id = cb.traced_runs[0].id
            run_ids.append(run_id)
            
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
    
    # 5. Push Results to LangSmith
    print("☁️ Pushing results to LangSmith...")
    client = Client()
    
    # Convert to pandas to easily iterate over rows
    df = results.to_pandas()
    
    # Iterate through the DataFrame rows
    for i, row in df.iterrows():
        if i < len(run_ids):
            run_id = run_ids[i]
            # row is a Series, we can convert to dict
            scores = row.to_dict()
            
            for metric_name, score in scores.items():
                # Filter out non-metric columns (like question, answer, contexts, ground_truth)
                # We only want numeric scores
                if isinstance(score, (int, float)) and metric_name not in ["question", "answer", "contexts", "ground_truth"]:
                    client.create_feedback(
                        run_id=run_id,
                        key=metric_name,
                        score=score,
                        source_info={"source": "ragas"}
                    )
    print("✨ Feedback submitted to LangSmith!")
    
    # Save to CSV (already done implicitly by logic above, but keeping the file write)
    df.to_csv("evaluation_results.csv", index=False)
    df.to_csv("evaluation_results.csv", index=False)
    print("💾 Detailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()
