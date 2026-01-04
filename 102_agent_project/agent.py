import os
import platform
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  WARNING: OPENAI_API_KEY is not set in your environment or .env file.")
    print("Please set it to run the agent. Example: export OPENAI_API_KEY='sk-...'")
    # We won't exit here to allow import, but running it will fail if key is needed.

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# Import our RAG retriever
# Add current directory to sys.path to ensure we can import rag_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline import get_retriever

# Define Tools
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for information about MCP (Model Context Protocol) 
    or Ragas (Evaluation Framework).
    """
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error searching knowledge base: {e}"

@tool
def check_system_info() -> str:
    """Returns the current system platform and python version."""
    return f"System: {platform.platform()}, Python: {sys.version}"

# Setup Agent
def build_agent():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Define tool list
    tools = [search_knowledge_base, check_system_info]
    
    # Create the ReAct agent graph
    system_prompt = "You are a helpful AI assistant. Use your tools to answer questions about MCP and Ragas. If you don't know, look it up in the knowledge base."
    
    graph = create_react_agent(llm, tools, state_modifier=system_prompt)
    return graph

def run_chat_loop():
    """Simple terminal chat loop for testing."""
    agent = build_agent()
    print("🤖 Agent is ready! (Type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Bye!")
                break
            
            # Stream the response
            print("Agent: ", end="", flush=True)
            inputs = {"messages": [("user", user_input)]}
            
            # We use stream to get tokens as they come, or just get the final state.
            # For simplicity in terminal, let's just invoke and print the last message.
            # But streaming is cooler. Let's try to just print the final response content.
            
            response = agent.invoke(inputs)
            
            # The response state contains 'messages'. The last one is the AI's answer.
            if response and "messages" in response:
                print(response["messages"][-1].content)
            else:
                print("(No response)")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    run_chat_loop()
