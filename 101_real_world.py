import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

# Load environment variables
load_dotenv()

# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# Configure model - Switching to Gemini
# Common Parameters Explained:
# temperature (0.0 - 1.0+): 
#   - 0.0: Deterministic, focused, "boring". Good for code, math, and factual agents.
#   - 0.7: Balanced creativity. Good for chat and writing.
#   - 1.0+: Chaotic, very creative, but prone to errors. Good for brainstorming.
# max_output_tokens (Integer):
#   - Limits the length of the response. 
#   - Good for controlling costs or forcing brevity (e.g., set to 100 for short summaries).
# top_p (0.0 - 1.0):
#   - "Nucleus Sampling". 0.1 means "only consider the top 10% most likely words".
#   - Similar effect to temperature but mathematically different. usually set one or the other.
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,      # Changed to 0.7 for more creative puns
    max_output_tokens=500 # Limit response size to keep it concise
)

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

print("--- Turn 1: Asking about weather ---")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(f"Agent Response: {response['structured_response']}")


print("\n--- Turn 2: Follow up ---")
# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(f"Agent Response: {response['structured_response']}")
