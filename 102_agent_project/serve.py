#!/usr/bin/env python
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
import os
import sys
from dotenv import load_dotenv

# Load environment variables
if not load_dotenv():
    load_dotenv("../.env")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import build_agent

# Initialize FastAPI
app = FastAPI(
    title="AI Agent Server",
    version="1.0",
    description="A LangChain/LangGraph Agent exposed via LangServe",
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.get("/agent")
async def redirect_agent_to_playground():
    return RedirectResponse("/agent/playground/")

# Create and bind the agent
# Note: build_agent() returns a compiled graph
agent_graph = build_agent()

# Add LangServe routes
# The path must match what the UI expects or what you configure
add_routes(
    app,
    agent_graph,
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="localhost", port=8000)
