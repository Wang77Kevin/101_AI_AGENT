import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add current dir to path to import agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import build_agent

st.set_page_config(page_title="AI Agent RAG", page_icon="🤖")

st.title("🤖 AI Agent with Knowledge Base")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Check for API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        user_key = st.text_input("Enter OpenAI API Key:", type="password")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            api_key = user_key
            st.success("API Key set!")
    else:
        st.success("API Key detected from environment.")
        
    st.markdown("---")
    st.markdown("### Capabilities")
    st.markdown("- 🧠 **RAG**: Can search internal knowledge base (MCP, Ragas).")
    st.markdown("- 🛠️ **Tools**: Can check system information.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI agent. Ask me about the Model Context Protocol (MCP) or Ragas!"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if prompt := st.chat_input("Type your question here..."):
    # 1. Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Generate Response
    if not api_key:
        st.error("Please set your OpenAI API Key in the sidebar to continue.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build agent (lazy init to pick up new env vars if set)
                    agent = build_agent()
                    
                    # Invoke agent
                    inputs = {"messages": [("user", prompt)]}
                    response = agent.invoke(inputs)
                    
                    # Get final message
                    ai_content = response["messages"][-1].content
                    st.write(ai_content)
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": ai_content})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
