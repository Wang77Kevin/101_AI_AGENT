# 🤖 AI Agent with RAG & Tools

This project implements an AI Agent that can search a local knowledge base (RAG) and perform system checks. It uses **LangGraph**, **LangChain**, **FAISS**, **LangServe**, and **Google Gemini**.

## 📂 Structure
- `data/`: Knowledge base (Markdown files).
- `rag_pipeline.py`: Handles data ingestion and retrieval using FAISS and Local Embeddings.
- `agent.py`: Defines the ReAct agent using LangGraph and Google Gemini.
- `serve.py`: LangServe entry point (API) for connecting to Agent Chat UI.
- `evaluate.py`: Evaluation script using Ragas and Gemini.

## 🚀 Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `faiss-cpu` installed if on Mac/Linux)*

2. **Set API Keys**
   Copy `.env.example` to `.env` and add your keys:
   ```bash
   cp .env.example .env
   # Edit .env file
   ```
   Required:
   - `GOOGLE_API_KEY`: For Gemini Model.
   
   Optional (for Tracing):
   - `LANGCHAIN_TRACING_V2=true`
   - `LANGCHAIN_API_KEY`: Your LangSmith Key.

3. **Ingest Data**
   Build the vector database:
   ```bash
   python rag_pipeline.py
   ```

## 🏃‍♂️ Usage

### 1. Run as an API (LangServe)
To expose the agent as a REST API (compatible with LangChain Agent UI):
```bash
python serve.py
```
*Access docs at http://localhost:8000/docs*
*Connect via Agent Chat UI at http://localhost:8000/agent*

### 2. Run CLI Agent
Simple terminal chat:
```bash
python agent.py
```

### 3. Run Evaluation
Evaluate the RAG pipeline using Ragas:
```bash
python evaluate.py
```
