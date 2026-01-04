# 🤖 AI Agent with RAG & Tools

This project implements an AI Agent that can search a local knowledge base (RAG) and perform system checks. It uses **LangGraph**, **LangChain**, **FAISS**, and **Streamlit**.

## 📂 Structure
- `data/`: Knowledge base (Markdown files).
- `rag_pipeline.py`: Handles data ingestion and retrieval using FAISS.
- `agent.py`: Defines the ReAct agent using LangGraph.
- `app.py`: Streamlit Chat UI.
- `evaluate.py`: Evaluation script using Ragas.

## 🚀 Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `faiss-cpu` installed if on Mac/Linux)*

2. **Set API Key**
   Copy `.env.example` to `.env` and add your OpenAI API Key:
   ```bash
   cp .env.example .env
   # Edit .env file
   ```

3. **Ingest Data**
   Build the vector database:
   ```bash
   python rag_pipeline.py
   ```

## 🏃‍♂️ Usage

### Run the Chat UI
```bash
streamlit run app.py
```

### Run CLI Agent
```bash
python agent.py
```

### Run Evaluation
```bash
python evaluate.py
```
