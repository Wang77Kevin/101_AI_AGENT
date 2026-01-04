import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load env (though we are using local embeddings, so no API key strictly needed for this part)
load_dotenv()

# Constants
DATA_PATH = "data"
DB_PATH = "faiss_db"

def ingest_data():
    """Reads MD files, splits them, and saves to Vector Store."""
    print("🚀 Starting ingestion...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data directory '{DATA_PATH}' not found.")
        return

    loader = DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} documents.")

    # 2. Split Data
    # We use a small chunk size because our docs are small
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks.")

    # 3. Embed & Save
    # Using a free, local embedding model (no API cost!)
    print("🧠 Creating embeddings (this may take a moment)...")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_function
    )
    
    vector_db.save_local(DB_PATH)
    print(f"💾 Saved to {DB_PATH}")

def get_retriever():
    """Returns a retriever object for the agent to use."""
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(DB_PATH, embedding_function, allow_dangerous_deserialization=True)
    
    # "k=2" means return the top 2 most relevant chunks
    return vector_db.as_retriever(search_kwargs={"k": 2})

if __name__ == "__main__":
    # If run directly, perform ingestion
    ingest_data()
