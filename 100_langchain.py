import langchain
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

def check_environment():
    print(f"✅ LangChain version: {langchain.__version__}")
    
    # Check for common API keys
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    configured_keys = [key for key in api_keys if os.environ.get(key)]
    
    if configured_keys:
        print(f"🔑 Configured API Keys: {', '.join(configured_keys)}")
        if "GOOGLE_API_KEY" in configured_keys:
            test_gemini()
    else:
        print("⚠️  No API keys found. Please configure GOOGLE_API_KEY in .env")

def list_gemini_models():
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        print("\n📋 Available Gemini Models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"   - {m.name}")
    except Exception as e:
        print(f"⚠️  Could not list models: {e}")

def test_gemini():
    print("\n🧪 Testing Gemini connection...")
    
    # Try gemini-2.0-flash which is confirmed working
    model_name = "gemini-2.0-flash"
    try:
        print(f"   Attempting to use model: {model_name}")
        llm = ChatGoogleGenerativeAI(model=model_name)
        response = llm.invoke("Hello, are you working?")
        print(f"🤖 Gemini says: {response.content}")
        print("✅ Gemini connection successful!")
        return
    except Exception as e:
        print(f"❌ Failed with {model_name}: {e}")
        list_gemini_models()

if __name__ == "__main__":
    check_environment()
