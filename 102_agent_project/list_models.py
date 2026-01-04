import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env from parent if needed
if not load_dotenv():
    load_dotenv("../.env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ No GOOGLE_API_KEY found.")
    exit(1)

print(f"🔑 Key found: {api_key[:5]}...")

try:
    genai.configure(api_key=api_key)
    print("📋 Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
except Exception as e:
    print(f"❌ Error listing models: {e}")
