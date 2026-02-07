import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Listing available models...")
with open("models_list.txt", "w") as f:
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                f.write(f"{m.name}\n")
                print(m.name)
    except Exception as e:
        f.write(f"Error listing models: {e}\n")
        print(f"Error: {e}")
