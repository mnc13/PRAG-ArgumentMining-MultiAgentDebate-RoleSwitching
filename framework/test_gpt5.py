import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    print("Testing gpt-5-mini with Responses API...")
    # Based on user snippet and search result
    response = client.responses.create(
        model="gpt-5-mini",
        input="Say 'I am gpt-5-mini and I am working correctly.'"
    )
    print("Full gpt-5-mini Response (Responses API):")
    print(f"Output: '{response.output_text}'")
    
    # Check other fields if any
    print(f"ID: {response.id}")

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying to inspect client.responses methods...")
    try:
        print(dir(client.responses))
    except:
        print("client.responses not found")
