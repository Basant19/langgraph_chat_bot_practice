import os
from google import genai
from dotenv import load_dotenv

# 1. Load your .env file
load_dotenv()

# 2. Get the key (Checks both common names)
API_KEY = os.getenv("GOOGLE_API_KEY") 

def check_google_ai():
    if not API_KEY:
        print("❌ Error: No API key found. Did you set GOOGLE_API_KEY in your .env file?")
        return

    try:
        # Initialize the modern client
        client = genai.Client(api_key=API_KEY)
        
        print("--- Checking Model Availability ---")
        # List all models to verify connection
        for m in client.models.list():
            print(f"Model found: {m.name}")
        
        print("\n--- Testing Content Generation ---")
        # Testing with the newest stable model
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", 
            contents="Say 'The API is working and Gemini is ready!'"
        )
        print(f"Response: {response.text}")
        print("\n✅ Success! Your API key is valid and working.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: Ensure your key is from https://aistudio.google.com/app/apikey")

if __name__ == "__main__":
    check_google_ai()