from google import genai
import os
from dotenv import load_dotenv
# 1. Load your .env file
load_dotenv()

# 2. Get the key (Checks both common names)
API_KEY = os.getenv("GOOGLE_API_KEY") 
# Create client (automatically reads GOOGLE_API_KEY from env)
client = genai.Client(api_key=API_KEY)

# Simple test call
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Say hello in one sentence."
)

print(response.text)