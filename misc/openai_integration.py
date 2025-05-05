import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API Key is loaded
if not api_key:
    raise ValueError("API Key not found. Please check your .env file.")

# Set API Key for OpenAI
client = openai.OpenAI(api_key=api_key)

# Function to test API with a sample query
def chat_with_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 instead of GPT-4
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Test query
if __name__ == "__main__":
    user_query = "Explain recidivism in simple terms."
    response = chat_with_gpt(user_query)
    print("GPT Response:", response)
