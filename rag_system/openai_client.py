import os
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def generate_completion(self, prompt: str, model: str = "gpt-4o", temperature: float = 0.2, max_tokens: int = 1500):
        """Generate a completion using OpenAI"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research assistant specializing in criminology and recidivism studies. Your answers should be factual, nuanced, and based exclusively on the provided research context. Always cite your sources. When the research is inconclusive, acknowledge this clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content