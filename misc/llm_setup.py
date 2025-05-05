from transformers import pipeline

# Load a lightweight open-source LLM (e.g., Falcon 7B Instruct)
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

# Test prompt
prompt = "Explain the importance of reducing recidivism in criminal justice."
response = generator(prompt, max_length=100, num_return_sequences=1)

print("\n📝 LLM Response:\n", response[0]["generated_text"])
