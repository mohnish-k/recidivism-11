from transformers import pipeline
import faiss
import pickle
import numpy as np
import pymongo
from config import MONGO_URI, DB_NAME, COLLECTION_NAME
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("vector_store.index")
with open("doc_ids.pkl", "rb") as f:
    doc_ids = pickle.load(f)

# MongoDB connection
client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Falcon LLM
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

def retrieve_context(query, top_k=3):
    """Get top documents using FAISS"""
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    context = ""
    for i in range(top_k):
        doc_id = doc_ids[indices[0][i]]
        doc = collection.find_one({"_id": doc_id})
        context += doc.get("processed_content", "")[:1000] + "\n\n"
    return context.strip()

print("🤖 RAG Chatbot is ready! Type 'exit' to quit.")

while True:
    user_input = input("\n🧑‍💻 You: ")
    if user_input.lower() == "exit":
        break

    # 1. Retrieve context from FAISS
    context = retrieve_context(user_input)

    # 2. Generate response using context + user query
    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
    response = generator(prompt, max_new_tokens=150, num_return_sequences=1, truncation=True)[0]["generated_text"]

    
    print("\n🤖 Falcon:", response)
