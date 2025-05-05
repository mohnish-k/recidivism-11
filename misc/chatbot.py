from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import pymongo
import os
import requests
from config import MONGO_URI, DB_NAME, COLLECTION_NAME
 
# 👇 Groq API Key
GROQ_API_KEY = "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"
 
# 🔧 FastAPI app
app = FastAPI()
 
# 🔓 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# 🧾 Request schema
class Query(BaseModel):
    query: str
 
# 📦 Load FAISS index
index = faiss.read_index("vector_store.index")
with open("doc_ids.pkl", "rb") as f:
    doc_ids = pickle.load(f)
 
# 🧠 Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
 
# 🧬 Connect MongoDB
client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
 
# 🔁 FAISS context retrieval
def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, top_k)
    context = ""
    for i in range(top_k):
        doc_id = doc_ids[indices[0][i]]
        doc = collection.find_one({"_id": doc_id})
        context += doc.get("processed_content", "")[:1000] + "\n\n"
    return context.strip()
 
# 🔗 Call Groq Cloud API
def query_groq_llm(prompt, model="mixtral-8x7b-32768", max_tokens=200):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
 
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
 
# 🚀 /search endpoint
@app.post("/search")
def search_endpoint(query: Query):
    context = retrieve_context(query.query)
    prompt = f"Context:\n{context}\n\nQuestion: {query.query}\nAnswer:"
    result = query_groq_llm(prompt)
    return {"results": [{"generated_text": result}]}
 
# 📤 /upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = "./uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
 
    with open(file_path, "wb") as f:
        f.write(await file.read())
 
    return {"message": "File uploaded successfully"}
 