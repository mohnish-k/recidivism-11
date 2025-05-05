import faiss
import numpy as np
import pickle
import pymongo
from sentence_transformers import SentenceTransformer
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Load FAISS index and doc IDs
index = faiss.read_index("vector_store.index")
with open("doc_ids.pkl", "rb") as f:
    doc_ids = pickle.load(f)

try:
    with open("doc_info.pkl", "rb") as f:
        doc_info = pickle.load(f)
except FileNotFoundError:
    doc_info = None

print(f"✅ Loaded FAISS index with {index.ntotal} documents")

# MongoDB connection
client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Load embedding model (should match the one used for building index)
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
print("✅ Embedding model loaded")

# Sample query (replace this with any question)
query = "What are the best rehabilitation methods to reduce recidivism?"
print(f"\n🔍 Query: {query}")

# Generate embedding and normalize
query_embedding = model.encode([query]).astype(np.float32)
query_embedding = query_embedding / np.linalg.norm(query_embedding)

# Search index
top_k = 5
distances, indices = index.search(query_embedding, top_k)

print("\n📄 Top Matching Documents:")
for rank, idx in enumerate(indices[0]):
    if idx >= len(doc_ids):
        continue
    doc_id = doc_ids[idx]
    doc = collection.find_one({"_id": doc_id})
    filename = doc.get("filename", "Unknown") if doc else "Not found"
    print(f"{rank+1}. {filename} — Score: {distances[0][rank]:.4f}")
