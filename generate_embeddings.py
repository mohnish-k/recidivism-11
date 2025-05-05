import pymongo
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Load a better embedding model
# 'multi-qa-mpnet-base-dot-v1' has much better retrieval performance
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Function to check for and remove duplicates
def clean_duplicates():
    # Group by filename
    filenames = {}
    duplicates = []
    
    for doc in collection.find({}, {"filename": 1}):
        filename = doc.get("filename")
        if filename:
            if filename in filenames:
                # This is a duplicate
                duplicates.append(doc["_id"])
            else:
                filenames[filename] = doc["_id"]
    
    # Remove duplicates
    if duplicates:
        result = collection.delete_many({"_id": {"$in": duplicates}})
        print(f"🧹 Removed {result.deleted_count} duplicate documents")
    else:
        print("✅ No duplicates found")

def generate_embeddings():
    """Generate embeddings for processed text and store in MongoDB."""
    # First clean any duplicates
    clean_duplicates()
    
    # Find documents that need embeddings
    docs = list(collection.find(
        {"processed_content": {"$exists": True}, "embedding": {"$exists": False}}
    ))

    if not docs:
        print("✅ All documents already have embeddings!")
        return
        
    print(f"🚀 Generating embeddings for {len(docs)} documents...")
    
    # Process in batches to avoid memory issues
    batch_size = 16  # Smaller batch size for larger model
    total_batches = (len(docs) - 1) // batch_size + 1
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        
        # Extract text and IDs
        texts = [doc["processed_content"] for doc in batch_docs]
        doc_ids = [doc["_id"] for doc in batch_docs]
        
        # Generate embeddings with progress tracking
        print(f"⚙️ Processing batch {i//batch_size + 1}/{total_batches}...")
        embeddings = model.encode(texts, show_progress_bar=True, 
                                 normalize_embeddings=True)  # Normalize for better similarity search
        
        # Store embeddings with batch updates
        for j, doc_id in enumerate(doc_ids):
            collection.update_one(
                {"_id": doc_id},
                {"$set": {"embedding": embeddings[j].tolist(),
                         "embedding_model": "multi-qa-mpnet-base-dot-v1"}}  # Track which model created embeddings
            )
    
    print("✅ Embeddings generated and stored successfully!")

if __name__ == "__main__":
    generate_embeddings()
