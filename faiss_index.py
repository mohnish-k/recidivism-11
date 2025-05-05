import pymongo
import faiss
import numpy as np
import pickle
import os
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

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

def load_embeddings():
    """Load embeddings and document IDs from MongoDB."""
    # First ensure no duplicates
    clean_duplicates()
    
    embeddings = []
    doc_ids = []
    doc_info = []  # Store additional info for better retrieval

    # Get only documents with embeddings
    for doc in collection.find({"embedding": {"$exists": True}}):
        embeddings.append(doc["embedding"])
        doc_ids.append(doc["_id"])
        # Store filename for easier reference later
        doc_info.append({
            "id": doc["_id"],
            "filename": doc.get("filename", "Unknown"),
        })

    if not embeddings:
        print("❌ No embeddings found in database!")
        return np.array([], dtype=np.float32), [], []
    
    # Convert to numpy array and normalize
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Check if embeddings are already normalized, if not, normalize them
    norms = np.linalg.norm(embeddings_array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        print("⚠️ Normalizing embeddings for better search results")
        embeddings_array = embeddings_array / norms[:, np.newaxis]
    
    return embeddings_array, doc_ids, doc_info

def build_faiss_index():
    """Create and store an improved FAISS index."""
    embeddings, doc_ids, doc_info = load_embeddings()

    if embeddings.shape[0] == 0:
        print("❌ No embeddings found in database!")
        return

    print(f"📊 Building FAISS index with {len(embeddings)} embeddings")
    
    # Get embedding dimension
    d = embeddings.shape[1]
    
    # Choose appropriate index type based on dataset size
    if embeddings.shape[0] > 1000:
        # For larger datasets, use IVF (Inverted File Index) for faster search
        n_clusters = min(int(4 * np.sqrt(embeddings.shape[0])), embeddings.shape[0] // 10)
        n_clusters = max(n_clusters, 1)  # Ensure at least 1 cluster
        
        # Use inner product similarity (for normalized vectors this is equivalent to cosine similarity)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)
        
        # Need to train IVF index
        print(f"🧠 Training FAISS index with {n_clusters} clusters...")
        index.train(embeddings)
        index.add(embeddings)
        
        # Set search parameters - higher nprobe means more accurate but slower search
        index.nprobe = min(n_clusters, 10)
        print(f"✅ Index trained with nprobe={index.nprobe}")
    else:
        # For smaller datasets, use a simple flat index with inner product (cosine) similarity
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        print("✅ Using flat index for small dataset")

    # Save FAISS index and document IDs
    faiss.write_index(index, "vector_store.index")
    with open("doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)
    with open("doc_info.pkl", "wb") as f:
        pickle.dump(doc_info, f)

    print("✅ FAISS index built and saved successfully!")
    print(f"   - Vector dimension: {d}")
    print(f"   - Number of documents: {len(doc_ids)}")
    
    # Test the index with a simple query
    print("\n🔍 Testing index with a simple query...")
    query = np.ones((1, d), dtype=np.float32)
    query = query / np.linalg.norm(query)
    D, I = index.search(query, min(5, len(doc_ids)))
    print(f"   - Top 5 document IDs: {[doc_info[i]['filename'] for i in I[0] if i < len(doc_info)]}")

if __name__ == "__main__":
    build_faiss_index()