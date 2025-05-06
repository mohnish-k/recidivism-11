import os
from typing import List, Dict, Any
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        
        # MongoDB connection
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("MongoDB URI is required but not provided")
            
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["Recidivism"]
        self.collection = self.db["Recidivism LLM"]
        print(f"MongoDB connection established to {self.mongo_uri}")
        print(f"Collection document count: {self.collection.count_documents({})}")
        
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the vector store"""
        print(f"Retrieving context for query: {query}")
        
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode([query]).astype(np.float32)
            print(f"Generated query embedding with shape: {query_embedding.shape}")
            
            # Normalize the query vector
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search FAISS index
            distances, indices = self.vector_store.search(query_embedding, top_k*2)
            print(f"FAISS search returned {len(indices[0])} results")

            context_items = []
            
            # Debugging - print first few doc IDs
            print("Sample doc IDs from vector store:")
            for i in range(min(5, len(self.vector_store.doc_ids))):
                print(f"  {i}: {self.vector_store.doc_ids[i]} (type: {type(self.vector_store.doc_ids[i])})")
            
            # Instead of trying to look up by document ID, just retrieve some documents
            print("Retrieving sample documents from MongoDB:")
            sample_docs = list(self.collection.find({}).limit(3))
            for doc in sample_docs:
                print(f"  Sample doc ID: {doc['_id']} (type: {type(doc['_id'])})")
                print(f"  Filename: {doc.get('filename', 'Unknown')}")
            
            print("Attempting to match vector store indices to MongoDB documents...")
            
            for i in range(min(len(indices[0]), top_k*2)):
                idx = indices[0][i]
                
                # Safety check
                if idx >= len(self.vector_store.doc_ids) or idx < 0:
                    print(f"Index {idx} out of bounds for doc_ids array of length {len(self.vector_store.doc_ids)}")
                    continue
                
                doc_id = self.vector_store.doc_ids[idx]
                print(f"Looking up document with ID: {doc_id} (type: {type(doc_id)})")
                
                # Try different approaches to find the document
                approaches = [
                    {"_id": doc_id},  # Direct match
                    {"_id": str(doc_id)},  # String conversion
                    {"_id": {"$in": [doc_id, str(doc_id)]}}  # Try both
                ]
                
                doc = None
                for approach_num, query_filter in enumerate(approaches):
                    doc = self.collection.find_one(query_filter)
                    if doc:
                        print(f"Found document using approach {approach_num+1}: {doc.get('filename', 'Unknown')}")
                        break
                
                if not doc:
                    print(f"Document with ID {doc_id} not found in MongoDB using any approach")
                    continue
                
                # Process the document if found
                filename = doc.get("filename", "Unknown document")
                content = doc.get("content", "")
                
                if content:
                    # Extract a relevant snippet
                    text_snippet = content[:2000]  # Default to first 2000 chars
                    keywords = [k for k in query.lower().split() if len(k) > 3]
                    
                    if keywords:
                        print(f"Using keywords for snippet extraction: {keywords}")
                        best_score = 0
                        best_snippet = text_snippet
                        
                        for keyword in keywords:
                            keyword_pos = content.lower().find(keyword.lower())
                            if keyword_pos > 0:
                                start = max(0, keyword_pos - 500)
                                end = min(len(content), keyword_pos + 1500)
                                snippet = content[start:end]
                                
                                # Count keyword occurrences
                                score = sum(1 for k in keywords if k.lower() in snippet.lower())
                                if score > best_score:
                                    best_score = score
                                    best_snippet = snippet
                        
                        if best_score > 0:
                            print(f"Found better snippet with score {best_score}")
                            text_snippet = best_snippet
                    
                    context_items.append({
                        "document_id": doc_id,
                        "filename": filename,
                        "content": text_snippet,
                        "score": float(distances[0][i])
                    })
                    print(f"Added document '{filename}' to context items")
                else:
                    print(f"Document has no content")
            
            # If no documents found, try a fallback approach
            if not context_items:
                print("No matching documents found, trying fallback approach")
                random_docs = list(self.collection.aggregate([{"$sample": {"size": 3}}]))
                
                for doc in random_docs:
                    if doc and "content" in doc:
                        context_items.append({
                            "document_id": doc["_id"],
                            "filename": doc.get("filename", "Unknown document"),
                            "content": doc["content"][:2000],
                            "score": 0.5  # Arbitrary score
                        })
                        print(f"Added random document '{doc.get('filename', 'Unknown')}' to context items")
            
            print(f"Returning {len(context_items)} context items")
            return context_items
            
        except Exception as e:
            import traceback
            print(f"Error in retrieve_context: {str(e)}")
            traceback.print_exc()
            return []  # Return empty list on error