import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FAISSVectorStore:
    def __init__(self):
        # Get path from environment variable or use default
        vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_store.index")
        print(f"VECTOR_DB_PATH from env: {vector_db_path}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Try different paths to find the index file
        possible_paths = [
            vector_db_path,
            os.path.abspath(vector_db_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vector_store.index"),
            "./data/vector_store.index",
            "../data/vector_store.index",
            "../../data/vector_store.index",
            "C:/Users/akhil/Downloads/Project (1)/recidivism-rag-chatbot/data/vector_store.index"
        ]
        
        self.index_path = None
        self.doc_ids_path = None
        
        # Try each path to find the files
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found vector store index at: {path}")
                self.index_path = path
                self.doc_ids_path = os.path.join(os.path.dirname(path), "doc_ids.pkl")
                
                if os.path.exists(self.doc_ids_path):
                    print(f"Found doc_ids at: {self.doc_ids_path}")
                    break
                else:
                    print(f"doc_ids.pkl not found at expected location: {self.doc_ids_path}")
                    self.index_path = None  # Reset if doc_ids not found
        
        if not self.index_path or not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found. Tried paths: {possible_paths}")
        
        if not self.doc_ids_path or not os.path.exists(self.doc_ids_path):
            raise FileNotFoundError(f"Document IDs not found at {self.doc_ids_path}")
        
        # Load the index and document IDs
        try:
            print(f"Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            print(f"Index loaded with {self.index.ntotal} vectors")
            
            print(f"Loading doc_ids from {self.doc_ids_path}")
            with open(self.doc_ids_path, "rb") as f:
                self.doc_ids = pickle.load(f)
            print(f"Loaded {len(self.doc_ids)} document IDs")
            
            # Try to load doc_info if available
            try:
                doc_info_path = os.path.join(os.path.dirname(self.index_path), "doc_info.pkl")
                if os.path.exists(doc_info_path):
                    print(f"Loading doc_info from {doc_info_path}")
                    with open(doc_info_path, "rb") as f:
                        self.doc_info = pickle.load(f)
                    print(f"Loaded document info for {len(self.doc_info)} documents")
                else:
                    print("No doc_info.pkl found")
                    self.doc_info = None
            except Exception as e:
                print(f"Error loading doc_info: {str(e)}")
                self.doc_info = None
                
        except Exception as e:
            import traceback
            print(f"Error loading FAISS index or document IDs: {str(e)}")
            traceback.print_exc()
            raise
            
    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """Search the FAISS index for similar vectors"""
        print(f"Searching FAISS index with vector of shape {query_vector.shape}")
        try:
            distances, indices = self.index.search(query_vector, top_k)
            print(f"Search returned {len(indices[0])} results")
            return distances, indices
        except Exception as e:
            print(f"Error during FAISS search: {str(e)}")
            import traceback
            traceback.print_exc()
            raise