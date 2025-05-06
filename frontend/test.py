def retrieve_context(self, query: str, top_k: int = 5):
    print(f"Query: {query}")
    # Encode the query
    query_embedding = self.embedding_model.encode([query]).astype(np.float32)
    print(f"Generated embedding with shape: {query_embedding.shape}")
    
    # Rest of your code...