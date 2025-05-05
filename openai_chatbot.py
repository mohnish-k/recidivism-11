from openai import OpenAI
import faiss
import torch
import pickle
import numpy as np
import pymongo
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Load FAISS index and doc IDs ===
try:
    index = faiss.read_index("vector_store.index")
    with open("doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
except Exception as e:
    print(f"❌ Error loading FAISS index: {e}")
    exit(1)

# === MongoDB connection ===
MONGO_URI = "mongodb+srv://akhil:2432@recidivism.b6el1st.mongodb.net/?retryWrites=true&w=majority&appName=Recidivism"
DB_NAME = "Recidivism"
COLLECTION_NAME = "Recidivism LLM"

client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# === Load Sentence Transformer ===
embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
print("✅ Loaded embedding model: multi-qa-mpnet-base-dot-v1")

# === Initialize OpenAI client ===
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Social query handler ===
def handle_social_query(query):
    """Handle casual social interactions before using the RAG system"""
    query_lower = query.lower().strip()
    
    # Greetings
    if query_lower in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]:
        return f"Hello! I'm a research assistant specializing in criminology and recidivism studies. How can I help you today?"
    
    # Goodbyes
    if query_lower in ["bye", "goodbye", "see you", "farewell", "exit"]:
        return "Goodbye! Feel free to come back if you have more questions about recidivism research."
    
    # Identity questions
    if "who are you" in query_lower or "what are you" in query_lower or "your name" in query_lower:
        return "I'm a specialized research assistant focused on criminology and recidivism studies. I can help answer questions about factors contributing to recidivism, effectiveness of interventions, and other related research topics."
    
    # Capability questions
    if "what can you do" in query_lower or "how can you help" in query_lower or "what do you know" in query_lower:
        return "I can help answer research questions about recidivism, including factors that contribute to reoffending, effectiveness of rehabilitation programs, impact of education and employment, and evidence-based approaches to reducing recidivism. I access a database of academic papers and research to provide you with accurate information."
    
    # Appreciation
    if query_lower in ["thanks", "thank you", "appreciate it", "helpful"]:
        return "You're welcome! I'm glad I could help. Feel free to ask if you have more questions about recidivism research."
    
    # If not a social query, return None to indicate it should go to the regular process
    return None

# === Enhanced retrieval function ===
def retrieve_context(query, top_k=5):
    """Retrieve relevant context from the vector store"""
    # Encode the query
    query_embedding = embedding_model.encode([query]).astype(np.float32)
    
    # Normalize the query vector
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k*2)

    context_items = []
    for i in range(min(len(indices[0]), top_k*2)):
        idx = indices[0][i]
        if idx >= len(doc_ids):  # Safety check
            continue

        doc_id = doc_ids[idx]
        doc = collection.find_one({"_id": doc_id})

        if doc:
            # Include document metadata for better citations
            filename = doc.get("filename", "Unknown document")
            
            # Try to find a relevant section using keywords
            content = doc.get("content", "")
            keywords = [k for k in query.lower().split() if len(k) > 3]
            
            # Find sections that contain keywords
            text_snippet = content[:2000]  # Default to first 2000 chars
            if keywords:
                # Look for sections with keywords
                best_score = 0
                best_snippet = text_snippet
                
                for keyword in keywords:
                    keyword_pos = content.lower().find(keyword.lower())
                    if keyword_pos > 0:
                        start = max(0, keyword_pos - 500)
                        end = min(len(content), keyword_pos + 1500)
                        snippet = content[start:end]
                        
                        # Count how many keywords are in this snippet
                        score = sum(1 for k in keywords if k.lower() in snippet.lower())
                        if score > best_score:
                            best_score = score
                            best_snippet = snippet
                
                if best_score > 0:
                    text_snippet = best_snippet

            context_items.append({
                "filename": filename,
                "content": text_snippet,
                "score": float(distances[0][i])
            })

    # Sort by score and take top_k
    context_items.sort(key=lambda x: x["score"], reverse=True)
    return context_items[:top_k]

# === Improved prompt template for better information extraction ===
def build_prompt(query, context_items):
    """Build a prompt optimized for better information extraction"""
    context_text = ""
    for idx, item in enumerate(context_items):
        # Format filename to be more readable
        readable_name = item['filename'].replace('_', ' ').replace('.pdf', '')
        context_text += f"[Document {idx+1}: {readable_name}]\n{item['content']}\n\n"

    prompt = f"""You are a research specialist in criminology and recidivism studies analyzing academic literature.
Answer the following question based ONLY on the provided research contexts.
If the answer cannot be determined from the provided context, explain what information IS available in the documents and what seems to be missing.

Instructions:
1. Always cite your sources using document numbers (e.g., "According to Document 3...")
2. Extract all relevant information from the documents, even if it only partially addresses the question
3. Include key statistics if relevant
4. If studies contradict each other, acknowledge these differences
5. Be comprehensive and thorough in extracting information from the context

RESEARCH CONTEXTS:
{context_text}

QUESTION: {query}

Think step by step before providing your final answer:
1. Identify which documents contain information related to the question
2. Extract all specific factors, predictors, or variables mentioned in these documents
3. Organize this information into a coherent answer with proper citations
4. If information is limited, explain what IS available and what's missing

ANSWER:"""
    return prompt

# === Enhanced GPT-4 generation function ===
def generate_response(query):
    """Generate a comprehensive response using GPT-4 with retrieved context"""
    print("🔍 Searching for relevant research...")
    context_items = retrieve_context(query)
    
    if not context_items:
        return "I couldn't find relevant information in the research database to answer your question."
    
    print(f"📚 Found {len(context_items)} relevant documents")
    prompt = build_prompt(query, context_items)

    print("💬 Generating response...")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a research assistant specializing in criminology and recidivism studies. Provide thorough answers that fully extract and synthesize all relevant information from the provided context. Always cite sources and be direct in addressing the question."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800  # Increased for more comprehensive answers
    )

    return response.choices[0].message.content

# === Chat loop ===
print("🤖 Enhanced RAG Chatbot for Recidivism Research is ready! Type 'exit' to quit.")
print("📝 Type 'help' for available commands")

while True:
    user_input = input("\n🧑‍💻 You: ")
    
    if user_input.lower() == "exit":
        print("👋 Exiting chatbot.")
        break
    elif user_input.lower() == "help":
        print("\n📚 Available Commands:")
        print("- exit: Quit the application")
        print("- help: Display this help message")
        continue

    # Check if it's a social query first
    social_response = handle_social_query(user_input)
    if social_response:
        print(f"\n🤖 Research Assistant: {social_response}")
        continue
        
    # If it's not a social query, process it through the RAG system
    try:
        response = generate_response(user_input)
        print(f"\n🤖 Research Assistant: {response}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please try again with a different question.")