from typing import List, Dict, Any

class ResponseGenerator:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
    def build_prompt(self, query: str, context_items: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None):
        """Build a prompt for the OpenAI model"""
        # Format the context
        context_text = ""
        for idx, item in enumerate(context_items):
            # Format filename to be more readable
            readable_name = item['filename'].replace('_', ' ').replace('.pdf', '')
            context_text += f"[Document {idx+1}: {readable_name}]\n{item['content']}\n\n"
            
        # Format conversation history if provided
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            history_text = "Previous conversation:\n"
            for msg in conversation_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
            history_text += "\n"
            
        # Build the prompt
        prompt = f"""You are a research specialist in criminology and recidivism studies analyzing academic literature.
Answer the following question based ONLY on the provided research contexts.
If the answer cannot be determined from the provided context, say "I don't have enough information to answer this question based on the provided research papers."

{history_text}

Important instructions:
1. Always cite your sources using document numbers (e.g., "According to Document 3...")
2. If studies contradict each other, acknowledge these differences
3. Include relevant statistics and figures when available
4. If the question asks for solutions or recommendations, prioritize evidence-based approaches
5. Be objective and present multiple perspectives when the research shows diverse viewpoints

RESEARCH CONTEXTS:
{context_text}

QUESTION: {query}

Think step by step before providing your final answer:
1. Identify which documents contain relevant information
2. Analyze what each source says about the specific question
3. Synthesize the information to provide a comprehensive answer
4. Ensure all claims are properly cited

ANSWER:"""
        return prompt
        
    def generate_response(self, query: str, context_items: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None):
        """Generate a response using OpenAI with retrieved context"""
        # Build the prompt
        prompt = self.build_prompt(query, context_items, conversation_history)
        
        # Generate the response
        response = self.openai_client.generate_completion(prompt)
        
        return response