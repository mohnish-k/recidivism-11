import os
import sys
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configure the API endpoint - default to localhost if not specified
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Recidivism Research Assistant",
    page_icon="📚",
    layout="wide",
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .source-citation {
        font-size: 0.85rem;
        color: #bababa;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm a research assistant specializing in criminology and recidivism studies. How can I help you today?"}
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Main app layout
st.title("📚 Recidivism Research Assistant")

st.markdown("""
This AI-powered research assistant helps answer questions about recidivism studies and criminology research.
Ask questions about research findings, statistics, or evidence-based approaches to reducing recidivism.
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about recidivism research..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant response in chat
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Researching your question...")
        
        # Prepare conversation history for the API request
        conversation_history = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in st.session_state.messages[:-1]  # Exclude the current user message
        ]
        
        # Make API request
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "query": prompt,
                    "session_id": st.session_state.session_id,
                    "conversation_history": conversation_history
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                sources = result.get("sources", [])
                
                # Display answer
                message_placeholder.markdown(answer)
                
                # Display sources if available
                if sources:
                    source_text = "<div class='source-citation'><strong>Sources:</strong><br>"
                    for i, source in enumerate(sources):
                        source_text += f"- {source.get('filename', 'Unknown document')}<br>"
                    source_text += "</div>"
                    st.markdown(source_text, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_msg = f"Error connecting to the API: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with additional information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This research assistant uses Retrieval Augmented Generation (RAG) to provide accurate, 
    evidence-based answers about recidivism studies.
    
    The system searches through a database of academic papers and research to find the 
    most relevant information for your queries.
    """)
    
    st.title("Sample Questions")
    st.markdown("""
    - What factors contribute to recidivism rates?
    - How effective are rehabilitation programs in reducing reoffending?
    - What does research say about the impact of education on recidivism?
    - How do employment opportunities affect reoffending rates?
    - What are evidence-based approaches to reducing juvenile recidivism?
    """)