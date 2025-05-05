import os
import openai
import pymongo
import dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables (store your OpenAI API Key in a .env file)
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["Recidivism-LLM"]
collection = db["research_papers"]

# Fetch text from MongoDB
documents = collection.find({}, {"text": 1})  # Only fetch text field
text_data = " ".join([doc["text"] for doc in documents])

# Step 1: Split Text into Chunks for Efficient Retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_text(text_data)

# Step 2: Create a Vector Database
vector_store = Chroma.from_texts(text_chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

# Step 3: Initialize OpenAI Model for Q&A
llm = OpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA(llm=llm, retriever=vector_store.as_retriever())

# Step 4: Query the LLM
def ask_llm(query):
    response = qa_chain.run(query)
    return response

# Example Query
if __name__ == "__main__":
    query = "What are the key insights from recidivism studies?"
    response = ask_llm(query)
    print(f"🤖 LLM Response: {response}")

