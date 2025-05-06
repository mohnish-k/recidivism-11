from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb+srv://akhil:2432@recidivism.b6el1st.mongodb.net/?retryWrites=true&w=majority&appName=Recidivism")
db = client["Recidivism"]
collection = db["Recidivism LLM"]

# Test the connection
doc_count = collection.count_documents({})
print(f"Total documents in collection: {doc_count}")

# Print a sample document (if available)
if doc_count > 0:
    sample_doc = collection.find_one({})
    print("Sample document:")
    print(f"Document ID: {sample_doc['_id']}")
    print(f"Filename: {sample_doc.get('filename', 'N/A')}")
    print(f"Content length: {len(sample_doc.get('content', ''))}")