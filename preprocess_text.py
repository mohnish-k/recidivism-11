import pymongo
import re
import spacy
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Text Cleaning Function
def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove references [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Remove page numbers and headers/footers
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip if just a page number
        if re.match(r'^\s*\d+\s*$', line):
            continue
        # Skip if looks like a header/footer
        if len(line.strip()) < 30 and re.search(r'page|chapter|section', line.lower()):
            continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines).strip()

# NLP Preprocessing Function
def preprocess_text(text):
    # More selective stopword removal - keep important words
    custom_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else',
                         'when', 'so', 'than', 'that', 'this', 'these', 'those'}

    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        # Keep numbers, percentages, and meaningful words
        if (token.is_alpha and not token.is_stop) or token.like_num:
            tokens.append(token.lemma_)

    return " ".join(tokens)

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

# Main Processing Function with progress tracking
def process_documents():
    # First, clean up duplicates
    clean_duplicates()

    # Only find documents that need processing
    docs = list(collection.find({"processed_content": {"$exists": False}}))
    if not docs:
        print("✅ All documents already processed!")
        return

    print(f"🧠 Found {len(docs)} documents to process. Starting preprocessing...\n")

    for i, doc in enumerate(docs):
        raw_text = doc.get("content", "")
        cleaned = clean_text(raw_text)
        processed = preprocess_text(cleaned)

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"processed_content": processed}}
        )

        # Print progress
        progress = (i + 1) / len(docs) * 100
        print(f"✅ Processed ({progress:.1f}%): {doc.get('filename', 'Unnamed Document')}")
        
if __name__ == "__main__":
    process_documents()
