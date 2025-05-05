import os
import re
import pymongo
import fitz  # PyMuPDF for PDF processing
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

FOLDER_PATH = "./Research Papers"
if not os.path.exists(FOLDER_PATH):
    print(f"❌ Error: Folder not found: {FOLDER_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    exit(1)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file with improved layout handling."""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        # Get text with better block handling
        blocks = page.get_text("blocks")
        # Sort blocks by vertical position for better reading order
        blocks.sort(key=lambda b: b[1])  # Sort by y0 coordinate

        for block in blocks:
            block_text = block[4]
            # Skip page numbers and headers/footers (usually short blocks)
            if len(block_text.strip()) < 20:
                continue
            text += block_text + "\n"

    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

def store_in_mongodb():
    """Extract text from PDFs and store in MongoDB."""
    processed_count = 0
    skipped_count = 0

    # First, check for duplicates and clean them up
    filenames = {}
    for doc in collection.find({}, {"filename": 1}):
        filename = doc.get("filename")
        if filename:
            if filename in filenames:
                # This is a duplicate, remove it
                print(f"🔄 Removing duplicate: {filename}")
                collection.delete_one({"_id": doc["_id"]})
            else:
                filenames[filename] = doc["_id"]

    print(f"✅ Database cleaned: {len(filenames)} unique documents kept")

    # Now process new files
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".pdf"):
            file_path = os.path.join(FOLDER_PATH, filename)

            # Check if file already exists in database
            if filename in filenames:
                print(f"⏭️ Already processed: {filename}")
                skipped_count += 1
                continue

            print(f"📄 Processing: {filename}")
            text = extract_text_from_pdf(file_path)
            data = {"filename": filename, "content": text}

            collection.insert_one(data)
            processed_count += 1
            print(f"✅ Stored in MongoDB: {filename}")

    print(f"✅ Processed {processed_count} new documents, skipped {skipped_count} existing documents")

if __name__ == "__main__":
    store_in_mongodb()
