import PyPDF2
import os

# Update the folder path to match your actual folder name
FOLDER_PATH = "./Research Papers"

# Ensure the folder exists
if not os.path.exists(FOLDER_PATH):
    print(f"❌ Folder '{FOLDER_PATH}' not found! Please create the folder and add PDFs.")
    exit()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Read all PDFs in the Research Papers folder
documents = {}
pdf_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".pdf")]

if not pdf_files:
    print("❌ No PDF files found in the 'Research Papers' folder! Please add PDFs and re-run the script.")
    exit()

for file in pdf_files:
    file_path = os.path.join(FOLDER_PATH, file)
    text = extract_text_from_pdf(file_path)
    documents[file] = text

# Print the extracted text (first 1000 characters for preview)
for doc, content in documents.items():
    print(f"\n📄 Extracted Text from {doc}:\n")
    print(content[:1000])  # Print a preview of the extracted text
