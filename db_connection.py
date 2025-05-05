from pymongo import MongoClient

# MongoDB Connection Details
MONGO_URI = "mongodb+srv://akhil:2432@recidivism.b6el1st.mongodb.net/?retryWrites=true&w=majority&appName=Recidivism"
DB_NAME = "Recidivism"
COLLECTION_NAME = "Recidivism LLM"

# Establish connection
client = MongoClient(MONGO_URI)

# Access database and collection
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Check connection by listing collections
try:
    print("✅ Connected to MongoDB!")
    print("Available Collections:", db.list_collection_names())
except Exception as e:
    print("❌ MongoDB Connection Error:", e)
