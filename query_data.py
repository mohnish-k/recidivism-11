import pandas as pd
from pymongo import MongoClient
import config  # Ensure config.py contains MongoDB credentials

# Step 1: Connect to MongoDB
try:
    client = MongoClient(config.MONGO_URI)
    db = client[config.DB_NAME]
    collection = db[config.COLLECTION_NAME]
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")
    exit()

# Step 2: Fetch Data from MongoDB
try:
    query_result = collection.find({}, {"_id": 0})  # Fetch all records, exclude `_id`
    data = list(query_result)  # Convert cursor to list

    if not data:
        print("⚠ No data found in the collection.")
        exit()
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)
    print(f"✅ Retrieved {len(df)} records from MongoDB.")

    # Step 3: Display Data
    try:
        from IPython.display import display
        display(df)  # Works in Jupyter Notebook
    except ImportError:
        print(df.head())  # Fallback for terminal output

    # Step 4: Save Data to CSV
    df.to_csv("queried_data.csv", index=False)
    print("✅ Data saved successfully as 'queried_data.csv'.")

except Exception as e:
    print(f"❌ Error while querying data: {e}")
