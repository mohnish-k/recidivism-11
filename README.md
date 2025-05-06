# Recidivism Project

### Description

The **Recidivism Project** aims to address the problem of recidivism by utilizing machine learning models and various data analysis techniques. This repository contains code for analyzing and predicting recidivism in individuals using datasets, including processing text, generating embeddings, and performing query-based searches using the **FAISS** library for efficient nearest-neighbor search.

The project also leverages an **embedding model** to convert text data into vector embeddings, allowing for semantic similarity-based retrieval of relevant information from the dataset. This model improves the quality and relevance of responses, making it easier to query large datasets and provide context-aware insights related to recidivism.

Additionally, the project includes tools for building a **chatbot** using **OpenAI's API**, which interacts with data stored in a vector index for querying and providing responses related to recidivism. The frontend is built using **Streamlit**, providing a user-friendly interface where users can input questions and receive answers based on the analysis of research papers and data related to recidivism.

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/mohnish-k/recidivism-11.git
   cd recidivism-11
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up any required environment variables:
   - Make sure to configure the environment variables by updating the `.env` file in the project root (this file is not shared for security reasons).

### **Usage**

1. **Preprocessing Data**: 
   The `preprocess_text.py` file is used for cleaning and preparing text data for analysis. You can modify it to include additional preprocessing steps as needed.

2. **Generate Embeddings**:
   Run the following script to generate embeddings for your text data:
   ```bash
   python generate_embeddings.py
   ```

3. **Building FAISS Index**:
   The `faiss_index.py` script creates a FAISS index to store vector embeddings efficiently for querying. You can run the following to build the index:
   ```bash
   python faiss_index.py
   ```

4. **Querying Data**:
   Use the `search_faiss.py` file to search through the FAISS index. Pass your query to retrieve similar documents or results from the dataset:
   ```bash
   python search_faiss.py "your search query"
   ```

5. **Running the Chatbot**:
   The `rag_chatbot.py` and `openai_chatbot.py` scripts integrate OpenAI's GPT models to interact with the stored data. Run the chatbot as follows:
   ```bash
   python rag_chatbot.py
   ```

### **Code Structure**

```
recidivism-11/
│
├── config.py                  # Configuration file
├── db_connection.py           # Handles database connections
├── extract_text.py             # Text extraction from documents
├── faiss_index.py             # FAISS index creation for embeddings
├── generate_embeddings.py     # Script to generate embeddings
├── openai_chatbot.py          # OpenAI GPT-based chatbot
├── preprocess_text.py         # Text preprocessing functions
├── query_data.py              # Query handling script
├── rag_chatbot.py             # Retrieval-Augmented Generation chatbot
├── search_faiss.py            # Search through FAISS index
├── .env                       # Environment variables (not shared)
└── README.md                  # Project documentation (this file)
```

### **Dependencies**

The following Python packages are required for this project:

*   fastapi>=0.104.0
*   uvicorn>=0.23.2
*   pydantic>=2.4.2
*   streamlit>=1.28.0
*   python-dotenv>=1.0.0
*   faiss-cpu>=1.7.4
*   numpy>=1.24.0
*   sentence-transformers>=2.2.2
*   openai>=1.3.0
*   pymongo>=4.5.0
*   requests>=2.31.0

You can install all dependencies via the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
