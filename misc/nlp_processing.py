import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the extracted CSV file
df = pd.read_csv("queried_data.csv")

# Ensure 'text' column exists
if "text" not in df.columns:
    raise ValueError("❌ 'text' column is missing from CSV!")

# Step 1: Named Entity Recognition (NER)
def extract_entities(text):
    doc = nlp(text)
    entities = {"organizations": [], "locations": []}
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["locations"].append(ent.text)
    return entities

df["entities"] = df["text"].apply(extract_entities)

# Step 2: TF-IDF for Keyword Extraction
vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
tfidf_matrix = vectorizer.fit_transform(df["text"])
tfidf_keywords = vectorizer.get_feature_names_out()

# Plot TF-IDF Terms
plt.figure(figsize=(10, 5))
sns.barplot(x=tfidf_keywords, y=tfidf_matrix.sum(axis=0).A1)
plt.xticks(rotation=45)
plt.title("Top TF-IDF Terms")
plt.show()

# Step 3: LDA Topic Modeling
text_data = [[word for word in doc.split() if word not in stopwords.words('english')] for doc in df["text"]]
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# Display Topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx+1}: {topic}")

# Save Processed Data
df.to_csv("processed_data.csv", index=False)
print("✅ NLP processing complete! Results saved to 'processed_data.csv'")
