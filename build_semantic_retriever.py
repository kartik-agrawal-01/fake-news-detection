import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Starting the semantic retriever build process...")

# --- Configuration ---
TRAIN_FILE = "train_processed.csv"
INDEX_FILE = "faiss_semantic_index.bin"
DOCS_FILE = "faiss_documents.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2' # A fast and effective model

# Check if the training file exists
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"'{TRAIN_FILE}' not found. Please ensure the processed training data is present.")

# --- 1. Load and Prepare Data ---
print(f"Loading real news from '{TRAIN_FILE}'...")
df = pd.read_csv(TRAIN_FILE)
# We only build the knowledge base from real news (label=1)
real_docs = df[df['label'] == 1]['text'].dropna().tolist()
print(f"Loaded {len(real_docs)} real news documents as the knowledge base.")

# --- 2. Generate Embeddings ---
print(f"Loading the Sentence Transformer model: '{MODEL_NAME}'...")
# This model is great for creating vector representations of sentences/paragraphs
model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings for all documents... (This may take a few minutes)")
# The encode method converts our list of documents into a list of vectors
doc_embeddings = model.encode(real_docs, show_progress_bar=True)

# Ensure embeddings are in a format FAISS can use (float32 numpy array)
doc_embeddings = np.array(doc_embeddings).astype('float32')
print(f"Embeddings generated. Shape: {doc_embeddings.shape}")

# --- 3. Build and Save the FAISS Index ---
print("Building the FAISS index...")
# Get the dimension of the vectors
d = doc_embeddings.shape[1]
# We use IndexFlatL2, a standard index for dense vector similarity search
index = faiss.IndexFlatL2(d)
# Add our document vectors to the index
index.add(doc_embeddings)

print(f"FAISS index built. Total vectors in index: {index.ntotal}")

# Save the index to a file
faiss.write_index(index, INDEX_FILE)
print(f"Index saved to '{INDEX_FILE}'")

# Save the list of documents separately, as the index only stores vectors
with open(DOCS_FILE, 'wb') as f:
    pickle.dump(real_docs, f)
print(f"Original documents saved to '{DOCS_FILE}'")

print("\nSemantic retriever build complete! You can now run the main application.")