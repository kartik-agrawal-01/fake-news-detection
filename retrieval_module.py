import pandas as pd
import pickle
import os
import re  # For query cleaning
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet  # Explicit import for WordNet

# Optional: Wikipedia for dynamic facts (pip install wikipedia)
try:
    import wikipedia
    wikipedia.set_lang("en")  # English
    WIKI_ENABLED = True  # Set to False to disable
except ImportError:
    WIKI_ENABLED = False
    print("Wikipedia not installed – install with 'pip install wikipedia' for dynamic facts.")

# Download NLTK data if needed (one-time, with force for WordNet fix)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Additional for lemmatizer (fixes Windows bug)

# Force re-download WordNet if corrupted (fixes '_LazyCorpusLoader__args' error)
try:
    nltk.download('wordnet', force=True)
    nltk.download('omw-1.4', force=True)
    print("NLTK WordNet data refreshed successfully.")
except Exception as e:
    print(f"Warning: NLTK download issue: {e} – lemmatizer may fallback.")

# Explicitly load WordNet to avoid lazy loading errors
try:
    wordnet.synsets('test')  # Test load
    print("WordNet loaded successfully.")
except Exception as e:
    print(f"WordNet load error: {e} – using fallback lemmatizer.")

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Tokenize, lemmatize, remove stopwords (matches data cleaning) with fallback for WordNet errors."""
    try:
        tokens = word_tokenize(text.lower())
        # Lemmatize with fallback (if WordNet fails, use token as is)
        lemmatized_tokens = []
        for token in tokens:
            if token.isalpha() and token not in stop_words:
                try:
                    lemmatized = lemmatizer.lemmatize(token, pos='v')  # Verb pos for better lemmatization
                    lemmatized_tokens.append(lemmatized)
                except Exception:
                    lemmatized_tokens.append(token)  # Fallback to original token
        return lemmatized_tokens
    except Exception as e:
        print(f"Preprocess error: {e} – returning raw tokens.")
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalpha() and token not in stop_words]

# Paths
TRAIN_FILE = "train_processed.csv"
RETRIEVER_FILE = "bm25_retriever.pkl"

def build_bm25_retriever():
    """Build BM25 from real news in train_processed.csv."""
    print("Loading and preprocessing real news from train_processed.csv...")
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"{TRAIN_FILE} not found. Run Main_dataset_cleaning_and_loading.py first.")
    
    df = pd.read_csv(TRAIN_FILE)
    real_docs = df[df['label'] == 1]['text'].dropna().tolist()  # Real only (label=1)
    print(f"Knowledge base size: {len(real_docs)} real news samples")
    
    # Preprocess
    tokenized_corpus = [preprocess_text(doc) for doc in real_docs]
    
    # Build BM25
    print("BM25 retriever initialized successfully!")
    retriever = BM25Okapi(tokenized_corpus)
    
    # Save retriever + corpus
    with open(RETRIEVER_FILE, 'wb') as f:
        pickle.dump({'retriever': retriever, 'corpus': real_docs}, f)
    
    print(f"Retriever saved as '{RETRIEVER_FILE}' for use in RAG pipeline.")
    return retriever, real_docs

def load_bm25_retriever():
    """Load pre-built BM25."""
    if not os.path.exists(RETRIEVER_FILE):
        raise FileNotFoundError(f"{RETRIEVER_FILE} not found. Run build_bm25_retriever() first.")
    
    with open(RETRIEVER_FILE, 'rb') as f:
        data = pickle.load(f)
    print("BM25 retriever loaded from 'bm25_retriever.pkl'")
    return data['retriever'], data['corpus']

def retrieve_relevant_docs(query, k=3):
    """
    Retrieve top-k real news docs for query + optional Wikipedia.
    Returns: (docs: list[str], scores: list[float]) – exactly 2 items.
    """
    retriever, corpus = load_bm25_retriever()
    
    # Preprocess query
    tokenized_query = preprocess_text(query)
    
    # Get BM25 scores and top indices
    scores = retriever.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    
    # Extract docs and scores
    docs = [corpus[i] for i in top_indices]
    scores = [scores[i] for i in top_indices]
    
    # Optional Wikipedia hybrid (dynamic fact for latest news) with robust error handling and better query parsing
    if WIKI_ENABLED:
        print(f"[DEBUG] Wikipedia query: {query}")
        wiki_summary = None
        for attempt in range(2):  # Retry once
            try:
                # Better query parsing: Extract key terms (e.g., year + event for elections)
                clean_query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query).strip().lower()
                # Parse for specific events (e.g., "2019 Indian general election" from "Narendra Modi won the 2019 Indian general election")
                if 'won the' in clean_query and 'election' in clean_query:
                    # Extract event phrase (e.g., "2019 Indian general election")
                    event_match = re.search(r'(\d{4}\s+[a-z\s]+election)', clean_query)
                    if event_match:
                        parsed_query = event_match.group(1).strip()
                    else:
                        parsed_query = clean_query
                else:
                    parsed_query = clean_query
                
                wiki_summary = wikipedia.summary(parsed_query, sentences=2)  # Use parsed query
                print(f"[DEBUG] Wikipedia added (parsed '{parsed_query}'): {wiki_summary[:50]}...")
                docs.append(wiki_summary)
                scores.append(5.0)  # Boost for external fact
                break
            except wikipedia.exceptions.DisambiguationError as e:
                # If ambiguous, use first result
                search_results = wikipedia.search(parsed_query)
                if search_results:
                    try:
                        wiki_page = wikipedia.page(search_results[0])
                    except Exception as page_error:  # first result can itself be ambiguous or missing
                        print(f"[DEBUG] Wikipedia fallback failed (attempt {attempt+1}): {page_error} – skipping.")
                        if attempt == 1:
                            docs.append("No additional dynamic fact available – relying on KB.")
                            scores.append(2.0)
                        continue
                    wiki_summary = wiki_page.summary[:200]
                    docs.append(wiki_summary)
                    scores.append(4.0)
                    print(f"[DEBUG] Wikipedia fallback to first search: {wiki_summary[:50]}...")
                    break
            except Exception as e:
                print(f"[DEBUG] Wikipedia error (attempt {attempt+1}): {e} – skipping.")
                if attempt == 1:  # Last attempt failed, add neutral fallback
                    docs.append("No additional dynamic fact available – relying on KB.")
                    scores.append(2.0)  # Low boost for fallback
                continue
        
        if wiki_summary is None:
            print("[DEBUG] Wikipedia completely failed – no dynamic fact added.")
    
    return docs, scores  # Tuple of 2 lists

# Build if not exists
if not os.path.exists(RETRIEVER_FILE):
    build_bm25_retriever()
else:
    load_bm25_retriever()

print("Retrieval module ready! Import retrieve_relevant_docs for use.")