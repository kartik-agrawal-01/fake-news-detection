import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load datasets
fake_df = pd.read_csv("politifact_fake.csv")
real_df = pd.read_csv("politifact_real.csv")

# Standardize text column
fake_df['statement'] = fake_df['title']
real_df['statement'] = real_df['title']

# Assign labels
fake_df['label'] = 0
real_df['label'] = 1

# Combine datasets
combined_df = pd.concat([fake_df[['statement', 'label']], real_df[['statement', 'label']]], ignore_index=True)

# Shuffle dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Apply preprocessing
combined_df['clean_text'] = combined_df['statement'].apply(preprocess_text)
combined_df = combined_df[combined_df['clean_text'].str.len() > 0].reset_index(drop=True)

# Split dataset
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    combined_df['clean_text'], combined_df['label'], test_size=0.1, random_state=42, stratify=combined_df['label']
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.111, random_state=42, stratify=train_val_labels
)

# Save splits
train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

train_df.to_csv('train_processed.csv', index=False)
val_df.to_csv('val_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)

print("Datasets combined, preprocessed, split, and saved successfully.")
