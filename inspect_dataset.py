import pandas as pd
# Load the file (adjust filename if needed, e.g., 'politifact_fake (1).csv')
filename = "politifact_fake.csv"
df = pd.read_csv(filename)
# Inspect
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nSample text (first entry):")
if 'text' in df.columns:
    print(df['text'].iloc[0][:200] + "...")  # Preview text
elif 'statement' in df.columns:
    print(df['statement'].iloc[0][:200] + "...")
elif 'title' in df.columns:
    print(df['title'].iloc[0][:200] + "...")
# Check for labels (if present)
label_cols = [col for col in df.columns if 'label' in col.lower() or 'fake' in col.lower()]
if label_cols:
    print(f"\nLabel columns: {label_cols}")
    print(df[label_cols[0]].value_counts())
else:
    print("\nNo obvious label column found. This might be only fake news (label all as 0).")