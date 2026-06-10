import re
import pandas as pd
from datasets import load_dataset, interleave_datasets
from textblob import TextBlob
from sklearn.model_selection import train_test_split

def clean_sec_text(raw_text):
    if not isinstance(raw_text, str): return ""
    text = raw_text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity < -0.05: return 0
    elif polarity > 0.05: return 2
    else: return 1

def load_and_preprocess_data(n_samples=3000):
    stream1 = load_dataset("winterForestStump/10-K_sec_filings", split="001", streaming=True)
    stream2 = load_dataset("winterForestStump/10-K_sec_filings", split="002", streaming=True)
    combined_stream = interleave_datasets([stream1, stream2])
    
    df = pd.DataFrame(list(combined_stream.take(n_samples)))
    df['cleaned_text'] = df['Business'].apply(clean_sec_text)
    df['label'] = df['cleaned_text'].apply(get_sentiment)
  
    df = df[df['cleaned_text'].str.strip().astype(bool)]
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], 
        test_size=0.2, random_state=42, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test