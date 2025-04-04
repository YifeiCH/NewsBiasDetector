# src/data_prep.py
import pandas as pd
import re
from nltk.corpus import stopwords

def load_data_local(file_path: str) -> pd.DataFrame:
    """
    Load local CSV from the 'data/raw/' folder.
    """
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    return df

def clean_text(text: str) -> str:
    """
    Example text preprocessing: lowercasing, removing punctuation, removing stopwords.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)
