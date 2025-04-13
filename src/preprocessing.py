import ssl

# -------------------------- Workaround for SSL certificate issues --------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Some older Python versions do not verify HTTPS certificates by default
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ------------------------------------------------------------------------------------------

import os
import pandas as pd
import nltk
import re

# Try loading the NLTK stopwords. If they're missing, download them on the fly.
try:
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Basic text cleaning steps:
    1. Convert to lowercase.
    2. Remove non-alphabetic characters (numbers, punctuation, etc.).
    3. Split into tokens and remove English stopwords.
    4. Re-join tokens into a single string.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and whitespace
    tokens = text.split()
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(tokens)


def main():
    """
    1. Reads 'Political_Bias.csv' from 'data/raw/'.
    2. Applies cleaning to the 'Text' column.
    3. Creates a new 'cleaned_text' column.
    4. Saves the processed data to 'data/processed/Political_Bias_cleaned.csv'.
    """
    raw_data_path = os.path.join('data', 'raw', 'Political_Bias.csv')

    # 1. Load the dataset
    df = pd.read_csv(raw_data_path)
    print("Columns in CSV:", df.columns.tolist())

    # 2. Clean the 'Text' column
    #    Adjust column name if your CSV uses a different name like "text" or "Content"
    df['cleaned_text'] = df['Text'].apply(clean_text)

    # 3. Save the cleaned dataset
    processed_path = os.path.join('data', 'processed', 'Political_Bias_cleaned.csv')
    df.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to: {processed_path}")


if __name__ == "__main__":
    main()
