import os
import pandas as pd
import nltk
import re

# If you're using NLTK stopwords, you'll need to download them once:
# nltk.download('stopwords')
from nltk.corpus import stopwords


def clean_text(text):
    """
    Basic text cleaning:
    1. Lowercase
    2. Remove non-alphabetic characters
    3. Remove stopwords
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and whitespace
    tokens = text.split()

    # remove English stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # re-join tokens
    return " ".join(tokens)


def main():
    # Path to the raw dataset (adjust filename if it's not exactly 'political_news.csv')
    raw_data_path = os.path.join('data', 'raw', 'Political_Bias.csv')

    # Read dataset
    df = pd.read_csv(raw_data_path)

    # Assume there's a column named 'text' or 'content' in the CSV
    # Replace 'text_column_name' with the actual column name in your CSV
    df['cleaned_text'] = df['text_column_name'].apply(clean_text)

    # Save to processed folder
    processed_path = os.path.join('data', 'processed', 'Political_Bias_cleaned.csv')
    df.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to: {processed_path}")


if __name__ == "__main__":
    main()
