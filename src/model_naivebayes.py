import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler  # <-- for oversampling

def map_bias_to_binary(bias_label):
    """
    Convert multi-class Bias labels to binary:
    - 'left' or 'lean left' => 'Democratic'
    - 'right' or 'lean right' => 'Republican'
    - everything else => 'Center'
    """
    if bias_label in ["left", "lean left"]:
        return "Democratic"
    elif bias_label in ["right", "lean right"]:
        return "Republican"
    else:
        return "Center"

def train_and_evaluate_naive_bayes():
    processed_path = os.path.join('data', 'processed', 'Political_Bias_cleaned.csv')
    df = pd.read_csv(processed_path)

    # Drop rows with missing cleaned_text
    df.dropna(subset=['cleaned_text'], inplace=True)

    # Map multi-class => two-class labels
    df['Bias_binary'] = df['Bias'].apply(map_bias_to_binary)

    # Remove 'Center'
    df = df[df['Bias_binary'] != "Center"]

    X = df['cleaned_text']
    y = df['Bias_binary']

    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Convert raw text to TF-IDF features
    tfidf = TfidfVectorizer()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # 3. Oversample the training set to balance classes
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_vec, y_train)

    # 4. Train Naive Bayes on the oversampled data
    nb = MultinomialNB()
    nb.fit(X_train_resampled, y_train_resampled)

    # 5. Predict on the (non-oversampled) test set
    y_pred = nb.predict(X_test_vec)

    # 6. Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print("Naive Bayes Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate_naive_bayes()
