import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def map_bias_to_binary(bias_label):
    """
    Convert the multi-class Bias labels to a binary scheme:
    - 'left' or 'lean left' => 'Democratic'
    - 'right' or 'lean right' => 'Republican'
    - everything else => 'Center' (which we then drop, if not needed)
    """
    if bias_label in ["left", "lean left"]:
        return "Democratic"
    elif bias_label in ["right", "lean right"]:
        return "Republican"
    else:
        return "Center"  # or "Neutral" if you prefer that naming

def train_and_evaluate_logreg():
    processed_path = os.path.join('data', 'processed', 'Political_Bias_cleaned.csv')
    df = pd.read_csv(processed_path)

    # Drop rows with missing cleaned_text
    df.dropna(subset=['cleaned_text'], inplace=True)

    # Map original Bias labels (5 classes) to 2 classes
    df['Bias_binary'] = df['Bias'].apply(map_bias_to_binary)

    # If you only want two classes, drop rows labeled "Center"
    df = df[df['Bias_binary'] != "Center"]

    # Now use these columns for training
    X = df['cleaned_text']
    y = df['Bias_binary']

    # 80/20 train/test split, with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build a pipeline: TF-IDF vectorizer + LogisticRegression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate_logreg()
