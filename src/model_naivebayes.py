import os
import pandas as pd
import joblib  # for saving the model
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import RandomOverSampler

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

    # Map multi-class => two-class
    df['Bias_binary'] = df['Bias'].apply(map_bias_to_binary)

    # Remove 'Center'
    df = df[df['Bias_binary'] != "Center"]

    X = df['cleaned_text']
    y = df['Bias_binary']

    # 1. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Vectorize
    tfidf = TfidfVectorizer()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # 3. Oversample the minority class
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_vec, y_train)

    # 4. Use the best alpha found from your trials
    best_alpha = 0.01
    nb = MultinomialNB(alpha=best_alpha)
    nb.fit(X_train_resampled, y_train_resampled)

    # 5. Predict on the test set
    y_pred = nb.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1_dem = f1_score(y_test, y_pred, pos_label="Democratic")
    f1_rep = f1_score(y_test, y_pred, pos_label="Republican")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Naive Bayes (alpha={best_alpha}) Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the vectorizer and final NB model for future use
    joblib.dump(tfidf, "tfidf_nb.pkl")
    joblib.dump(nb, "nb_model.pkl")
    print("\nSaved tfidf_nb.pkl and nb_model.pkl")

    # 7. Save evaluation metrics
    metrics = {
        "accuracy":      round(accuracy, 4),
        "f1_democratic": round(f1_dem, 4),
        "f1_republican": round(f1_rep, 4),
        "macro_f1":      round(macro_f1, 4)
    }
    with open("metrics_naivebayes.json", "w") as f:
        json.dump(metrics, f)
    print("Saved metrics_naivebayes.json")

if __name__ == "__main__":
    train_and_evaluate_naive_bayes()
