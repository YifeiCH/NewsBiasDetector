import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    roc_curve, auc
)
from imblearn.over_sampling import RandomOverSampler

def map_bias_to_binary(bias_label):
    if bias_label in ["left", "lean left"]:
        return "Democratic"
    elif bias_label in ["right", "lean right"]:
        return "Republican"
    else:
        return "Center"

def train_and_evaluate_logreg():
    processed_path = os.path.join("data", "processed", "Political_Bias_cleaned.csv")
    df = pd.read_csv(processed_path)

    # Preprocessing
    df.dropna(subset=["cleaned_text"], inplace=True)
    df["Bias_binary"] = df["Bias"].apply(map_bias_to_binary)
    df = df[df["Bias_binary"] != "Center"]

    X = df["cleaned_text"]
    y = df["Bias_binary"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    tfidf = TfidfVectorizer()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # Oversample
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_vec, y_train)

    # Train model
    logreg = LogisticRegression()
    logreg.fit(X_train_res, y_train_res)

    # Save model & vectorizer
    joblib.dump(tfidf, "tfidf_logreg.pkl")
    joblib.dump(logreg, "logreg_model.pkl")
    print("Saved tfidf_logreg.pkl and logreg_model.pkl")

    # Evaluate
    y_pred = logreg.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1_dem = f1_score(y_test, y_pred, pos_label="Democratic")
    f1_rep = f1_score(y_test, y_pred, pos_label="Republican")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("Logistic Regression Accuracy (with oversampling):", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save metrics
    metrics = {
        "accuracy":      round(accuracy, 4),
        "f1_democratic": round(f1_dem, 4),
        "f1_republican": round(f1_rep, 4),
        "macro_f1":      round(macro_f1, 4)
    }
    with open("metrics_logreg.json", "w") as f:
        json.dump(metrics, f)
    print("Saved metrics_logreg.json")

    # --------- Generate ROC Curve ---------
    y_prob = logreg.predict_proba(X_test_vec)[:, 1]  # prob for Republican
    y_true_bin = (y_test == "Republican").astype(int)

    fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Logistic Regression")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_logreg.png")
    plt.close()
    print("Saved roc_logreg.png")

if __name__ == "__main__":
    train_and_evaluate_logreg()
