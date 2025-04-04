# src/model_train.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data_prep import load_data_local, clean_text

def train_bias_detection():
    # 1) Load the local CSV
    df = load_data_local("data/raw/political_bias.csv")  # Adjust filename if different

    # 2) Apply cleaning to the text column
    #    Replace "text" with the correct column name for your dataset
    df["clean_text"] = df["text"].apply(clean_text)

    # 3) Convert text to features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])

    # 4) Label column
    #    Replace "label" with the correct column name if it's different (e.g., "bias")
    y = df["label"]

    # 5) Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6) Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    print("Naive Bayes Accuracy:", nb_acc)

    # 7) Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
    print("Logistic Regression Accuracy:", lr_acc)

    return nb_model, lr_model, vectorizer