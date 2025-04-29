import os
import joblib
import json
from flask import Flask, request, jsonify, send_from_directory

# ---------------- load vectorizers & models once ----------------
tfidf_lr  = joblib.load("tfidf_logreg.pkl")
logreg    = joblib.load("logreg_model.pkl")
tfidf_nb  = joblib.load("tfidf_nb.pkl")
nb_model  = joblib.load("nb_model.pkl")

# ---------------- load pre-saved metrics ----------------
with open("metrics_logreg.json") as f:
    metrics_logreg = json.load(f)

with open("metrics_naivebayes.json") as f:
    metrics_naivebayes = json.load(f)

# ---------------- Flask setup ----------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# ---------------- helper function ----------------
def predict(text: str):
    # Logistic Regression prediction
    X_lr = tfidf_lr.transform([text])
    proba_lr = logreg.predict_proba(X_lr)[0]
    label_lr = logreg.classes_[proba_lr.argmax()]
    conf_lr  = round(proba_lr.max(), 3)

    # Naive Bayes prediction
    X_nb = tfidf_nb.transform([text])
    proba_nb = nb_model.predict_proba(X_nb)[0]
    label_nb = nb_model.classes_[proba_nb.argmax()]
    conf_nb  = round(proba_nb.max(), 3)

    # Select winner based on confidence
    winner = label_lr if conf_lr >= conf_nb else label_nb

    return {
        "logreg": {
            "label": label_lr,
            "conf": conf_lr,
            "metrics": metrics_logreg
        },
        "naive_bayes": {
            "label": label_nb,
            "conf": conf_nb,
            "metrics": metrics_naivebayes
        },
        "winner": winner
    }

# ---------------- routes ----------------
@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def api_predict():
    text = request.get_json().get("text", "")
    return jsonify(predict(text))

# ---------------- run app ----------------
if __name__ == "__main__":
    app.run(debug=True)
