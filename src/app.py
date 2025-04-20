import joblib, os
from flask import Flask, request, jsonify, send_from_directory

# ---------------- load models once ----------------
tfidf_lr = joblib.load("tfidf_logreg.pkl")
logreg    = joblib.load("logreg_model.pkl")
tfidf_nb  = joblib.load("tfidf_nb.pkl")
nb_model  = joblib.load("nb_model.pkl")

# tell Flask where the static HTML lives
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# ---------- helpers ----------
def predict(text: str):
    X_lr = tfidf_lr.transform([text])
    proba_lr = logreg.predict_proba(X_lr)[0]
    label_lr = logreg.classes_[proba_lr.argmax()]
    conf_lr  = round(proba_lr.max(), 3)

    X_nb = tfidf_nb.transform([text])
    proba_nb = nb_model.predict_proba(X_nb)[0]
    label_nb = nb_model.classes_[proba_nb.argmax()]
    conf_nb  = round(proba_nb.max(), 3)

    winner = label_lr if conf_lr >= conf_nb else label_nb
    return {
        "logreg":      {"label": label_lr, "conf": conf_lr},
        "naive_bayes": {"label": label_nb, "conf": conf_nb},
        "winner": winner
    }

# ---------- routes ----------
@app.route("/", methods=["GET"])
def serve_frontend():
    """Send the single‑page HTML when visiting /"""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def api_predict():
    text = request.get_json().get("text", "")
    return jsonify(predict(text))

if __name__ == "__main__":
    app.run(debug=True)
