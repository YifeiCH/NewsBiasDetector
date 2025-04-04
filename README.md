# Political Bias Detection in News Articles

**Authors**:  
- Yifei Chen  
- Bazil Akram  
- Andrew Abdelshahed

## Overview

This project aims to detect the political bias (Democratic vs. Republican) in news articles. We compare **Naive Bayes** and **Logistic Regression** to determine which classifier performs better on our dataset. The code is written in Python using popular libraries such as `pandas`, `scikit-learn`, and `nltk`.

## Getting Started

### Prerequisites

1. **Python 3.10+** (or higher)
2. **Pip** or **Conda** for package management

### Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your_username/PoliticalBiasNews.git
   cd PoliticalBiasNews
   ```

2. **Create and Activate a Virtual Environment (Recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install Requirements**  
   ```bash
   pip install -r requirements.txt
   ```
   > If `requirements.txt` is missing, you can manually install packages such as:
   > ```bash
   > pip install numpy pandas scikit-learn nltk
   > ```

4. **Download NLTK Stopwords** (if not already installed)  
   ```python
   import nltk
   nltk.download('stopwords')
   ```

### Dataset

- We use a public dataset of news articles labeled with political bias (Democratic vs. Republican).  
- You can place the dataset file (e.g., `your_news_dataset.csv`) under `data/` or wherever makes sense in your folder structure.

## Usage

1. **Run Preprocessing & Feature Extraction**  
   - Inside `src/preprocess.py` (example), a script cleans and transforms text into TF-IDF vectors (or Bag-of-Words).  

2. **Train the Models**  
   - **Naive Bayes** and **Logistic Regression** are both trained on the same train set.  
   - For instance, `python src/train.py` might:
     1. Load the dataset.  
     2. Preprocess text.  
     3. Split into train/test sets.  
     4. Fit both classifiers.  
     5. Print accuracy, classification report, confusion matrix, etc.

3. **Evaluate and Compare**  
   - Check the printed metrics or generated results (e.g., a CSV of predictions, console output, or a visualization).  
   - Decide which model is better based on metrics like accuracy, precision, recall, or F1 score.

### Example Command (if you have a single script):
```bash
cd src
python train.py
```
This will:
- Read `data/your_news_dataset.csv`  
- Preprocess text  
- Split train/test data  
- Train both Naive Bayes & Logistic Regression  
- Print evaluation metrics  

## Results

- Example metrics printed by the code:

  ```
  Naive Bayes Accuracy: 0.85
  Classification Report (Naive Bayes):
                precision    recall  f1-score   support
    ...
  
  Logistic Regression Accuracy: 0.87
  Classification Report (Logistic Regression):
                precision    recall  f1-score   support
    ...
  ```

- A **confusion matrix** will show how often each class (Democratic or Republican) was correctly identified.

## Future Enhancements

1. **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to optimize `alpha` in Naive Bayes or `C` in Logistic Regression.  
2. **Advanced NLP**: Experiment with word embeddings (GloVe, FastText) or transformer-based models (BERT).  
3. **More Classes**: Expand beyond a simple Democrat/Republican bias to other parties or to detect more nuanced biases.  
4. **Deployment**: Expose the model via a simple web API (Flask, FastAPI, etc.) to classify new, incoming articles in real-time.

## Contributing

1. **Fork** this repository  
2. Create a **new branch** for your changes  
3. **Commit** your changes  
4. Submit a **Pull Request**  

We welcome improvements, bug reports, and new feature suggestions!

## License

You can include a license of your choice (e.g., MIT, Apache 2.0). If unsure, the MIT license is a popular, permissive option.

---

**Contact**:  
- If you have any questions, feel free to reach out via GitHub Issues or directly to any of the project contributors.  
- **Yifei Chen** – [email/contact info if desired]  
- **Bazil Akram** – [email/contact info if desired]  
- **Andrew Abdelshahed** – [email/contact info if desired]

Happy Coding!  

---  

> _Disclaimer: This project is meant for educational purposes. Real-world detection of political bias can be complex and context-sensitive. Always verify performance and ethical considerations if deploying such systems in production._
