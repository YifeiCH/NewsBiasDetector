# PoliticalNewsDetector

Detect political bias (Democratic vs. Republican) in news articles using **scikit-learn pipelines** for text classification.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Algorithms Used](#algorithms-used)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

## Overview
This project classifies news articles based on political bias—Democratic or Republican—using two machine learning algorithms: **Logistic Regression** and **Naive Bayes**. We harness **scikit-learn pipelines** to streamline text preprocessing (TF-IDF vectorization) and model training.

### Team Members
- **Yifei Chen**  
- **Bazil Akram**  
- **Andrew Abdelshahed**  

### Goals
- Compare **Logistic Regression** and **Naive Bayes** on news text data.  
- Use **scikit-learn pipelines** to simplify data vectorization and model building.  
- Determine which algorithm performs better at predicting political bias.

---

## Dataset
We use a publicly available dataset from **Kaggle**:  
[News Political Bias Dataset](https://www.kaggle.com/datasets/mayobanexsantana/political-bias)

1. **Raw Data**: Place the downloaded dataset in `data/raw/Political_Bias.csv`.  
2. **Preprocessed Data**: A cleaning script (`preprocessing.py`) will read the raw CSV and create a cleaned version in `data/processed/`.

---

## Algorithms Used
1. **Logistic Regression** (scikit-learn)  
2. **Naive Bayes** (scikit-learn, typically MultinomialNB)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix (optional for deeper insight)

---
## Project Structure
Below is an overview of the folder layout:

```
PoliticalNewsDetector/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Basic text cleaning and data loading
│   ├── model_logreg.py         # Logistic Regression code
│   ├── model_naivebayes.py     # Naive Bayes code
│   └── ...
├── tests/
│   └── test_models.py
├── requirements.txt
└── README.md
```

- **data/raw**: Original dataset files  
- **data/processed**: Cleaned or transformed data  
- **notebooks**: Jupyter notebooks for exploratory analysis and experimentation  
- **src**: Core project source code, including preprocessing and modeling scripts  
- **tests**: Optional directory for unit or integration tests  
- **requirements.txt**: Python dependencies  
- **README.md**: Project overview and usage instructions  

---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone <YOUR_REPO_URL>.git
   cd PoliticalNewsDetector
   ```

2. **Set Up a Virtual Environment (Recommended)**  
   ```bash
   # For example, using Python 3.10:
   python3.10 -m venv venv
   source venv/bin/activate   # On macOS/Linux
   # or
   venv\Scripts\activate.bat  # On Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download Dataset**  
   - Go to [Kaggle dataset page](https://www.kaggle.com/datasets/mayobanexsantana/political-bias).
   - Place the downloaded files in `data/raw`.

---

## Usage

1. **Preprocessing**  
   - Run the text preprocessing script to clean the data:
     ```bash
     python src/preprocessing.py
     ```
   - This produces a cleaned CSV file in `data/processed/`.

2. **Training and Evaluation**  
   - Train and evaluate Logistic Regression:
     ```bash
     python src/model_logreg.py
     ```
   - Train and evaluate Naive Bayes:
     ```bash
     python src/model_naivebayes.py
     ```
   - Compare accuracy, precision, recall, and F1-scores from both outputs.

3. **Results**  
   - Metrics can be printed to the console or saved to a file.  
   - Review confusion matrices, classification reports, or any plots for additional insights.

---

## Contributing
1. **Branching**: Create a new branch for each feature or bugfix.  
2. **Pull Requests**: Submit PRs for review before merging into the `main` branch.  
3. **Code Style**: Follow [PEP 8 guidelines](https://peps.python.org/pep-0008/) or any team-agreed standard.

---

## License
You may choose a license if you wish to make your project open-source. A common choice is the [MIT License](https://opensource.org/licenses/MIT).

---

## Contact
For any questions, reach out to the team members:
- **Yifei Chen**  
- **Bazil Akram**  
- **Andrew Abdelshahed**

---