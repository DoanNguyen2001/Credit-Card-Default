# Credit Card Default Prediction

## 📌 Overview

This project aims to predict the likelihood of a customer defaulting on their credit card payments using machine learning techniques. It involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and performance optimization.

---

## 🧠 Problem Statement

The goal is to develop a classification model that can identify whether a credit card customer is likely to default on their next payment, based on their historical and demographic data.

---

## 📊 Dataset

The dataset includes customer information such as:

- Demographics (Age, Gender, Education, etc.)
- Payment History
- Bill Statements
- Previous Payment Amounts
- Credit Limits

> **Source**: UCI Machine Learning Repository - [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## ⚙️ Features

- `LIMIT_BAL`: Amount of given credit
- `SEX`, `EDUCATION`, `MARRIAGE`: Demographic features
- `AGE`: Age in years
- `PAY_0` to `PAY_6`: Past monthly payment history
- `BILL_AMT1` to `BILL_AMT6`: Monthly bill statements
- `PAY_AMT1` to `PAY_AMT6`: Amount paid in previous months
- `default.payment.next.month`: Target variable

---

## 🛠️ Tools & Libraries

- **Python**
- **pandas**, **NumPy** – Data handling
- **matplotlib**, **seaborn** – Data visualization
- **scikit-learn** – Modeling and evaluation
- **XGBoost**, **LightGBM** – Advanced machine learning
- **imbalanced-learn** – Handling imbalanced datasets

---

## 🔍 Exploratory Data Analysis (EDA)

- Distribution of default vs non-default cases
- Correlation matrix and feature importance
- Trends in payment behavior and bill amounts
- Outlier detection and handling

---

## 🧪 Modeling Approach

- Baseline models: Logistic Regression, Decision Trees
- Advanced models: Random Forest, XGBoost, LightGBM
- Data balancing using SMOTE/RandomUnderSampler
- Cross-validation and hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix

---

## 🧩 Key Insights

- Payment history is one of the strongest indicators of default
- Default rates tend to vary with education level and marital status
- Feature engineering and data balancing significantly improve performance

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-default.git
   cd credit-card-default
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

4. View the results and metrics in the console or exported files.

---
## 🔍 Model Performance Summary

From the classification report of the **fair model** (on a balanced test set of 9,000 samples):

- **Overall Accuracy**: **77%**
- **Class 0 (Non-default)**:  
  - Precision: **0.86**, Recall: **0.84**, F1-score: **0.85**
- **Class 1 (Default)**:  
  - Precision: **0.49**, Recall: **0.52**, F1-score: **0.50**
- **Macro Average F1-score**: **0.68**
- **Weighted Average F1-score**: **0.78**

➡️ **Insight**: The model performs strongly for non-default cases but has moderate performance in detecting defaults, highlighting the challenge of class imbalance and the need for further optimization.

---

## 📌 Future Work

- Integrate with a Flask/Django web app for real-time predictions
- Perform SHAP analysis for model interpretability
- Deploy the model using AWS or Streamlit

---

## 👨‍💻 Author

**Your Name**  
[LinkedIn](https://www.linkedin.com/nguyen-doan) | [GitHub](https://github.com/DoanNguyen2001)

---
