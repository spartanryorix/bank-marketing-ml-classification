# bank-marketing-ml-classification

## 📌 Project Overview

This project applies supervised machine learning techniques to predict whether a customer will subscribe to a term deposit based on demographic, financial, and campaign-related features.

Two classification models were developed and compared:

- Decision Tree Classifier  
- Logistic Regression  

The objective was not only to build predictive models but also to analyze model behavior, particularly overfitting and generalization performance.

---

## 🧠 Problem Statement

Given structured banking data, predict whether a customer will subscribe to a term deposit (`y = yes/no`) using historical campaign and customer information.

This is a binary classification problem.

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 📊 Data Preprocessing

- Categorical variables encoded into numerical form  
- Missing values handled using `fillna()`  
- Relevant features selected for modeling  
- Stratified 80/20 train-test split to preserve class distribution  

---

## 🤖 Models Implemented

### 1️⃣ Decision Tree Classifier

- Built using Scikit-learn  
- Intentionally left unpruned to demonstrate overfitting behavior  
- Used to analyze variance and memorization effects  

**Results:**  
- Train Accuracy: 1.00  
- Test Accuracy: 0.89  
- Test F1-Score: 0.529  

**Observation:**  
The model achieved perfect training accuracy, indicating overfitting. Performance dropped on test data, highlighting variance issues.

---

### 2️⃣ Logistic Regression

- Implemented using Scikit-learn (`liblinear` solver)  
- Used for comparison of generalization performance  

**Results:**  
- Test Accuracy: 91.16%  
- Test Precision: 0.693  
- Test Recall: 0.387  
- Test F1-Score: 0.497  

**Observation:**  
Logistic Regression demonstrated better generalization with a smaller gap between training and test performance.

---

## 📈 Model Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- ROC Curve & AUC  

The comparison between both models illustrates the impact of overfitting and the importance of evaluating beyond accuracy.

---

## 📂 Project Structure

```
bank-marketing-ml-classification/
│
├── README.md
├── bank dataset.csv
├── decision_tree.py
└── logistic_regression.py
```

---

## ▶️ How to Run

1. Install dependencies:

```
pip install pandas numpy matplotlib scikit-learn
```

2. Run either model:

```
python decision_tree.py
```

or

```
python logistic_regression.py
```

---

## 🎯 Key Learning Outcomes

- Understanding supervised classification  
- Detecting and analyzing overfitting  
- Comparing model generalization  
- Interpreting evaluation metrics beyond accuracy  
- Visualizing feature importance and ROC curves  

---

This project was developed as part of an HND Computing module in Machine Learning.

