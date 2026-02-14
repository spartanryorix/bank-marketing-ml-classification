import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

# Read the data from CSV file into a dataframe
df = pd.read_csv('bank dataset.csv')

# Converting 'job' column to numeric
dct_job = {
    'housemaid': 1, 'services': 2, 'admin.': 3, 'blue-collar': 4,
    'technician': 5, 'retired': 6, 'management': 7, 'unemployed': 8,
    'self-employed': 9, 'unknown': 10, 'entrepreneur': 11, 'student': 12
}
df['job'] = df['job'].map(dct_job)

# Converting 'marital' column to numeric
dct_marital = {'single': 0, 'married': 1, 'divorced': 2, 'unknown': 3}
df['marital'] = df['marital'].map(dct_marital)

# Converting 'education' column to numeric
dct_education = {
    'basic.4y': 0, 'high.school': 1, 'basic.6y': 2, 'basic.9y': 3,
    'professional.course': 4, 'university.degree': 5, 'unknown': 6
}
df['education'] = df['education'].map(dct_education)

# Converting 'default' column to numeric
dct_default = {'no': 0, 'yes': 1, 'unknown': 2}
df['default'] = df['default'].map(dct_default)

# Converting 'housing' column to numeric
dct_housing = {'yes': 1, 'no': 0, 'unknown': 2}
df['housing'] = df['housing'].map(dct_housing)

# Converting 'loan' column to numeric
dct_loan = {'yes': 1, 'no': 0, 'unknown': 2}
df['loan'] = df['loan'].map(dct_loan)

# Converting 'contact' column to numeric
dct_contact = {'telephone': 1, 'cellular': 0}
df['contact'] = df['contact'].map(dct_contact)

# Converting 'month' column to numeric
dct_month = {
    'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
    'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
}
df['month'] = df['month'].map(dct_month)

# Converting 'day_of_week' column to numeric
dct_day = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
df['day_of_week'] = df['day_of_week'].map(dct_day)

# Converting 'poutcome' column to numeric
dct_poutcome = {'success': 1, 'failure': 0, 'nonexistent': 2}
df['poutcome'] = df['poutcome'].map(dct_poutcome)

# Converting target variable 'y' column to numeric
dct_y = {'yes': 1, 'no': 0}
df['y'] = df['y'].map(dct_y)

# Handling missing values
df.fillna(0, inplace=True)

# Extracting features and target
X = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'duration', 'campaign',
        'pdays', 'previous', 'poutcome',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
        'euribor3m', 'nr.employed']]

y = df['y']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# Predictions
y_train_predict = dtree.predict(X_train)
y_test_predict = dtree.predict(X_test)

# Probabilities for ROC
y_probs = dtree.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"Decision Tree ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Decision Tree Model")
plt.legend()
plt.show()

# Evaluation metrics
print("Train Accuracy:", accuracy_score(y_train, y_train_predict))
print("Test Accuracy:", accuracy_score(y_test, y_test_predict))

print("Train Precision:", precision_score(y_train, y_train_predict, zero_division=0))
print("Test Precision:", precision_score(y_test, y_test_predict, zero_division=0))

print("Train Recall:", recall_score(y_train, y_train_predict))
print("Test Recall:", recall_score(y_test, y_test_predict))

print("Train F1:", f1_score(y_train, y_train_predict))
print("Test F1:", f1_score(y_test, y_test_predict))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_predict)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Feature importance
feature_importance = dtree.feature_importances_
fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(fi_df["Feature"], fi_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Decision Tree Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# Display Decision Tree
plot_tree(
    dtree,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True
)
plt.show()

# User input
age = int(input("Enter age of the customer: "))

print("Job codes:", dct_job)
job = int(input("Enter job code: "))

print("Marital: single=0, married=1, divorced=2, unknown=3")
marital = int(input("Enter marital status: "))

print("Education codes:", dct_education)
education = int(input("Enter education level: "))

print("Default: no=0, yes=1, unknown=2")
default = int(input("Enter default: "))

print("Housing: yes=1, no=0, unknown=2")
housing = int(input("Enter housing loan status: "))

print("Loan: yes=1, no=0, unknown=2")
loan = int(input("Enter personal loan status: "))

print("Contact: telephone=1, cellular=0")
contact = int(input("Enter contact type: "))

print("Month codes:", dct_month)
month = int(input("Enter month: "))

print("Day: mon=0, tue=1, wed=2, thu=3, fri=4")
day_of_week = int(input("Enter day of week: "))

duration = int(input("Enter call duration: "))
campaign = int(input("Enter campaign contacts: "))
pdays = int(input("Enter pdays: "))
previous = int(input("Enter previous contacts: "))

print("Poutcome: success=1, failure=0, nonexistent=2")
poutcome = int(input("Enter previous outcome: "))

emp_var_rate = 1.1
cons_price_idx = 93.99
cons_conf_idx = -36.4
euribor3m = 4.85
nr_employed = 5191

new_cus = pd.DataFrame([[
    age, job, marital, education, default, housing, loan,
    contact, month, day_of_week, duration, campaign, pdays,
    previous, poutcome, emp_var_rate, cons_price_idx,
    cons_conf_idx, euribor3m, nr_employed
]], columns=X.columns)

prediction = dtree.predict(new_cus)

if prediction[0] == 1:
    print("The customer will most probably subscribe to a term deposit.")
else:
    print("The customer will most probably NOT subscribe to a term deposit.")
