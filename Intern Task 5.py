import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

try:
    data = pd.read_csv("bank_marketing.csv")
except FileNotFoundError:
    print("❌ bank_marketing.csv not found")
    exit()

print(data.head())
print(data.info())


obj_cols = data.select_dtypes(include="object").columns
for col in obj_cols:
    data[col] = data[col].fillna(data[col].mode()[0])


num_cols = data.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())


plt.figure(figsize=(6, 4))
sns.histplot(data["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(x="job", data=data)
plt.xticks(rotation=45)
plt.title("Job Distribution")
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x="marital", data=data)
plt.title("Marital Status Distribution")
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(x="job", hue="y", data=data)
plt.xticks(rotation=45)
plt.title("Loan Acceptance by Job")
plt.show()


encoders = {}

for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le


X = data.drop("y", axis=1)
y = data["y"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))
print("Classification Report:\n", classification_report(y_test, log_pred))


tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, tree_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, tree_pred))
print("Classification Report:\n", classification_report(y_test, tree_pred))


feature_imp = pd.Series(tree_model.feature_importances_, index=X.columns)
feature_imp = feature_imp.sort_values(ascending=False)

print("\nTop Important Features:")
print(feature_imp.head(10))


plt.figure(figsize=(8, 5))
sns.barplot(x=feature_imp.head(10), y=feature_imp.head(10).index)
plt.title("Top Features Influencing Loan Acceptance")
plt.show()


print("\nBusiness Insights:")
print("- Certain job categories respond better to loan offers.")
print("- Marital status influences loan acceptance.")
print("- Age groups show different acceptance behavior.")
print("- Targeted marketing can improve conversion rates.")