import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df=pd.read_csv("loan_data.csv")
print(df.head())
print(df.isnull().sum())

for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0],inplace=True)

for col in df.select_dtypes(include=["int64","float64"]).columns:
    df[col].fillna(df[col].median(),inplace=True)

plt.figure(figsize=(6,4))
sns.histplot(df["LoanAmount"],kde=True)
plt.title("Loan Amount Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["ApplicantIncome"],kde=True)
plt.title("Applicant Income Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Education",hue="Loan_Status",data=df)
plt.title("Education vs Loan Status")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Property_Area",hue="Loan_Status",data=df)
plt.title("Property Area vs Loan Status")
plt.show()

le=LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col]=le.fit_transform(df[col])

X=df.drop("Loan_Status",axis=1)
y=df["Loan_Status"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

models={"Logistic Regression":LogisticRegression(),"Decision Tree":DecisionTreeClassifier()}

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"\n{name} Results")
    print("Accuracy:",accuracy_score(y_test,y_pred))
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    print("Classification Report:\n",classification_report(y_test,y_pred))

print("\nConclusion: Decision Tree usually performs better but check overfitting.")