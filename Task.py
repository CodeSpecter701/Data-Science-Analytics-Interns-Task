import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ===============================
# CONFIG
# ===============================
folder_path = r"D:\Internship Project"

print("\n📂 Checking project folder...")
files = os.listdir(folder_path)
print(files)

# ===============================
# FIND CSV AUTOMATICALLY
# ===============================
csv_file = None

for f in files:
    if f.endswith(".csv"):
        csv_file = f
        break

# ===============================
# LOAD DATA OR STOP SAFELY
# ===============================
if csv_file is None:
    print("\n❌ ERROR: No CSV dataset found in folder!")
    print("👉 You only have .ipynb (not usable as dataset).")
    print("👉 Please download 'train.csv' from Kaggle and place it here:")
    print(folder_path)
    exit()
else:
    path = os.path.join(folder_path, csv_file)
    print(f"\n✅ Loading dataset: {csv_file}")
    df = pd.read_csv(path)

# ===============================
# BASIC PREVIEW
# ===============================
print("\n📊 Data Preview:")
print(df.head())

# ===============================
# MISSING VALUE HANDLING
# ===============================
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Self_Employed'] = df['Self_Employed'].fillna('No')

for col in ['Gender','Married','Dependents','Loan_Amount_Term','Credit_History']:
    df[col] = df[col].fillna(df[col].mode()[0])

# ===============================
# ENCODING
# ===============================
cols = ['Gender','Married','Dependents','Education',
        'Self_Employed','Property_Area','Loan_Status']

le = LabelEncoder()

for c in cols:
    df[c] = le.fit_transform(df[c])

# ===============================
# MODEL
# ===============================
X = df[['Credit_History','Gender','Married','Education']]
y = df['Loan_Status']

model = DecisionTreeClassifier()
model.fit(X, y)

pred = model.predict(X)

# ===============================
# RESULT
# ===============================
print("\n🎯 Accuracy:", accuracy_score(y, pred))