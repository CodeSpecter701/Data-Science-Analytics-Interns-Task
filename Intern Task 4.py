import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
sns.set(style="whitegrid")

try:
    data = pd.read_csv("insurance.csv")

except FileNotFoundError:
    print(" ERROR: insurance.csv not found")
    exit()
print("Columns in dataset:", data.columns.tolist())
if "name" in data.columns:
    data.drop("name", axis=1, inplace=True)
expected_cols = ["age", "bmi", "smoker", "charges"]

for c in expected_cols:
    if c not in data.columns:
        print(f" Missing column: {c}")
        exit()
object_cols = data.select_dtypes(include="object").columns
for c in object_cols:
    data[c] = data[c].fillna(data[c].mode()[0])
num_cols = data.select_dtypes(include=["int64", "float64"]).columns
for c in num_cols:
    data[c] = data[c].fillna(data[c].median())
for c in ["age", "bmi", "charges"]:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data = data.dropna()
try:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="age", y="charges", data=data)
    plt.title("Age vs Charges")
    plt.show()
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="bmi", y="charges", data=data)
    plt.title("BMI vs Charges")
    plt.show()
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="smoker", y="charges", data=data)
    plt.title("Smoking vs Charges")
    plt.show()

except Exception as e:
    print(" Plotting error:", e)
data = pd.get_dummies(data, drop_first=True)
if "charges" not in data.columns:
    print(" charges column missing after encoding")
    exit()
X = data.drop("charges", axis=1)
y = data["charges"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
predictions = lr_model.predict(X_test)
mae_val = mean_absolute_error(y_test, predictions)
rmse_val = np.sqrt(mean_squared_error(y_test, predictions))
print("\nModel Evaluation:")
print(f"MAE: {mae_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")


try:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red"
    )
    plt.show()
except Exception as e:
    print(" Final plot error:", e)