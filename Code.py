import pandas as pd
import numpy as np

np.random.seed(42)

n = 500  # dataset size

data = {
    "Gender": np.random.choice(["Male", "Female"], n),
    "Married": np.random.choice(["Yes", "No"], n),
    "Dependents": np.random.choice(["0", "1", "2", "3+"], n),
    "Education": np.random.choice(["Graduate", "Not Graduate"], n),
    "Self_Employed": np.random.choice(["Yes", "No"], n),

    "ApplicantIncome": np.random.randint(1500, 20000, n),
    "CoapplicantIncome": np.random.randint(0, 8000, n),
    "LoanAmount": np.random.randint(50, 500, n),
    "Loan_Amount_Term": np.random.choice([120, 180, 240, 360], n),

    "Credit_History": np.random.choice([0.0, 1.0], n, p=[0.3, 0.7]),
    "Property_Area": np.random.choice(["Urban", "Rural", "Semiurban"], n)
}

# Create realistic target (Loan_Status)
# Simple rule-based probability
loan_status = []
for i in range(n):
    score = 0

    if data["Credit_History"][i] == 1.0:
        score += 2
    if data["ApplicantIncome"][i] > 5000:
        score += 1
    if data["LoanAmount"][i] < 300:
        score += 1
    if data["Education"][i] == "Graduate":
        score += 1

    if score >= 3:
        loan_status.append("Y")
    else:
        loan_status.append("N")

data["Loan_Status"] = loan_status

df = pd.DataFrame(data)

# Save CSV
df.to_csv("train.csv", index=False)

print("✅ Large dummy dataset created: train.csv")
print(df.head())
print("\nRows:", len(df))