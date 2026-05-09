import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("churn_data.csv")

print(df.head())

sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
sns.countplot(x="Exited",data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Gender",hue="Exited",data=df)
plt.title("Gender vs Churn")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Geography",hue="Exited",data=df)
plt.title("Geography vs Churn")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Age"],kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Balance"],kde=True)
plt.title("Balance Distribution")
plt.show()