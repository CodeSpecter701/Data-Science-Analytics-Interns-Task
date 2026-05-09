import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = sns.load_dataset("iris")


print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


plt.figure(figsize=(6, 4))

sns.scatterplot(
    data=df,
    x="sepal_length",
    y="petal_length",
    hue="species"
)

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")

plt.show()


plt.figure(figsize=(6, 4))

sns.histplot(
    df["sepal_length"],
    kde=True
)

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")

plt.show()


plt.figure(figsize=(6, 4))

sns.boxplot(
    data=df,
    x="species",
    y="sepal_width"
)

plt.title("Sepal Width Distribution by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width")

plt.show()


plt.figure(figsize=(6, 4))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.show()