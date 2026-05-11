import seaborn as sns
import matplotlib.pyplot as plt

# Extract - Load dataset
df = sns.load_dataset("tips")

# Display dataset
print("Original Dataset:")
print(df.head())

# Transform - Remove missing values
df = df.dropna()

print("\nCleaned Dataset:")
print(df.head())

# Load - Save cleaned data
df.to_csv("cleaned_tips.csv", index=False)

# -------------------------------
# Visualization 1 - Bar Chart
# -------------------------------

plt.figure(figsize=(6,4))

sns.barplot(
    x="day",
    y="total_bill",
    data=df
)

plt.title("Average Total Bill per Day")
plt.show()

# -------------------------------
# Visualization 2 - Scatter Plot
# -------------------------------

plt.figure(figsize=(6,4))

sns.scatterplot(
    x="total_bill",
    y="tip",
    data=df
)

plt.title("Total Bill vs Tip")
plt.show()

# -------------------------------
# Visualization 3 - Count Plot
# -------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x="day",
    data=df
)

plt.title("Customer Count per Day")
plt.show()
