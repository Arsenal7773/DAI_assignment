import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("sample_dataset2.csv")

# Summary statistics for numerical columns
print("Summary Statistics:\n")
print(df[['Salary', 'Experience']].describe())  # Mean, Median, Min, Max, etc.

# Compute additional metrics
print("\nAdditional Metrics:")
print("Mode:\n", df[['Salary', 'Experience']].mode())  # Most common value
print("Variance:\n", df[['Salary', 'Experience']].var())  # Variance
print("Skewness:\n", df[['Salary', 'Experience']].skew())  # Skewness (Symmetry)
print("Kurtosis:\n", df[['Salary', 'Experience']].kurtosis())  # Kurtosis (Peakedness)


# Frequency of categorical variables
print("\nFrequency Distribution of Departments:")
print(df['Department'].value_counts())  # Count occurrences of each category

print("\nFrequency Distribution of Gender:")
print(df['Gender'].value_counts())  


# Plot histogram for Salary
plt.figure(figsize=(8,5))
sns.histplot(df['Salary'], bins=10, kde=True)  # KDE adds smooth curve
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# Box plot for Salary
plt.figure(figsize=(6,4))
sns.boxplot(df['Salary'])
plt.title("Box Plot for Salary")
plt.show()

# Bar plot for Department distribution
plt.figure(figsize=(8,5))
sns.countplot(x=df['Department'], order=df['Department'].value_counts().index, palette='coolwarm')
plt.title("Department Frequency")
plt.xlabel("Department")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
