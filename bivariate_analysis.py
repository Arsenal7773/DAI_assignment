import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("sample_dataset2.csv")

# Compute correlation matrix
corr_matrix = df[['Salary', 'Experience']].corr()

# Plot heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Scatter plot: Salary vs Experience
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Experience'], y=df['Salary'])
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Box plot: Salary distribution across Departments
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Department'], y=df['Salary'], palette='coolwarm')
plt.title("Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Salary")
plt.xticks(rotation=45)
plt.show()

# Violin plot: Salary by Department
plt.figure(figsize=(8,5))
sns.violinplot(x=df['Department'], y=df['Salary'], palette='coolwarm')
plt.title("Salary Distribution by Department (Violin Plot)")
plt.xlabel("Department")
plt.ylabel("Salary")
plt.xticks(rotation=45)
plt.show()

# Bar plot: Average Salary per Department
plt.figure(figsize=(8,5))
sns.barplot(x=df['Department'], y=df['Salary'], estimator=np.mean, palette='coolwarm')
plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.show()
