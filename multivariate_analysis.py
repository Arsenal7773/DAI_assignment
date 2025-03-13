import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("sample_dataset2.csv")

# Pair plot for numerical variables
sns.pairplot(df[['Salary', 'Experience', 'Performance_Score']])
plt.show()

# Compute correlation matrix
corr_matrix = df[['Salary', 'Experience', 'Performance_Score']].corr()

# Plot heatmap
plt.figure(figsize=(8,5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#Box Plot (Salary Distribution by Department & Gender)
plt.figure(figsize=(10,5))
sns.boxplot(x='Department', y='Salary', hue='Gender', data=df, palette='coolwarm')
plt.title("Salary Distribution by Department & Gender")
plt.xticks(rotation=45)
plt.show()

#Grouped Bar Plot (Average Salary per Department by Gender)
plt.figure(figsize=(10,5))
sns.barplot(x='Department', y='Salary', hue='Gender', data=df, estimator=np.mean, palette='coolwarm')
plt.title("Average Salary by Department & Gender")
plt.xticks(rotation=45)
plt.show()

#Pivot Table (Analyzing Multiple Features)
pivot_table = df.pivot_table(values='Salary', index='Department', columns='Gender', aggfunc='mean')
print(pivot_table)