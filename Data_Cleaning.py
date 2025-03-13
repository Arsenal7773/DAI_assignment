import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("sample_dataset.csv")

#----------------------------------------------------------------------------
                            # Handle missing values 
#----------------------------------------------------------------------------

print("Dataset before removing missing rows:\n")
print(df.info())
print(df.describe(include = 'all'))

num_imputer = SimpleImputer(strategy = "median") #mean imputation
df[['Salary','Experience']] = num_imputer.fit_transform(df[['Salary','Experience']])

print("Dataset after replacing missing rows:\n")
print(df.info())
print(df.describe(include = 'all'))

#--------------------------------------------------------------------------------
                           # Handling duplicate records
#--------------------------------------------------------------------------------

print("Dataset before removing duplicates:\n")
print(df.info())
print(df.describe(include = "all"))

#identify duplicate rows
duplicates = df[df.duplicated()]
print("\nDuplicate rows found:\n",duplicates)

#remove duplicate rows
df.drop_duplicates(inplace = True)

df.reset_index(drop=True, inplace=True)

print("Duplicates Remaining:", df.duplicated().sum())  # Should return 0

print("\nDataset after removing duplicates:\n")
print(df.info())
print(df.describe(include = 'all'),"\n\n")

df.to_csv("sample_dataset2.csv",index = False)
#-------------------------------------------------------------------------------------
                               # Handling outliers
#-------------------------------------------------------------------------------------

#graphically detecting outliers using box-plot of column Salary before removing outliers
sns.boxplot(df['Salary'])
plt.title("Box Plot for Salary")
plt.show()

#compute Q1, Q3 and IQR
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

#compute bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("IQR range = ",lower_bound," to ",upper_bound)

#detect outliers
outlier = df[(df['Salary'] < lower_bound) | (df["Salary"] > upper_bound)]
print("Outliers detected:\n",outlier)

#removing outliers
df = df[(df['Salary'] >= lower_bound) & (df["Salary"] <= upper_bound)]

#detect outliers after removing outliers
outlier = df[(df['Salary'] < lower_bound) | (df["Salary"] > upper_bound)]
print("Outliers Detected:\n", outlier,"\n\n")

#graphically detecting outliers using box-plot of column Salary after removing outliers
sns.boxplot(df['Salary'])
plt.title("Box Plot for Salary after removing outlier")
plt.show()

