import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "5ai.csv" 
df = pd.read_csv(file_path)

col1 = "Internal" # Replace with the first col name
col2 = "External" # Replace with the second col name
selected_data = df[[col1, col2]]
plt.figure(figsize=(8, 6))
plt.scatter(selected_data[col1], selected_data[col2], color="blue", alpha=0.7)
plt.title(f"Scatter Plot of {col1} vs {col2}")
plt.xlabel(col1)
plt.ylabel(col2)
plt.grid(True)
plt.show()

correlation_coefficient = np.corrcoef(selected_data[col1],selected_data[col2])[0, 1]
print(f"Pearson Correlation Coefficient ({col1} vs {col2}):{correlation_coefficient}")

cov_matrix = np.cov(selected_data[col1], selected_data[col2])
print("\nCovariance Matrix:")
print(cov_matrix)

correlation_matrix = selected_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()

col3="Total"
col4="Internal"
selected_data=df[[col3,col4]]

plt.figure(figsize=(8,5))
plt.scatter(selected_data[col3],selected_data[col4],color='blue',alpha=0.8)
plt.title(f' scatter plot of {col3} vs {col4}')
plt.xlabel(col3)
plt.ylabel(col4)
plt.grid(True)
plt.show()