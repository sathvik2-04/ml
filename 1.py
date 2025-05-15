import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.DataFrame({
'Usn': ['22AI001','22AI002','22AI003','22AI004','22AI005','22AI006','22AI007','22AI008','22AI009','P22AI010','22AI011','22AI012','22AI013','22AI014','22AI015','22AI016','22AI017','22AI018','22AI019','22AI020','22AI021','22AI022','22AI023','22AI024','22AI025','22AI026','22AI027','22AI028','22AI029','22AI030','22AI031','22AI032','22AI033','22AI034','22AI035','22AI036','22AI037','22AI038','22AI039','22AI040','22AI041','22AI042','22AI043','22AI044','22AI045','22AI046','22AI047','22AI048','22AI049','22AI050','22AI051','22AI052','22AI053','22AI054','22AI055','22AI056','22AI057','22AI058','22AI059','22AI060','22AI061'],
'Marks': [58,76,78,84,80,72,68,68,80,86,89,79,73,75,72,76,73,77,68,76,78,63,71,81,81,74,73,69,82,78,83,70,85,81,73,83,63,63,78,88,73,66,74,77,79,71,55,75,71,81,85,79,65,80,70,75,66,74,77,81,74]
})
print(data.head())
num_col = 'Marks'
data_num = data[num_col]
print(data_num)
mean_val = data_num.mean()
median_val = data_num.median()
mode_val = data_num.mode()[0]
std_dev = data_num.std()
variance = data_num.var()
range_val = data_num.max() - data_num.min()

print("\nStatistics for Numerical Column:")
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Range: {range_val}")

plt.figure(figsize=(8, 5))
plt.hist(data_num, bins=10, color='skyblue', edgecolor='black')
plt.title(f"Histogram of {num_col}")
plt.xlabel(num_col)
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=data_num, color='lightgreen')
plt.title(f"Boxplot of {num_col}")
plt.show()

q1 = data_num.quantile(0.25)
q3 = data_num.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data_num[(data_num < lower_bound) | (data_num > upper_bound)]
print("\nOutliers:")
print(outliers)

def mark_group(Marks):
    if Marks <= 35:
        return 'Fail'
    elif 36 <= Marks <= 70:
        return 'Average'
    else:
        return 'Outstanding'
data['Mark Group'] = data['Marks'].apply(mark_group)

categorical_column = 'Mark Group'
data_cat = data[categorical_column]
category_counts = data_cat.value_counts()
print("\nCategory Frequencies:")
print(category_counts)

plt.figure(figsize=(8, 5))
category_counts.plot(kind='bar', color='coral', edgecolor='black')
plt.title(f"Bar Chart of {categorical_column}")
plt.xlabel(categorical_column)
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90,
colors=sns.color_palette('pastel'))
plt.title(f"Pie Chart of {categorical_column}")
plt.ylabel("") 
plt.show()
