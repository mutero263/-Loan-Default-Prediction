import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

#  Load & Clean Data 
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"
with open(file_path, 'r') as file:
    content = file.read()
values = [v.strip() for v in content.replace('\n', '').replace('\r', '').split(',') if v.strip()]

cleaned_values = []
for v in values:
    if v.lower().startswith('default') and len(v) > 7:
        num_part = ''.join(filter(str.isdigit, v))
        cleaned_values.append('default')
        if num_part:
            cleaned_values.append(num_part)
    else:
        cleaned_values.append(v)

n_cols = 9
num_rows = len(cleaned_values) // n_cols
data_rows = [cleaned_values[i*n_cols:(i+1)*n_cols] for i in range(num_rows)]
columns = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'default']
df = pd.DataFrame(data_rows, columns=columns)

for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df[df['default'].isin([0, 1])].dropna().reset_index(drop=True)

# Analyze 'debtinc' 
col = 'debtinc'
print(f"\nLOGICAL ANALYSIS: {col.upper()}")
print("="*50)
print(f"• Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}, Std: {df[col].std():.2f}")
print(f"• Range: {df[col].min():.2f} → {df[col].max():.2f}")
print(f"• Missing: {df[col].isnull().sum()}")

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
print(f"• Outliers Detected: {len(outliers)}")

avg_by_default = df.groupby('default')[col].mean()
print(f"\n• Average {col} by Loan Status:")
for status, avg in avg_by_default.items():
    label = "Bad (Default)" if status == 1 else "Good (No Default)"
    print(f"  {label}: {avg:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df[col], kde=True, ax=axes[0], color='indigo')
sns.boxplot(y=df[col], ax=axes[1], color='lavender')
sns.barplot(data=df, x='default', y=col, ax=axes[2], palette='husl')
axes[0].set_title(f'Distribution of {col}')
axes[1].set_title(f'Boxplot of {col}')
axes[2].set_title(f'{col} vs Loan Default')
axes[2].set_xticklabels(['Good (0)', 'Bad (1)'])
plt.tight_layout()
plt.show()