import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load and Clean Data (Embedded) ===
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"

try:
    with open(file_path, 'r') as file:
        content = file.read()
except FileNotFoundError:
    print(f" File not found: {file_path}")
    exit()

# Clean and split
values = [v.strip() for v in content.replace('\n', '').replace('\r', '').split(',') if v.strip()]

# Fix merged 'default41'
cleaned_values = []
for v in values:
    if v.lower().startswith('default') and len(v) > 7:
        num_part = ''.join(filter(str.isdigit, v))
        cleaned_values.append('default')
        if num_part:
            cleaned_values.append(num_part)
    else:
        cleaned_values.append(v)

# Group every 9 values
n_cols = 9
num_rows = len(cleaned_values) // n_cols
data_rows = [cleaned_values[i*n_cols:(i+1)*n_cols] for i in range(num_rows)]

# Create DataFrame
columns = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'default']
df = pd.DataFrame(data_rows, columns=columns)

# Convert to numeric
for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter valid defaults (0 or 1)
df = df[df['default'].isin([0, 1])]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Analyze 'age' ===
col = 'age'

print(f"\n LOGICAL ANALYSIS: {col.upper()}")
print("="*50)
print(f"• Data Type: {df[col].dtype}")
print(f"• Total Rows: {len(df)}")
print(f"• Unique Values: {df[col].nunique()}")
print(f"• Missing: {df[col].isnull().sum()}")

mean = df[col].mean()
median = df[col].median()
std = df[col].std()
min_val = df[col].min()
max_val = df[col].max()
print(f"\n• Mean: {mean:.2f}")
print(f"• Median: {median:.2f}")
print(f"• Std Dev: {std:.2f}")
print(f"• Range: {min_val:.0f} → {max_val:.0f}")

# Outliers (IQR)
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
print(f"• Outliers Detected: {len(outliers)}")

# Avg by default
avg_by_default = df.groupby('default')[col].mean()
print(f"\n• Average {col} by Loan Status:")
for status, avg in avg_by_default.items():
    label = "Bad (Default)" if status == 1 else "Good (No Default)"
    print(f"  {label}: {avg:.2f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title(f'Distribution of {col}')

sns.boxplot(y=df[col], ax=axes[1], color='lightcoral')
axes[1].set_title(f'Boxplot of {col}')

sns.barplot(data=df, x='default', y=col, ax=axes[2], palette='Set2')
axes[2].set_title(f'{col} vs Loan Default')
axes[2].set_xticklabels(['Good (0)', 'Bad (1)'])

plt.tight_layout()
plt.show()