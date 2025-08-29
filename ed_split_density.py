# ed_split_density.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# === Load and Clean Data ===
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"
with open(file_path, 'r') as file:
    content = file.read()

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

# Filter valid defaults
df = df[df['default'].isin([0, 1])].dropna().reset_index(drop=True)

# === Analyze 'ed' ===
col = 'ed'
not_default = df[df['default'] == 0][col]
default = df[df['default'] == 1][col]

plt.figure(figsize=(10, 6))
plt.hist(not_default, bins=15, alpha=0.5, color='skyblue', label='No Default (0)', density=True)
plt.hist(default, bins=15, alpha=0.5, color='crimson', label='Default (1)', density=True)
sns.kdeplot(not_default, color='skyblue', linewidth=2, label='No Default (0)')
sns.kdeplot(default, color='crimson', linewidth=2, label='Default (1)')
plt.title(f'Split Density Plot of {col} by Loan Status', fontsize=16, pad=20)
plt.xlabel('Education (Years)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()