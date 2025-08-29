import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

#  Read raw file ===
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"

try:
    with open(file_path, 'r') as file:
        content = file.read()
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Clean and split all values by comma ===
values = [v.strip() for v in content.replace('\n', '').replace('\r', '').split(',') if v.strip()]

#  Fix merged 'default41' → split into 'default' and number ===
cleaned_values = []
for v in values:
    if v.lower().startswith('default') and len(v) > 7:
        num_part = ''.join(filter(str.isdigit, v))
        cleaned_values.append('default')
        if num_part:
            cleaned_values.append(num_part)
    else:
        cleaned_values.append(v)

# Group every 9 values into a row ===
n_cols = 9
num_rows = len(cleaned_values) // n_cols
data_rows = []

for i in range(num_rows):
    start_idx = i * n_cols
    end_idx = start_idx + n_cols
    row = cleaned_values[start_idx:end_idx]
    if len(row) == n_cols:
        data_rows.append(row)

# Define column names ===
columns = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'default']

# Create DataFrame ===
df = pd.DataFrame(data_rows, columns=columns)

# Convert all columns to numeric ===
for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter valid 'default' values (0 or 1 only) ===
df = df[df['default'].isin([0, 1])]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f" Cleaned dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Analyze 'age' - Split Density Plot ===
col = 'age'

# Separate data by default status
not_default = df[df['default'] == 0][col]
default = df[df['default'] == 1][col]

# Create figure
plt.figure(figsize=(10, 6))

# Histograms (with density to match kde scale)
plt.hist(not_default, bins=20, alpha=0.5, color='skyblue', label='No Default (0)', density=True)
plt.hist(default, bins=20, alpha=0.5, color='crimson', label='Default (1)', density=True)

# Density curves (corrected: kdeplot, not kedePlot)
sns.kdeplot(not_default, color='skyblue', linewidth=2, label='No Default (0)')
sns.kdeplot(default, color='crimson', linewidth=2, label='Default (1)')

# Labels and title
plt.title(f'Split Density Plot of {col} by Loan Status', fontsize=16, pad=20)
plt.xlabel(col.capitalize(), fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Show plot
plt.tight_layout()
plt.show()