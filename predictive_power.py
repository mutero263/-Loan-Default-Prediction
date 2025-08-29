# predictive_power.py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read raw file
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"

with open(file_path, 'r') as file:
    content = file.read()

# Clean and split all values
values = [v.strip() for v in content.replace('\n', '').replace('\r', '').split(',') if v.strip()]

# Fix the 'default41' issue: split merged 'default' and number
cleaned_values = []
for v in values:
    if v.lower().startswith('default') and len(v) > 7:
        # Extract digits after 'default'
        num_part = ''.join(filter(str.isdigit, v))
        if num_part:
            cleaned_values.append('default')
            cleaned_values.append(num_part)
        else:
            cleaned_values.append('default')
    else:
        cleaned_values.append(v)

# Step 2: Group every 9 values into a row
n_cols = 9
num_rows = len(cleaned_values) // n_cols
data_rows = []

for i in range(num_rows):
    start_idx = i * n_cols
    end_idx = start_idx + n_cols
    row = cleaned_values[start_idx:end_idx]
    data_rows.append(row)

# Step 3: Define correct column names
columns = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'default']

# Step 4: Create DataFrame
df = pd.DataFrame(data_rows, columns=columns)

# Step 5: Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop any rows with missing/invalid values
df.dropna(inplace=True)

# Ensure 'default' is either 0 or 1
df['default'] = df['default'].astype(int)
valid_defaults = df['default'].isin([0, 1])
if not valid_defaults.all():
    print("Warning: Invalid values in 'default' column:", df['default'].unique())
    df = df[valid_defaults]

print(f"Cleaned dataset shape: {df.shape}")

# Step 6: Function to calculate GINI, IV
def calculate_gini_iv(df, target='default'):
    results = []
    for col in df.columns:
        if col == target:
            continue

        # Binning
        if df[col].nunique() > 10:
            bins = pd.cut(df[col], bins=10, duplicates='drop')
        else:
            bins = pd.cut(df[col], bins=df[col].nunique(), duplicates='drop')

        # Cross-tab
        grouped = pd.crosstab(bins, df[target])
        if grouped.shape[1] != 2:
            # Ensure two columns: 0 and 1
            if 0 not in grouped.columns:
                grouped[0] = 0
            if 1 not in grouped.columns:
                grouped[1] = 0
            grouped = grouped[[0, 1]].sort_index(axis=1)

        grouped.columns = ['non_event', 'event']
        total_event = grouped['event'].sum()
        total_non_event = grouped['non_event'].sum()

        # Avoid division by zero
        event_rate = grouped['event'] / (total_event + 1e-8)
        non_event_rate = grouped['non_event'] / (total_non_event + 1e-8)

        # IV per bin
        iv_per_bin = (non_event_rate - event_rate) * np.log((non_event_rate + 1e-8) / (event_rate + 1e-8))
        IV = iv_per_bin.sum()

        # GINI
        try:
            auc = roc_auc_score(df[target], df[col])
            GINI = 2 * auc - 1
        except:
            GINI = np.nan

        selected = 'Yes' if IV >= 0.02 else 'No'

        results.append({
            'Variable': col,
            'GINI': round(GINI, 3),
            'IV': round(IV, 3),
            'Selected': selected
        })

    return pd.DataFrame(results).sort_values(by='IV', ascending=False).reset_index(drop=True)

# Step 7: Run analysis
results = calculate_gini_iv(df, target='default')

# Step 8: Display
print("\n" + "="*60)
print("PREDICTIVE STRENGTH OF VARIABLES")
print("="*60)
print(results.to_string(index=False))

# Step 9: Plot IV
plt.figure(figsize=(10, 6))
sns.barplot(data=results, x='IV', y='Variable', palette='Blues_r')
plt.title('Information Value (IV) of Variables', fontsize=16)
plt.xlabel('Information Value (IV)')
plt.axvline(x=0.02, color='red', linestyle='--', label='IV Threshold (0.02)')
plt.legend()
plt.tight_layout()
plt.show()