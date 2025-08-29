import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the raw file
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"

with open(file_path, 'r') as file:
    content = file.read()

# Clean and split the data
# Replace all newlines and extra spaces
content = content.replace('\n', '').replace('\r', '').strip()

# Split by comma
values = content.split(',')

# Reconstruct rows of 9 columns each
n_cols = 9
total_values = len(values)

# Check if we have enough data
if total_values < n_cols:
    raise ValueError("Not enough data in the file.")

# Combine values into rows of 9
rows = []
i = 0

while i < len(values):
    # Take next 9 values
    chunk = values[i:i + n_cols]
    if len(chunk) == n_cols:
        rows.append(chunk)
    else:
        # Handle incomplete last row
        pass
    i += n_cols

# Extract header (first row)
header = rows[0]
data_rows = rows[1:]

# Check if header contains merged data like "default41"
last_header = header[-1]
if last_header.startswith('default') and len(last_header) > 7:
    try:
        # Split "default41" into "default" and "41"
        # Find where digits start
        split_idx = None
        for idx, char in enumerate(last_header):
            if char.isdigit():
                split_idx = idx
                break
        if split_idx is not None:
            header[-1] = last_header[:split_idx]  
            data_rows[0] = [last_header[split_idx:]] + data_rows[0][1:]  
    except Exception as e:
        print("Could not fix merged header:", e)

#  Create DataFrame
df = pd.DataFrame(data_rows, columns=header)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')  

# Drop rows with NaN 
df.dropna(inplace=True)

# Display basic info
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# Compute correlation matrix
correlation_matrix = df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.3f',
    square=True,
    cbar_kws={"shrink": 0.8},
    linewidths=0.5
)
plt.title('Correlation Matrix of Bank Loans Dataset', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Show correlation with 'default'
if 'default' in df.columns:
    print("\nCorrelation with Loan Default (target):")
    corr_with_default = correlation_matrix['default'].sort_values(key=lambda x: abs(x), ascending=False)
    print(corr_with_default.round(3))
else:
    print("\nWarning: 'default' column not found in data.")