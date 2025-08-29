import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Read file 
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"

try:
    with open(file_path, 'r') as file:
        content = file.read()
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Clean and split all values by comma 
values = [v.strip() for v in content.replace('\n', '').replace('\r', '').split(',') if v.strip()]

# Fix merged 'default41' → split into 'default' and number 
cleaned_values = []
for v in values:
    if v.lower().startswith('default') and len(v) > 7:
        num_part = ''.join(filter(str.isdigit, v))
        cleaned_values.append('default')
        if num_part:
            cleaned_values.append(num_part)
    else:
        cleaned_values.append(v)

# Group every 9 values into a row
n_cols = 9
num_rows = len(cleaned_values) // n_cols
data_rows = []

for i in range(num_rows):
    start_idx = i * n_cols
    end_idx = start_idx + n_cols
    row = cleaned_values[start_idx:end_idx]
    if len(row) == n_cols:
        data_rows.append(row)

# Define column names
columns = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'default']

# Create DataFrame
df = pd.DataFrame(data_rows, columns=columns)

# Convert all columns to numeric
for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter valid 'default' values (0 or 1 only)
df = df[df['default'].isin([0, 1])]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Cleaned dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Prepare features and target
X = df.drop('default', axis=1)
y = df['default']

# Add constant for statsmodels
X_sm = sm.add_constant(X)

# Fit Logistic Regression using statsmodels (for p-values)
print("\n" + "="*60)
print("TRADITIONAL LOGISTIC REGRESSION RESULTS")
print("="*60)

try:
    model = sm.Logit(y, X_sm).fit()

    # Extract coefficients and p-values
    coefficients = model.params
    p_values = model.pvalues

    # Create results table
    results_df = pd.DataFrame({
        'Variable': ['Intercept'] + X.columns.tolist(),
        'Coefficient': coefficients,
        'P-value': p_values
    }).round(4)

    # Add Variable Type (all continuous in this dataset)
    results_df['Variable Type'] = ['N/A'] + ['Continuous'] * len(X.columns)

    # Add Significance
    results_df['Significance'] = ['N/A'] + [
        'Significant' if p < 0.05 else 'Insignificant' for p in p_values[1:]
    ]

    # Print final table
    print("\nTable 5.1: Traditional Logistic Regression Results")
    print("="*80)
    print(results_df.to_string(index=False))

    # Model Evaluation (sklearn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_pred_prob = lr.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_pred_prob)
    acc = lr.score(X_test, y_test)

    print(f"\nModel Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

except Exception as e:
    print(f"Model fitting failed: {str(e)}")