import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load & Clean Data 
file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"
with open(file_path, 'r') as file:
    content = file.read()
values = [v.strip() for v in content.replace('\n', '').replace('\r', '').split(',') if v.strip()]

cleaned_values = [v for v in values]
for i, v in enumerate(values):
    if v.lower().startswith('default') and len(v) > 7:
        num = ''.join(filter(str.isdigit, v))
        cleaned_values[i] = 'default'
        if num:
            cleaned_values.insert(i+1, num)

n_cols = 9
data_rows = [cleaned_values[i*n_cols:(i+1)*n_cols] for i in range(len(cleaned_values)//n_cols)]
df = pd.DataFrame(data_rows, columns=['age','ed','employ','address','income','debtinc','creddebt','othdebt','default'])

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df[df['default'].isin([0,1])].dropna().reset_index(drop=True)

# Analyze 'employ'
col = 'employ'
print(f"\nLOGICAL ANALYSIS: {col.upper()}")
print("="*50)
print(f"• Mean: {df[col].mean():.2f}, Range: {df[col].min():.0f} → {df[col].max():.0f}")
print(f"• Outliers: {len(df[(df[col] < df[col].quantile(0.25)-1.5*(df[col].quantile(0.75)-df[col].quantile(0.25))) | (df[col] > df[col].quantile(0.75)+1.5*(df[col].quantile(0.75)-df[col].quantile(0.25)))])}")

avg_by_default = df.groupby('default')[col].mean()
print(f"\n• Avg {col} by Loan Status:")
for s, a in avg_by_default.items():
    print(f"  {'Bad' if s==1 else 'Good'}: {a:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df[col], kde=True, ax=axes[0], color='plum')
sns.boxplot(y=df[col], ax=axes[1], color='orange')
sns.barplot(data=df, x='default', y=col, ax=axes[2], palette='Accent')
axes[0].set_title(f'Distribution of {col}')
axes[1].set_title(f'Boxplot of {col}')
axes[2].set_title(f'{col} vs Loan Default')
axes[2].set_xticklabels(['Good (0)', 'Bad (1)'])
plt.tight_layout()
plt.show()