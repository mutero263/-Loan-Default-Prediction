import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTENSOR_FLAGS'] = 'cxx=' 

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_clean_data():
    """Load and clean the bankloans.csv file"""
    file_path = r"C:\Users\Algo-Tech Systems\Desktop\bankloans.csv"

    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f" File not found: {file_path}")
        return None

    # Clean and split
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

    print(f"Cleaned dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    return df

def run_bayesian_model():
    """Run Bayesian Logistic Regression using PyMC"""
    df = load_and_clean_data()
    if df is None:
        return

    X = df.drop('default', axis=1)
    y = df['default']

    # Standardize features
    X_scaled = (X - X.mean()) / X.std()
    X_scaled = X_scaled.astype(float)
    y = y.astype(float)

    print("\nFeatures standardized for Bayesian modeling.")

    try:
        import pymc as pm

        print("\n" + "="*80)
        print("BAYESIAN LOGISTIC REGRESSION MODEL RESULTS")
        print("="*80)

        with pm.Model() as bayesian_model:
            # Priors
            intercept = pm.Normal('Intercept', mu=0, sigma=10)
            coeffs = pm.Normal('Coefficients', mu=0, sigma=5, shape=X_scaled.shape[1])

            # Linear predictor
            mu = intercept + pm.math.dot(X_scaled.values, coeffs)

            # Likelihood
            likelihood = pm.Bernoulli('y', p=pm.math.sigmoid(mu), observed=y.values)

            # Sample
            trace = pm.sample(
                draws=1000,
                tune=1000,
                chains=2,
                target_accept=0.95,
                return_inferencedata=True,
                random_seed=42
            )

        # Summarize results
        summary = pm.summary(trace, round_to=4)
        summary_df = summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'ess_bulk', 'r_hat']].copy()
        summary_df.reset_index(inplace=True)
        summary_df.rename(columns={'index': 'Variable'}, inplace=True)

        # Map variable names
        var_names = ['Intercept'] + X.columns.tolist()
        summary_df['Variable'] = var_names

        # Add Significance (based on HDI excluding 0)
        significance = []
        for _, row in summary_df.iterrows():
            if row['Variable'] == 'Intercept':
                continue
            hdi_low = row['hdi_3%']
            hdi_high = row['hdi_97%']
            if hdi_low * hdi_high > 0:  
                significance.append('Significant')
            else:
                significance.append('Insignificant')
        summary_df['Significance'] = ['N/A'] + significance

        # Add Odds Ratio
        summary_df['Odds Ratio'] = np.exp(summary_df['mean'])

        # Reorder columns
        summary_df = summary_df[[
            'Variable', 'mean', 'sd', 'hdi_3%', 'hdi_97%', 'Odds Ratio', 'Significance', 'ess_bulk', 'r_hat'
        ]]

        # Print results
        print("\nBayesian Logistic Regression Summary")
        print("-" * 100)
        print(summary_df.to_string(index=False, float_format="%.4f"))

        # Convergence check
        print("\nConvergence Check:")
        print("• r_hat ≈ 1.0: Good convergence (should be < 1.05)")
        print("• ESS > 100: Reliable sampling")

        # Plot posterior distributions
        plt.figure(figsize=(10, 6))
        pm.plot_forest(trace, var_names=['Intercept', 'Coefficients'], combined=True, figsize=(10, 6))
        plt.title('Posterior Distributions of Coefficients')
        plt.xlabel('Coefficient Value')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Effect (0)')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot Odds Ratios with Credible Intervals
        odds_ratios = np.exp(summary_df[summary_df['Variable'] != 'Intercept'][['mean', 'hdi_3%', 'hdi_97%']])
        features = X.columns.tolist()

        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(features))
        plt.errorbar(
            odds_ratios['mean'], y_pos,
            xerr=[odds_ratios['mean'] - odds_ratios['hdi_3%'], odds_ratios['hdi_97%'] - odds_ratios['mean']],
            fmt='o', capsize=5, color='skyblue', label='Odds Ratio (94% HDI)'
        )
        plt.axvline(x=1, color='red', linestyle='--', label='No Effect (OR = 1)')
        plt.yticks(y_pos, features)
        plt.title('Bayesian Logistic Regression: Odds Ratios with 94% HDI')
        plt.xlabel('Odds Ratio (log scale)')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("PyMC not installed. Install with: pip install pymc")
    except Exception as e:
        print(f"Model sampling failed: {str(e)}")

if __name__ == '__main__':
    run_bayesian_model()