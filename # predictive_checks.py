import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTENSOR_FLAGS'] = 'cxx='  # Suppress g++ warning
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

    # Filter valid defaults
    df = df[df['default'].isin([0, 1])]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f" Cleaned dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    return df

def run_predictive_checks():
    """Run predictive checks for both Traditional and Bayesian Logistic Regression"""
    df = load_and_clean_data()
    if df is None:
        return

    X = df.drop('default', axis=1)
    y = df['default']

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.astype(float)
    y = y.astype(float)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    #  Traditional Logistic Regression Predictive Check 
    print("\n" + "="*80)
    print("1. TRADITIONAL LOGISTIC REGRESSION - PREDICTIVE CHECKS")
    print("="*80)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc, brier_score_loss
    from sklearn.calibration import calibration_curve

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    y_pred_proba = lr.predict_proba(X_test)[:, 1]

    # Predictive Check 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    #  Predictive Check 2: Calibration Plot 
    fraction_of_positives, mean_predicted = calibration_curve(y_test, y_pred_proba, n_bins=10)

    plt.subplot(1, 3, 2)
    plt.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend()

    #  Predictive Check 3: Brier Score & Residuals 
    brier = brier_score_loss(y_test, y_pred_proba)
    print(f"Brier Score (Mean Squared Calibration Error): {brier:.4f}")

    df_check = pd.DataFrame({'y_true': y_test, 'y_prob': y_pred_proba})
    df_check['bin'] = pd.cut(df_check['y_prob'], bins=10, labels=False)
    binned_residuals = df_check.groupby('bin').apply(lambda g: np.mean(g['y_true'] - g['y_prob']))

    plt.subplot(1, 3, 3)
    plt.bar(binned_residuals.index, binned_residuals.values, alpha=0.7, edgecolor='black')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Probability Bin')
    plt.ylabel('Average Residual')
    plt.title('Binned Residuals')
    plt.tight_layout()
    plt.show()

    print(" Traditional Model: Predictive checks completed.")

    # Bayesian Logistic Regression Predictive Check 
    print("\n" + "="*80)
    print("2. BAYESIAN LOGISTIC REGRESSION - PREDICTIVE CHECKS")
    print("="*80)

    try:
        import pymc as pm
        import arviz as az

        with pm.Model() as bayesian_model:
            # Priors
            intercept = pm.Normal('Intercept', mu=0, sigma=10)
            coeffs = pm.Normal('Coefficients', mu=0, sigma=5, shape=X_train.shape[1])

            # Linear predictor
            mu = intercept + pm.math.dot(X_train, coeffs)

            # Likelihood
            likelihood = pm.Bernoulli('y', p=pm.math.sigmoid(mu), observed=y_train)

            # Sample with log_likelihood
            trace = pm.sample(
                draws=100,
                tune=100,
                chains=2,
                target_accept=0.99,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
                random_seed=42
            )

        # Posterior Predictive Check
        with bayesian_model:
            ppc = pm.sample_posterior_predictive(trace, var_names=['y'])

        # Predicted probabilities (mean over posterior samples)
        ppc_probs = ppc.posterior_predictive['y'].mean(dim=['chain', 'draw'])

        # Ensure alignment with y_test
        if len(ppc_probs) != len(y_test):
            min_len = min(len(ppc_probs), len(y_test))
            y_test = y_test[:min_len]
            ppc_probs = ppc_probs[:min_len]

        # Predictive Check 1: PPC Plot 
        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        az.plot_ppc(ppc, num_pp_samples=100, mean=True, ax=plt.gca())
        plt.title('Posterior Predictive Check (PPC)')
        plt.ylabel('Density')

        # Predictive Check 2: Observed vs Predicted 
        plt.subplot(1, 3, 2)
        plt.scatter(y_test, ppc_probs, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.xlabel('Observed')
        plt.ylabel('Predicted (Bayesian)')
        plt.title('Observed vs Predicted Probabilities')

        # Predictive Check 3: Binned Residuals 
        df_bayes = pd.DataFrame({'y_true': y_test, 'y_prob': ppc_probs})
        df_bayes['bin'] = pd.cut(df_bayes['y_prob'], bins=10, labels=False)
        binned_residuals_bayes = df_bayes.groupby('bin').apply(
            lambda g: np.mean(g['y_true'] - g['y_prob'])
        )

        plt.subplot(1, 3, 3)
        plt.bar(binned_residuals_bayes.index, binned_residuals_bayes.values, alpha=0.7, color='green', edgecolor='black')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Probability Bin')
        plt.ylabel('Average Residual')
        plt.title('Bayesian: Binned Residuals')
        plt.tight_layout()
        plt.show()

        # Forest plot
        plt.figure(figsize=(10, 6))
        az.plot_forest(trace, var_names=['Intercept', 'Coefficients'], combined=True)
        plt.title('Posterior Distributions of Coefficients')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        print("Bayesian Model: Predictive checks completed.")

    except ImportError:
        print("Missing libraries: pip install pymc arviz")
    except Exception as e:
        print(f"Bayesian model failed: {str(e)}")

if __name__ == '__main__':
    run_predictive_checks()