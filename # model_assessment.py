 # model_assessment.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTENSOR_FLAGS'] = 'cxx='  # Suppress g++ warning

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
        print(f"❌ File not found: {file_path}")
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

    print(f"✅ Cleaned dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    return df

def run_model_assessment():
    """Run full model assessment for both Traditional and Bayesian Logistic Regression"""
    df = load_and_clean_data()
    if df is None:
        return

    X = df.drop('default', axis=1)
    y = df['default']

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to float
    X_train_scaled = X_train_scaled.astype(float)
    X_test_scaled = X_test_scaled.astype(float)
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    # === 1. Traditional Logistic Regression ===
    print("\n" + "="*80)
    print("1. TRADITIONAL LOGISTIC REGRESSION MODEL")
    print("="*80)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score, mean_squared_error, average_precision_score,
        roc_curve, precision_recall_curve
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    lr_accuracy = lr.score(X_test_scaled, y_test)
    lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba_lr))
    lr_bias = np.mean(y_pred_proba_lr - y_test)
    lr_precision = average_precision_score(y_test, y_pred_proba_lr)

    # AIC & BIC
    n = len(y_train)
    k = X_train_scaled.shape[1]
    y_proba_train = lr.predict_proba(X_train_scaled)[:, 1]
    log_likelihood = -np.sum(
        y_train * np.log(y_proba_train + 1e-8) +
        (1 - y_train) * np.log(1 - y_proba_train + 1e-8)
    )
    lr_aic = 2 * k - 2 * log_likelihood
    lr_bic = np.log(n) * k - 2 * log_likelihood

    print(f"Accuracy:  {lr_accuracy:.4f}")
    print(f"AUC:       {lr_auc:.4f}")
    print(f"RMSE:      {lr_rmse:.4f}")
    print(f"AIC:       {lr_aic:.4f}")
    print(f"BIC:       {lr_bic:.4f}")
    print(f"Bias:      {lr_bias:.4f}")
    print(f"Precision: {lr_precision:.4f}")

    # === 2. Bayesian Logistic Regression ===
    print("\n" + "="*80)
    print("2. BAYESIAN LOGISTIC REGRESSION MODEL")
    print("="*80)

    try:
        import pymc as pm
        import arviz as az

        with pm.Model() as bayesian_model:
            # Priors
            intercept = pm.Normal('Intercept', mu=0, sigma=10)
            coeffs = pm.Normal('Coefficients', mu=0, sigma=5, shape=X_train_scaled.shape[1])

            # Linear predictor
            mu = intercept + pm.math.dot(X_train_scaled, coeffs)

            # Likelihood
            likelihood = pm.Bernoulli('y', p=pm.math.sigmoid(mu), observed=y_train)

            # Sample with log_likelihood enabled
            trace = pm.sample(
                draws=100,
                tune=100,
                chains=2,
                target_accept=0.99,
                return_inferencedata=True,
                random_seed=42,
                idata_kwargs={"log_likelihood": True}  # Required for WAIC/LOO
            )

        # Posterior Predictive Check
        with bayesian_model:
            ppc = pm.sample_posterior_predictive(trace, var_names=['y'])

        # Predict on test set using posterior mean
        coeff_mean = trace.posterior['Coefficients'].mean(dim=['chain', 'draw']).values
        intercept_mean = trace.posterior['Intercept'].mean(dim=['chain', 'draw']).values
        y_pred_proba_bayes = 1 / (1 + np.exp(-(intercept_mean + X_test_scaled @ coeff_mean)))

        # Metrics
        y_pred_bayes = (y_pred_proba_bayes > 0.5).astype(int)
        bayes_accuracy = np.mean(y_pred_bayes == y_test)
        bayes_auc = roc_auc_score(y_test, y_pred_proba_bayes)
        bayes_rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba_bayes))
        bayes_bias = np.mean(y_pred_proba_bayes - y_test)
        bayes_precision = average_precision_score(y_test, y_pred_proba_bayes)

        # WAIC and LOO (Bayesian Information Criteria)
        try:
            waic = az.waic(trace)
            loo = az.loo(trace)

            # Safe access for different ArviZ versions
            try:
                waic_value = waic.estimates.loc['waic', 'value']
                loo_value = loo.estimates.loc['loo', 'value']
            except (AttributeError, KeyError):
                waic_value = getattr(waic, 'waic', np.nan)
                loo_value = getattr(loo, 'loo', np.nan)

            print(f"WAIC:      {waic_value:.4f}")
            print(f"LOO:       {loo_value:.4f}")
        except Exception as e:
            print(f"❌ Could not compute WAIC/LOO: {str(e)}")
            waic_value = np.nan
            loo_value = np.nan

        print(f"Accuracy:  {bayes_accuracy:.4f}")
        print(f"AUC:       {bayes_auc:.4f}")
        print(f"RMSE:      {bayes_rmse:.4f}")
        print(f"Bias:      {bayes_bias:.4f}")
        print(f"Precision: {bayes_precision:.4f}")

        # Posterior Predictive Check Plot
        plt.figure(figsize=(10, 6))
        az.plot_ppc(ppc, num_pp_samples=100, mean=True)
        plt.title('Posterior Predictive Check (Bayesian Model)')
        plt.show()

    except ImportError:
        print("❌ PyMC or ArviZ not installed. Install with: pip install pymc arviz")
        bayes_accuracy = bayes_auc = bayes_rmse = bayes_bias = bayes_precision = np.nan
        waic_value = loo_value = np.nan
    except Exception as e:
        print(f"❌ Bayesian model failed: {str(e)}")
        bayes_accuracy = bayes_auc = bayes_rmse = bayes_bias = bayes_precision = np.nan
        waic_value = loo_value = np.nan

    # === 3. Model Comparison Table ===
    print("\n" + "="*80)
    print("3. MODEL COMPARISON")
    print("="*80)

    comparison = pd.DataFrame({
        'Model': ['Traditional', 'Bayesian'],
        'AUC': [lr_auc, bayes_auc],
        'RMSE': [lr_rmse, bayes_rmse],
        'Bias': [lr_bias, bayes_bias],
        'Precision': [lr_precision, bayes_precision],
        'Complexity': [
            f"AIC: {lr_aic:.2f}",
            f"WAIC: {waic_value:.2f}" if not np.isnan(waic_value) else "N/A"
        ]
    })

    print(comparison.round(4))

    # === 4. ROC and PR Curves ===
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)

    plt.figure(figsize=(14, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr_lr, tpr_lr, label=f'Traditional (AUC = {lr_auc:.3f})', color='blue')
    if not np.isnan(bayes_auc):
        fpr_bayes, tpr_bayes, _ = roc_curve(y_test, y_pred_proba_bayes)
        plt.plot(fpr_bayes, tpr_bayes, label=f'Bayesian (AUC = {bayes_auc:.3f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # PR Curve
    plt.subplot(1, 2, 2)
    ap_lr = average_precision_score(y_test, y_pred_proba_lr)
    plt.plot(rec_lr, prec_lr, label=f'Traditional (AP = {ap_lr:.3f})', color='blue')
    if not np.isnan(bayes_precision):
        prec_bayes, rec_bayes, _ = precision_recall_curve(y_test, y_pred_proba_bayes)
        ap_bayes = average_precision_score(y_test, y_pred_proba_bayes)
        plt.plot(rec_bayes, prec_bayes, label=f'Bayesian (AP = {ap_bayes:.3f})', color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # === 5. Feature Importance (Odds Ratios) ===
    if hasattr(lr, 'coef_'):
        odds_ratios = np.exp(lr.coef_[0])
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Odds Ratio': odds_ratios})
        importance_df = importance_df.sort_values('Odds Ratio', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Odds Ratio', y='Feature', palette='viridis')
        plt.axvline(x=1, color='red', linestyle='--', label='No Effect (OR = 1)')
        plt.title('Traditional Model: Feature Importance (Odds Ratio)')
        plt.legend()
        plt.tight_layout()
        plt.show()

# 🔑 CRITICAL: Protect the entry point to fix multiprocessing error
if __name__ == '__main__':
    run_model_assessment()