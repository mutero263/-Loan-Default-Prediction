# Loan Default Prediction Project

This project analyzes a loan applicant dataset to predict the likelihood of loan default using **Traditional (Frequentist) and Bayesian Logistic Regression models**. It includes full data cleaning, exploratory analysis, model training, validation, and comparison.

---

##  Project Overview

| Feature | Description |
|--------|-------------|
| **Dataset** | `bankloans.csv` — contains 110 records of loan applicants |
| **Target Variable** | `default` (0 = No Default, 1 = Default) |
| **Predictors** | age, ed, employ, address, income, debtinc, creddebt, othdebt |
| **Modeling** | Traditional Logistic Regression + Bayesian Logistic Regression |
| **Validation** | ROC, Precision-Recall, Calibration, Posterior Predictive Checks |
| **Evaluation Metrics** | AUC, RMSE, AIC/BIC, WAIC/LOO, Bias, Precision |


##  Data Cleaning

The `bankloans.csv` file is **malformed**:
- No line breaks
- Merged column name: `default41` → split into `'default'` and `41`

 The code:
- Parses the single-line CSV
- Fixes the `default41` issue
- Converts all values to numeric
- Filters valid defaults (0 or 1)

---

##  Models Implemented

### 1. **Traditional Logistic Regression**
- Built using `scikit-learn`
- Coefficients, p-values (via `statsmodels`)
- Model fit: AIC, BIC
- Predictive checks: ROC, Calibration, Brier Score

### 2. **Bayesian Logistic Regression**
- Built using `pymc` and `arviz`
- Posterior distributions of coefficients
- Uncertainty quantification via credible intervals
- Model fit: WAIC, LOO
- Predictive checks: Posterior Predictive Checks (PPC)

---

##  Model Assessment

| Metric | Purpose |
|-------|--------|
| **AUC / ROC** | Discriminatory power |
| **RMSE** | Calibration of predicted probabilities |
| **Bias** | Average deviation of predictions from truth |
| **Precision** | Positive predictive value |
| **AIC / BIC** | Frequentist model fit (penalizes complexity) |
| **WAIC / LOO** | Bayesian model fit (more robust than AIC) |

---

##  Predictive Checks

### For Traditional Model:
-  ROC Curve
-  Calibration Plot
-  Binned Residuals
-  Brier Score

### For Bayesian Model:
-  Posterior Predictive Check (PPC)
-  Observed vs Predicted Scatter
-  Forest Plot of Coefficients
-  Binned Residuals

---

## Requirements

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pymc arviz