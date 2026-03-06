# Credit-Risk-Modeling-with-Regulatory-Compliance-and-Fairness-Analysis-week-8-
# 💳 Credit Risk Modeling with Regulatory Compliance and Fairness Analysis

## 📌 Project Overview

This project develops a production-ready credit risk modeling system to predict loan default probability while ensuring:

- Regulatory compliance
- Model transparency
- Fairness across demographic groups
- Profit-based decision optimization

The system includes data preprocessing, feature engineering, machine learning modeling, fairness auditing, SHAP explainability, and a deployed underwriting dashboard.

---

## 🎯 Business Objective

Financial institutions face default risk when issuing loans.  
This project builds a compliant credit scoring model that:

- Predicts default probability accurately
- Optimizes approval decisions based on expected profit
- Ensures fair lending practices
- Provides explainable adverse action codes

---

## 📊 Dataset

- German Credit Dataset
- 1000 loan applicants
- Target variable: `default` (derived from `credit_risk`)
- Class distribution: ~70% bad loans, ~30% good loans

No artificial target creation was performed (no data leakage).

---

## 🧹 Data Preprocessing

- Missing values handled using:
  - Median (numerical features)
  - Mode (categorical features)
- Categorical encoding via OneHotEncoder
- Stratified 80/20 train-test split
- No leakage between training and testing data

---

## ⚙️ Feature Engineering

Engineered features:

- `amount_duration_ratio`
- `log_amount`
- `duration_squared`
- `young_flag`
- `senior_flag`
- `multiple_credits`
- `long_duration`
- `high_installment`

Strong predictive variables retained:
- status
- credit_history
- savings
- employment_duration
- housing
- demographic indicators

---

## 🤖 Model Development

### Baseline Model
- Logistic Regression
- AUC: **0.79**

### Final Model
- LightGBM Classifier
- AUC: **0.7569**
- Class imbalance handled using `class_weight='balanced'`

The final model meets the requirement of AUC > 0.75.

Model saved as:
```
models/final_credit_model.pkl
```

---

## 📈 Model Validation

- Stratified train-test split (80/20)
- ROC-AUC used as primary metric
- No data leakage
- Reproducible random_state=42

---

## 💰 Profit Optimization

Approval threshold optimized using simulation.

Recommended threshold:
```
0.1
```

This threshold maximizes expected portfolio profit while controlling default exposure.

---

## ⚖️ Fairness & Bias Audit

Protected attribute analyzed:
- Gender (derived from personal_status_sex)

Metrics evaluated:
- Disparate Impact Ratio: 0.88
- Equal Opportunity difference: minimal

Conclusion:
Model operates within acceptable fairness bounds.

---

## 🔍 Model Explainability

SHAP used for interpretability:

- Global feature importance
- Local waterfall explanations
- Adverse action reason mapping

Top negative contributors:
- High loan amount
- Long loan duration
- High installment burden

---

## 🖥️ Dashboard Deployment

Built using Streamlit.

Features:
- Applicant input form
- Real-time risk score
- Approval recommendation
- Integrated feature engineering
- Connected to trained LightGBM model

Run dashboard:

```
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
Fairness_Analysis/
│
├── data/
├── models/
│   └── final_credit_model.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── dashboard/
│   └── app.py
├── reports/
│   ├── 01_Regulatory_Model_Development_Report.pdf
│   ├── 02_Executive_Summary.pdf
│   ├── 03_Fairness_Bias_Analysis_Report.pdf
│   ├── 04_Model_Validation_Performance_Report.pdf
│   └── 05_Dashboard_Deployment_Documentation.pdf
└── README.md
```

---

## 🔐 Regulatory Compliance Considerations

- No target leakage
- Transparent preprocessing
- Documented feature engineering
- Fair lending testing
- SHAP-based explainability
- Profit-based decision framework
- Model monitoring plan defined

---

## 🚀 Future Improvements

- Temporal validation
- Drift detection monitoring
- API-based deployment
- Risk-based pricing extension
- Automated fairness re-evaluation

---
