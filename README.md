# Customer Churn Analysis – Data Preparation & EDA

## 📌 Introduction

This project performs the initial steps required to create a predictive **customer churn machine learning model**.The goals of this initial phase are to:

- Combine multiple data sources relevant to customer behavior.
- Perform **exploratory data analysis (EDA)** to uncover patterns associated with churn.
- Clean and preprocess the dataset for modeling.
- Engineer new features to improve model accuracy.

---

## 📂 Data Sources & Rationale

The analysis uses multiple datasets, each providing different behavioral and demographic insights:

| Dataset Name                    | Description                                                                                                      |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Customer_Demographics** | Contains personal attributes for each customer.                                                                  |
| **Transaction_History**   | Dates and amounts of transactions capturing**spending behavior**, **recency**, and **tenure**. |
| **Customer_Service**      | Records of customer support interactions (dates & types), which may signal dissatisfaction or high engagement.   |
| **Online_Activity**       | Login frequency, last login date, and service usage, useful for identifying retention signals.                   |
| **Churn_Status**          | Binary label (0 = active, 1 = churned).                                                                          |

> Together, these datasets provide a **holistic view of customer behavior** across purchasing, support interactions, and online activity.

---

## 🔄 Data Processing Summary

1. **Data Merging**

# Customer Churn Analysis — Lloyds Banking Group (EDA, Prep & Modelling)

## Overview

This repository contains a complete end-to-end preparation and exploratory analysis of customer data aimed at building a churn prediction model for Lloyds Banking Group. The work merges demographics, transactions, service interactions and online activity; engineers features; runs EDA; and evaluates classification models to identify customers at risk of churn.

Goals:

- Produce a cleaned, feature-engineered dataset ready for modelling.
- Identify behavioral and demographic signals correlated with churn.
- Train and evaluate classification models and surface actionable recommendations for retention.

---

## Data sources

The analysis combines these inputs:

- `Customer_Demographics` — personal attributes.
- `Transaction_History` — transaction dates & amounts (used to compute recency, tenure, lifetime spend).
- `Customer_Service` — support interaction records (counts & types).
- `Online_Activity` — login frequency and recent online engagement.
- `Churn_Status` — binary target (0 = active, 1 = churned).

These sources provide complementary behavioral and demographic signals used throughout the analysis and modelling pipeline.

---

## Processing & feature engineering

Key steps applied to the raw data:

- Merged datasets on `CustomerID` (left join onto demographics) and aggregated logs (first/last dates, counts, totals).
- Converted date columns to `datetime` and used the dataset's max transaction date as the recency reference.
- Created features: `FirstTransactionDate`, `LastTransactionDate`, `TotalAmountSpent`, `AvgAmountPerTransaction`, `TransactionCount`, `TotalServiceInteractions`, `ServiceDiversity`, `RecencyDays`, `TenureDays`, `DaysSinceLastLogin`.
- Missing values: counts → `0`; date proxies used where appropriate; numerics → median imputation.
- Outliers: capped (`winsorized`) `TotalAmountSpent` and `TransactionCount` at the 99th percentile.
- Encoding & scaling: categorical variables one-hot encoded; numeric features standardized with `StandardScaler`. The scaler is saved as `numeric_scaler.save` for consistent preprocessing.

Produced dataset: `customer_churn_analysis_cleaned.csv` (ready for modelling).

---

## Exploratory Data Analysis — key insights

Visuals are available in `phaseOne/phase1_charts/` and embedded below.

- Churn distribution: balanced enough to compare modelling approaches.
  ![Churn Distribution](phaseOne/phase1_charts/churn_status_distribution.png)
- Age profile: most customers are aged 25–44; ages 35–54 demonstrate higher average spend (priority targets).
  ![Age Distribution](phaseOne/phase1_charts/customer_age_distribution.png)
- Spending by age: 35–54 shows highest average spend — useful for targeted offers.
  ![Spending Average by Age](phaseOne/phase1_charts/totalaverage_amount_spent_agerange.png)
- Service-channel usage: certain channels dominate; consider reinforcing preferred channels and improving underused ones.
  ![Client Interaction by Type](phaseOne/phase1_charts/customer_interactions_by_type.png)
- Recency vs spending: high spenders generally have low `RecencyDays`; high recency + low spend indicates churn risk.
  ![Recency vs Spending](phaseOne/phase1_charts/recency_total_amount_spent.png)
- Spending by churn status: non-churn customers show higher median spend and transaction counts.
  ![Total Spending Boxplot](phaseOne/phase1_charts/total_amount_spent_boxplot.png)
- Correlations: `TotalAmountSpent` and `TransactionCount` are strongly positively correlated; `RecencyDays` negatively correlates with spending/transactions.
  ![Correlation Heatmap](phaseOne/phase1_charts/correlation_heatma.png)

EDA conclusion: recency, total spend and transaction frequency are among the strongest early-warning signals for churn. Demographic segments (age/income) help prioritise interventions.

---

## Modelling approach & results

Class imbalance handling:

- Applied SMOTE (oversampling minority) then random undersampling of the majority class (imbalanced-learn) to create a more balanced training set. Splits were stratified to preserve distribution.

Algorithms evaluated:

- Random Forest (class weights balanced)
- Support Vector Machine (SVM) with probability estimates
- XGBoost (tuned with scale_pos_weight and depth)

Key metrics for the churn (positive) class:

| Model         | Precision | Recall | F1-score | ROC-AUC | Notes                                                                                        |
| ------------- | --------: | -----: | -------: | ------: | -------------------------------------------------------------------------------------------- |
| Random Forest |      0.19 |   0.68 |     0.30 |    0.35 | High recall, many false positives — useful for catch-all screening.                         |
| SVM           |      0.22 |   0.56 |     0.32 |    0.46 | Best-balanced performance and highest ROC-AUC in this evaluation — chosen as primary model. |
| XGBoost       |      0.16 |   0.37 |     0.23 |    0.44 | Underperformed on recall; may benefit from further tuning and feature work.                  |

Evaluation summary: Random Forest maximises recall at the expense of precision; SVM offers the best operational balance and was selected as the recommended primary model for deployment.

---

## Recommendations & business actions

- Deploy SVM in a production scoring pipeline to flag at-risk customers for retention outreach.
- Use SHAP explainability on the chosen model to identify top drivers and create targeted interventions (e.g., offers for low-spend, high-recency customers).
- Consider a two-stage operational flow: (1) high-recall model or rule set to shortlist candidates; (2) higher-precision model or rules to prioritise outreach to the best targets.
- Monitor precision/recall over time and retrain frequently with new data; track lift on retention campaigns to validate model ROI.

---

## Files produced

- `customer_churn_analysis_cleaned.csv` — cleaned, feature-engineered dataset for modelling.
- `numeric_scaler.save` — `StandardScaler` object for consistent preprocessing.
- Visualizations: `phaseOne/phase1_charts/` (several PNGs used above).

---

## Quick start

1. Open the notebooks in `phaseOne/` for EDA and preprocessing steps (notebooks contain the exact code used to generate the cleaned dataset and charts).
2. To retrain models locally, ensure dependencies (pandas, scikit-learn, xgboost, imbalanced-learn) are installed and run the model notebook in `phaseTwo/`.

Example (PowerShell):

```powershell
# create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # create if you want to capture deps
```

---
