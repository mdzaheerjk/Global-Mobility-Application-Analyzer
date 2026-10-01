# Global Mobility Application Analyzer — Project Draft

## Project Title

Global Mobility Application Analyzer using Machine Learning

## Overview

Predict whether a visa application will be approved or denied using historical immigration application data. The system will help immigration consultants, employers, and visa applicants assess approval likelihood, identify key risk factors, and automate triage of cases.

---

## Objectives

* Build and evaluate ML models to predict visa approval (binary classification).
* Interpret drivers of approval decisions and surface actionable explanations.
* Produce a reproducible pipeline: data ingestion, preprocessing, modeling, evaluation, and deployment.
* Provide a web dashboard for probability scores, feature explanations (SHAP/LIME), and batch scoring.

---

## Success Criteria

* A model with ROC-AUC ≥ 0.80 on held-out test data (baseline: 0.70).
* Precision/recall tradeoff tuned to stakeholder needs (e.g., minimize false negatives for high-risk cases).
* Clear, auditable feature importance and sample-level explanations.
* End-to-end pipeline and a basic web UI for inference.

---

## Data Sources

The data contains the different attributes of the employee and the employer. The detailed data dictionary is given below.

👉🏻Data Link: https://www.kaggle.com/datasets/moro23/easyvisa-dataset

---

## Target Variable

* `approval_status` (binary): `1 = Approved`, `0 = Denied`.
* Optionally, multi-class (Approved, Denied, Pending, Withdrawn) or regression on `time_to_decision`.

---

## Features (Project-Specific)

Your dataset includes the following features:

* **Continent:** Asia, Africa, North America, Europe, South America, Oceania
* **Education:** High School, Bachelor’s, Master’s Degree, Doctorate
* **Job Experience:** Yes, No
* **Required Training:** Yes, No
* **Number of Employees:** 15,000 to 40,000
* **Region of Employment:** West, Northeast, South, Midwest, Island
* **Prevailing Wage:** 700 to 70,000
* **Contract Tenure:** Hour, Week, Month, Year
* **Full Time:** Yes, No
* **Age of Company:** 15 to 180

These features will be cleaned, encoded, and incorporated directly into the ML pipeline.

---

## Data Preprocessing

1. Data cleaning: remove duplicates, fix inconsistent encodings, handle corrupt records.
2. Missing values: analyze missingness pattern. Use imputation strategies (simple impute, KNN, or model-based) and flags for missingness.
3. Categorical encoding: target/ordinal encoding for high-cardinality features, one-hot for low-cardinality.
4. Date/time features: extract day-of-week, month, seasonality, time-since-last-application.
5. Text features: process free-text reasons or notes with NLP (TF-IDF or embeddings). Consider extracting indicator flags from notes (e.g., "incomplete medical").
6. Balance classes: if approval/denial is imbalanced, consider resampling (SMOTE, ADASYN) or class weighting.
7. Feature selection: mutual information, recursive feature elimination, L1 regularization.

---

## Modeling Approaches

* **Baseline models:** Logistic Regression, Decision Tree.
* **Tree-based ensembles:** Random Forest, XGBoost, LightGBM, CatBoost.
* **Linear models with regularization:** Logistic Regression with L1/L2.
* **Neural networks:** simple MLP or text+tabular fusion model if textual notes are important.
* **Probabilistic models:** Calibrated classifiers (Platt scaling / isotonic) for well-calibrated probabilities.

Choose best model based on validation metrics and explainability needs.

---

## Evaluation Metrics

* Primary: ROC-AUC, PR-AUC (if classes imbalanced).
* Secondary: Accuracy, Precision, Recall, F1-score, Brier score (probability calibration), confusion matrix.
* Business-oriented: False Negative Rate (denied predicted as approved) and False Positive Rate depending on stakeholder costs.
* Explainability: SHAP value stability across folds.

---

## Validation Strategy

* Time-aware split (train on earlier submissions, validate/test on more recent ones) to avoid leakage.
* If no strong temporal ordering, use stratified K-fold (with grouping by applicant_id if multiple records per applicant).
* Nested cross-validation for hyperparameter tuning if dataset size permits.

---

## Explainability & Fairness

* Use SHAP or LIME for local and global explanations.
* Evaluate fairness: assess model behavior across protected groups (gender, nationality) and highlight disparate impact.
* Add guardrails: if a protected attribute strongly drives decisions, present human-in-the-loop review.

---

## Deployment Plan

1. Model packaging: save model and preprocessing pipeline 
2. Serving: simple FastAPI endpoint for single/batch predictions.
3. Dashboard: Streamlit or React app showing probability, top contributing features, and batch upload.
4. Monitoring: track data drift, model performance, and prediction distributions. Set alerts for performance degradation.

---



## Risks & Mitigations

* **Data quality issues:** implement robust validation checks and work with SME to correct.
* **Concept drift from policy changes:** include versioning and retraining schedule; add policy-change flags as features.
* **Privacy/compliance:** apply de-identification, access controls, and legal review.
* **Bias and fairness:** detect and mitigate; involve legal/ethics team for review.


---

## Ethics & Legal Considerations

* Be transparent about the model’s limitations and intended use (decision-support, not automated denial without human review).
* Maintain logs for auditability: inputs, outputs, and explanations.
* Ensure compliance with immigration, employment, and data protection laws.

---
