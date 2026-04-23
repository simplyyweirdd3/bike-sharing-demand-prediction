# Bike Sharing Demand Prediction
**Course:** DATA 5321 - Statistical Machine Learning I  
**Institution:** Seattle University, MS in Data Science  
**Term:** Winter 2026

---

## Overview

This project builds and compares predictive models for daily bike sharing demand using data from the Capital Bikeshare system in Washington, D.C. (2011-2012, 731 days total). The dataset is sourced from the UCI Machine Learning Repository and includes weather conditions, calendar features, and total daily rental counts.

The goal is to predict daily rentals accurately enough to support real operational decisions - staffing, bike rebalancing, and capacity planning. The project is structured across four phases, progressively adding modeling complexity.

---

## Dataset

- **Source:** UCI Machine Learning Repository - Bike Sharing Dataset
- **Response variable:** `cnt` - total daily bike rentals
- **Train/test split:** 2011 data for training (n = 365), 2012 data for testing (n = 366)
- **Predictors:** Season, month, holiday, weekday, working day, weather situation, temperature, apparent temperature, humidity, wind speed

---

## Project Phases

### Phase 1 - Exploratory Analysis & Linear Regression
- Full EDA on 2011 training data: distributions, seasonal patterns, weather effects
- Single-predictor regression models for all variables
- Three interaction models tested: `temp × hum`, `season × weathersit`, `workingday × weekday`
- Backward selection final model retaining: season, month, working day, weather situation, temperature, humidity, and wind speed
- **Training R² ≈ 0.84** - but poor generalization (test R² = −0.70), motivating regularization

### Phase 2 - Regularized Regression & Dimension Reduction
Five models compared, all tuned with 10-fold cross-validation on 2011 training data:

| Model | Key Detail |
|-------|-----------|
| Ridge Regression | L2 penalty; selected λ = 4.35; handles correlated predictors well |
| Lasso Regression | L1 penalty; selected λ = 4.71; zeroed 5 features including `atemp` (collinear with `temp`) |
| PCR | 26 of 28 components selected; no clear scree elbow |
| PLS | 19 components; more efficient than PCR due to response-oriented construction |
| GAM | Nonlinear smooth terms for continuous predictors; captured curved temperature effect |

### Phase 4 - Final Report & Model Comparison
- Full comparison of all five models on 2012 test data
- Evaluation metrics: R², RMSE, MAE
- Best-performing model identified and interpreted
- Discussion of overfitting, year-over-year demand shifts, and practical implications

---

## Key Findings

- **Temperature** is the single strongest predictor of daily demand (R² = 0.595 alone)
- **Apparent temperature (`atemp`)** is nearly perfectly collinear with `temp` - Lasso correctly zeroed it out
- The **season × weather interaction** was statistically significant (p = 0.002): adverse weather reduces demand more sharply in summer than winter
- The Phase 1 backward-selected model overfit badly - test R² of -0.70 was worse than predicting the training mean
- Regularized models significantly improved generalization
- **Shorter models (PLS, Lasso)** achieved comparable performance to PCR with fewer components

---

## Tools & Technologies

- **Language:** R
- **Methods:** Linear regression, Ridge, Lasso, PCR, PLS, GAM
- **Packages:** `glmnet`, `pls`, `mgcv`, `ggplot2`, `dplyr`
- **Validation:** 10-fold cross-validation

---

## Files

| File | Description |
|------|-------------|
| `Project_Phase_1_Report_-_Ruman_Sidhu.docx` | Phase 1: EDA, single-predictor models, interaction tests, backward selection |
| `Project_Phase_2_Report_-_Ruman_Sidhu.docx` | Phase 2: Ridge, Lasso, PCR, PLS, GAM methods and results |
| `Project_Phase_4_Final_Report_-_Ruman_Sidhu.docx` | Final report consolidating all phases with full model comparison |
| `Table1_variable_summary.csv` | Summary statistics for all predictors and response |
| `Table2_single_predictors.csv` | R² and significance for all single-predictor models |
| `Table3_interactions.csv` | Interaction model p-values and R² |
| `Table4_final_model.csv` | Tuning parameters selected for each regularized model |
| `Table5_test_metrics.csv` | Final test set evaluation: R², RMSE, MAE for all five models |

---

## About

Part of the MS in Data Science program at Seattle University. This project covers the full statistical learning pipeline: exploratory analysis, model selection, regularization, dimension reduction, and nonlinear modeling - applied to a real-world operational prediction problem.
