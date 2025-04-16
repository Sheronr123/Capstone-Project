# Capstone-Project
# Loan Default Risk Prediction

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## 📌 Overview

This project develops machine learning models to predict loan default risk using historical loan application data (2007-2020). The solution helps financial institutions make better lending decisions by identifying high-risk loans while minimizing unnecessary rejections.

## 📂 Dataset

The dataset contains approximately 100,000 loan records with 150+ features including:

- **Borrower information**: Credit score, income, employment history
- **Loan characteristics**: Amount, term, interest rate
- **Payment history**
- **Credit utilization metrics**

**Key target variable**: `loan_status` (categorical with multiple statuses)

## 🏗️ Project Structure

### 1. Data Cleaning & Preprocessing

#### Null Value Handling
- Dropped columns with >90% null values
- Used multiple imputation strategies (MICE, BayesianRidge, RandomForest)
- Applied domain-specific logic for structured missingness

#### Outlier Treatment
- Winsorization for moderate skew
- Clipping for extreme values
- Z-score based imputation for near-normal distributions

#### Feature Engineering
- Created time-based features from dates
- Derived financial ratios (DTI-adjusted, utilization changes)
- Added COVID-era risk flags
- Binned credit scores into tiers

### 2. Feature Selection
- **Variance Threshold**: Removed low-variance features
- **Mutual Information**: Selected features with strong predictive power
- **Correlation Analysis**: Eliminated redundant features
- **Domain Knowledge**: Removed data leakage variables

### 3. Modeling Approach

#### Baseline Model: Multinomial Logistic Regression
**Strengths**: Interpretable coefficients, fast training  
**Results**:
- Accuracy: 82.07%
- F1-Macro: 0.56
- Good performance on majority classes but struggled with rare cases

#### Challenger Model: Random Forest Classifier
**Strengths**: Handles non-linear relationships, better with imbalance  
**Results**:
- Accuracy: 92.75% (+10.68% over baseline)
- F1-Macro: 0.65 (+16%)
- Improved detection of high-risk loans (F1=0.81 vs 0.74)

## 🔍 Key Findings

### Strong Predictors
- Interest rates
- FICO scores
- Debt-to-income ratios
- Credit utilization changes

### Critical Insights
- 60-month loans have higher default rates at same interest levels
- COVID-era loans with high DTI are particularly risky
- Very few defaults occur with exceptional credit scores (800+)

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Key libraries:
  - pandas, numpy
  - scikit-learn
  - matplotlib, seaborn
  - scipy, pykalman

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction
![image](https://github.com/user-attachments/assets/06fb8441-6d2f-4cd4-80ae-2288db7e9479)
