# Credit Default Risk Modelling

## Overview
An end-to-end machine learning project to predict the probability 
of loan default using LendingClub loan data. The project compares 
multiple classification models and uses SHAP-based explainability 
to support transparent credit decisions.

## Business Problem
Banks and financial institutions need to assess the credit risk of 
loan applicants before approving loans. This project builds a credit 
scoring pipeline that:
- Predicts probability of default for each borrower
- Compares multiple ML models using industry-standard metrics
- Provides borrower-level explanations using SHAP values
- Supports business threshold decisions for loan approval

## Dataset
- Source: LendingClub Loan Data (Kaggle)
- Size: ~890,000 loans issued between 2007 and 2018
- Target Variable: loan_status (Default = 1, Fully Paid = 0)

## Project Structure
credit-risk-modelling/
├── notebooks/          # Step-by-step Jupyter notebooks
├── src/                # Python scripts
├── data/               # Raw and processed data (not tracked by git)
├── models/             # Saved model files (not tracked by git)
├── reports/            # Evaluation results and figures
└── app/                # Streamlit deployment

## Methodology
1. Data Understanding and Target Variable Definition
2. Exploratory Data Analysis
3. Data Leakage Prevention
4. Feature Engineering
5. Time-Based Train-Test Split
6. Baseline Model: Logistic Regression
7. Advanced Models: Random Forest, XGBoost, LightGBM
8. Model Evaluation: AUC-ROC, KS Statistic, Precision-Recall
9. Explainability: SHAP Values
10. Business Threshold Optimization

## Results
To be updated as project progresses

| Model | AUC-ROC | KS Statistic | Recall |
|---|---|---|---|
| Logistic Regression | - | - | - |
| Random Forest | - | - | - |
| XGBoost | - | - | - |
| LightGBM | - | - | - |

## How to Run

### 1. Clone the repository
git clone https://github.com/anuragjaine/credit-risk-modelling.git
cd credit-risk-modelling

### 2. Create and activate virtual environment
python -m venv credit_risk_env
credit_risk_env\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download Dataset
Download LendingClub dataset from:
https://www.kaggle.com/datasets/wordsforthewise/lending-club
Place the CSV file in data/raw/ folder

### 5. Run notebooks in order
Start with notebooks/01_data_understanding.ipynb

## Author
Anurag Jain
M.Sc. Economics, IGIDR (2025-2027)

## License
MIT License
