# Credit Risk Prediction App

Interactive Streamlit app for predicting loan default probability.

## Features
- Real-time default probability prediction
- 3-tier risk classification (Low/Medium/High)
- SHAP-based prediction explanations
- Business recommendations for each tier

## Tech Stack
- Frontend: Streamlit
- Model: XGBoost (AUC-ROC: 0.72, KS: 0.32)
- Explainability: SHAP
- Training Data: 1.3M LendingClub loans (2007-2018)

## Live Demo
[Click here to use the app](YOUR_STREAMLIT_URL_WILL_GO_HERE)

## Local Setup
Install dependencies:
    pip install -r requirements.txt

Run the app:
    streamlit run streamlit_app.py

## Model Performance
- AUC-ROC: 0.7214
- KS Statistic: 0.3233
- Captures 65% of defaulters at default threshold

## Author
[Your Name] | M.Sc. Economics, IGIDR (2025-2027)
