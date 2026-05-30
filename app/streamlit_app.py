
"""
Credit Default Risk Prediction App
Author: [Your Name]
Project: M.Sc. Economics, IGIDR (2025-2027)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load model (cached for speed)
@st.cache_resource
def load_model():
    model = joblib.load("models/credit_risk_model.pkl")
    with open("models/features.txt", "r") as f:
        features = f.read().strip().split("\n")
    return model, features

model, feature_list = load_model()

# Header
st.title("🏦 Credit Default Risk Predictor")
st.markdown("**Machine Learning-Based Loan Default Prediction with Explainability**")
st.markdown("---")

# Sidebar - About
with st.sidebar:
    st.header("📋 About This App")
    st.markdown("""
    This app predicts the probability of loan default using a 
    machine learning model trained on **1.3 million loans** from 
    LendingClub (2007-2018).
    
    **Model Performance:**
    - AUC-ROC: 0.72
    - KS Statistic: 0.32
    - Algorithm: XGBoost
    
    **Features Used:** 22 borrower attributes
    
    ---
    
    **Created by:** [Your Name]  
    **Institution:** IGIDR  
    **Program:** M.Sc. Economics (2025-2027)
    
    [GitHub Repo](https://github.com/YOUR-USERNAME/credit-risk-modelling)
    """)

# Main content - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Borrower Information")
    
    # Loan details
    st.markdown("**Loan Details**")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=15000, step=1000)
    term = st.selectbox("Loan Term (months)", [36, 60])
    int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.5)
    installment = loan_amnt * (int_rate/100/12) / (1 - (1 + int_rate/100/12)**(-term))
    
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    grade = st.selectbox("Loan Grade", list(grade_map.keys()))
    sub_grade_num = st.slider("Sub-grade (1-5)", min_value=1, max_value=5, value=3)
    
    # Borrower details
    st.markdown("**Borrower Details**")
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
    emp_length = st.slider("Employment Length (years)", min_value=0, max_value=10, value=5)
    dti = st.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0, value=18.0, step=0.5)

with col2:
    st.subheader("📊 Credit History")
    
    fico_avg = st.slider("FICO Score", min_value=600, max_value=850, value=700, step=5)
    open_acc = st.slider("Number of Open Credit Accounts", min_value=1, max_value=50, value=10)
    total_acc = st.slider("Total Credit Accounts", min_value=1, max_value=100, value=25)
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=100000, value=15000, step=1000)
    revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    credit_history_years = st.slider("Credit History (years)", min_value=1, max_value=40, value=15)
    
    st.markdown("**Credit Issues**")
    has_pub_rec = st.checkbox("Has Public Records")
    has_delinq = st.checkbox("Has Delinquencies in Last 2 Years")
    inq_last_6mths = st.slider("Credit Inquiries (last 6 months)", min_value=0, max_value=20, value=1)
    mort_acc = st.slider("Mortgage Accounts", min_value=0, max_value=10, value=1)
    pub_rec_bankruptcies = st.slider("Bankruptcies", min_value=0, max_value=5, value=0)

# Calculate derived features
loan_to_income = loan_amnt / annual_inc
installment_to_income = installment / (annual_inc / 12)
sub_grade_combined = (grade_map[grade] - 1) * 5 + sub_grade_num

# Prediction button
st.markdown("---")
predict_button = st.button("🔮 Predict Default Risk", use_container_width=True, type="primary")

if predict_button:
    # Prepare input data
    input_data = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'grade': grade_map[grade],
        'sub_grade': sub_grade_combined,
        'emp_length': emp_length,
        'annual_inc': annual_inc,
        'dti': dti,
        'fico_avg': fico_avg,
        'open_acc': open_acc,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc,
        'credit_history_years': credit_history_years,
        'loan_to_income': loan_to_income,
        'installment_to_income': installment_to_income,
        'has_pub_rec': int(has_pub_rec),
        'has_delinq': int(has_delinq),
        'mort_acc': mort_acc,
        'inq_last_6mths': inq_last_6mths,
        'pub_rec_bankruptcies': pub_rec_bankruptcies
    }
    
    # Create DataFrame matching training features
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_list]  # Reorder to match training
    
    # Predict
    probability = model.predict_proba(input_df)[0][1]
    
    # Display results
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric("Default Probability", f"{probability:.1%}")
    
    with result_col2:
        if probability < 0.20:
            tier = "🟢 Low Risk"
            decision = "AUTO APPROVE"
            color = "green"
        elif probability < 0.40:
            tier = "🟡 Medium Risk"
            decision = "MANUAL REVIEW"
            color = "orange"
        else:
            tier = "🔴 High Risk"
            decision = "DECLINE"
            color = "red"
        st.metric("Risk Tier", tier)
    
    with result_col3:
        st.metric("Recommendation", decision)
    
    # Progress bar visualization
    st.markdown("**Risk Level Visualization**")
    st.progress(float(probability))
    
    # Business recommendation
    st.markdown("---")
    st.subheader("💼 Business Recommendation")
    
    if probability < 0.20:
        st.success(f"""
        **✅ APPROVE this loan**
        
        - This applicant is in the lowest risk tier
        - Default probability is only {probability:.1%}
        - Suitable for standard interest rates (8-12%)
        - Consider this borrower for premium products
        """)
    elif probability < 0.40:
        st.warning(f"""
        **⚠️ MANUAL REVIEW required**
        
        - This applicant is in the medium risk tier
        - Default probability is {probability:.1%}
        - Suggested interest rate: 14-18%
        - Recommendations:
          - Verify income with additional documentation
          - Consider shorter loan term
          - May require co-signer
        """)
    else:
        st.error(f"""
        **❌ DECLINE this application**
        
        - This applicant is in the high risk tier
        - Default probability is {probability:.1%} (significant)
        - Approving would likely result in loss
        - Alternative: 
          - Suggest secured loan with collateral
          - Recommend credit improvement first
          - Refer to credit counseling
        """)
    
    # SHAP Explanation
    st.markdown("---")
    st.subheader("🔍 Why This Prediction? (SHAP Analysis)")
    
    try:
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # Create SHAP plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=feature_list
            ),
            max_display=10,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **How to read this chart:**
        - **Red bars**: Features pushing toward HIGHER default risk
        - **Blue bars**: Features pushing toward LOWER default risk
        - **Base value**: Average prediction across all borrowers
        - **Final value**: This applicant's predicted probability
        """)
    except Exception as e:
        st.info("SHAP analysis is loading...")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | XGBoost | SHAP | Powered by Machine Learning")
