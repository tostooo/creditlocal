# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Global Setup
st.set_page_config(
    page_title="Credit Score Profiler",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Real-Time Credit Score Classifier")
st.markdown("Provide the core customer financial credentials below to analyze risk tier assignments.")
st.write("---")

# 2. Secure Model Path Context Map
MODEL_PATH = "artifacts/best_credit_model.pkl"

@st.cache_resource
def load_cached_pipeline():
    """Loads the model artifact into memory once and caches it to keep things snappy."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Missing artifact: '{MODEL_PATH}'. Run 'python pipeline.py' first in your terminal!")
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_cached_pipeline()

if pipeline is not None:
    st.subheader("📋 Enter Customer Financial Attributes")

    # --- INPUT LAYER FOR ALL FEATURES EXPECTED BY TRAINED PIPELINE ---
    age = st.number_input("Age", min_value=18, max_value=110, value=35)
    
    occupation = st.selectbox("Occupation", [
        "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", 
        "Lawyer", "Media_Manager", "Doctor", "Accountant", "Musician"
    ])
    
    month = st.selectbox("Current Month", [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ])
    
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=3)
    num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, max_value=20, value=4)
    annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
    monthly_inhand_salary = st.number_input("Monthly In-hand Salary ($)", min_value=0.0, value=4200.0, step=100.0)
    total_emi_per_month = st.number_input("Total EMI Paid Monthly ($)", min_value=0.0, value=150.0, step=10.0)
    amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=200.0, step=10.0)
    monthly_balance = st.number_input("End-of-Month Balance ($)", value=500.0, step=50.0)
    interest_rate = st.number_input("Average Interest Rate (%)", min_value=0, max_value=50, value=12)
    num_of_loans = st.number_input("Number of Active Loans", min_value=0, max_value=20, value=2)
    delay_from_due_date = st.number_input("Average Delay from Due Date (Days)", min_value=0, value=5)
    num_of_delayed_payments = st.number_input("Number of Delayed Payments", min_value=0, value=2)
    changed_credit_limit = st.number_input("Changed Credit Limit Alteration", value=10.5, step=0.5)
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=1)
    outstanding_debt = st.number_input("Total Outstanding Debt ($)", min_value=0.0, value=1200.0, step=100.0)
    credit_utilization_ratio = st.slider("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
    credit_mix = st.selectbox("Credit Mix Profile", ["Good", "Standard", "Bad"])
    payment_of_min_amount = st.selectbox("Payment of Minimum Amount Only?", ["Yes", "No", "NM"])
    
    payment_behaviour = st.selectbox("Payment Behaviour Profile", [
        "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments",
        "High_spent_Small_value_payments", "High_spent_Medium_value_payments", "High_spent_Large_value_payments"
    ])
    
    # Structural column filler required so that the model gets the exact shape it was trained o

    st.write("---")
    
    # 3. Execution Trigger
    if st.button("🎯 Run Diagnostics", type="primary", use_container_width=True):
        
        # Package data inside a dictionary with matching column headers 
        input_dict = {
            'Month': [month], 'Age': [age], 'Occupation': [occupation], 'Annual_Income': [annual_income],
            'Monthly_Inhand_Salary': [monthly_inhand_salary], 'Num_Bank_Accounts': [num_bank_accounts],
            'Num_Credit_Card': [num_credit_cards], 'Interest_Rate': [interest_rate], 'Num_of_Loan': [num_of_loans],
            'Delay_from_due_date': [delay_from_due_date],
            'Num_of_Delayed_Payment': [num_of_delayed_payments], 'Changed_Credit_Limit': [changed_credit_limit],
            'Num_Credit_Inquiries': [num_credit_inquiries], 'Credit_Mix': [credit_mix], 'Outstanding_Debt': [outstanding_debt],
            'Credit_Utilization_Ratio': [credit_utilization_ratio],
            'Payment_of_Min_Amount': [payment_of_min_amount], 'Total_EMI_per_month': [total_emi_per_month],
            'Amount_invested_monthly': [amount_invested_monthly], 'Payment_Behaviour': [payment_behaviour],
            'Monthly_Balance': [monthly_balance]
        }
        
        # Format map directly to Pandas DataFrame
        features_df = pd.DataFrame(input_dict)
        
        with st.spinner("Processing features through production pipeline..."):
            prediction = pipeline.predict(features_df)[0]
            probabilities = pipeline.predict_proba(features_df)[0]
            classes = pipeline.classes_
            
        # 4. Present Output Dynamic Styling
        st.subheader("📊 Model Classification Output")
        
        if prediction == "Good":
            st.success(f"### Predicted Class: **{prediction} Credit Tier** 🎉")
        elif prediction == "Standard":
            st.info(f"### Predicted Class: **{prediction} Credit Tier** ⚖️")
        else:
            st.error(f"### Predicted Class: **{prediction} Credit Tier** ⚠️")
            
        # Output probability spread table
        st.write("#### Confidence Metrics Vector:")
        prob_df = pd.DataFrame([probabilities], columns=classes)
        st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)