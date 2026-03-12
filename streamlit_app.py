import streamlit as st
from src.predict import predict_loan

st.title("🏦 Loan Approval Prediction")

st.subheader("Enter Applicant Details")

age = st.number_input("Age", 18, 70)
annual_income = st.number_input("Annual Income")
credit_score = st.number_input("Credit Score")
loan_amount = st.number_input("Loan Amount Requested")
monthly_expenses = st.number_input("Monthly Expenses")
outstanding_debt = st.number_input("Outstanding Debt")
loan_term = st.number_input("Loan Term (months)")
interest_rate = st.number_input("Interest Rate")

bank_history = st.slider("Bank Account History (years)", 0, 20)
default_risk = st.slider("Default Risk", 0.0, 1.0)
transaction_freq = st.number_input("Transaction Frequency")

city = st.selectbox("City / Town", ["Urban", "Suburban", "Rural"])
loan_type = st.selectbox("Loan Type", ["Personal", "Home", "Business"])
co_applicant = st.selectbox("Co Applicant", ["No", "Yes"])

if st.button("Predict Loan Approval"):

    data = {
        "Age": age,
        "Annual_Income": annual_income,
        "Credit_Score": credit_score,
        "Loan_Amount_Requested": loan_amount,
        "Monthly_Expenses": monthly_expenses,
        "Outstanding_Debt": outstanding_debt,
        "Loan_Term": loan_term,
        "Interest_Rate": interest_rate,
        "Bank_Account_History": bank_history,
        "Default_Risk": default_risk,
        "Transaction_Frequency": transaction_freq,
        "Co-Applicant": 1 if co_applicant == "Yes" else 0,

        "City/Town_Urban": 1 if city == "Urban" else 0,
        "City/Town_Suburban": 1 if city == "Suburban" else 0,

        "Loan_Type_Personal": 1 if loan_type == "Personal" else 0,
        "Loan_Type_Home": 1 if loan_type == "Home" else 0,
        "Loan_Type_Business": 1 if loan_type == "Business" else 0,
    }

    prediction, probability = predict_loan(data)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"Loan Approved ✅ (Probability: {probability:.2f})")
    else:
        st.error(f"Loan Rejected ❌ (Probability: {probability:.2f})")