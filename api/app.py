import streamlit as st
from src.predict import predict_loan

st.set_page_config(page_title="Loan Approval Predictor")

st.title("Loan Approval Prediction App")

st.write("Enter applicant details below")

age = st.number_input("Age", min_value=18, max_value=100)

income = st.number_input("Annual Income", min_value=0)

credit_score = st.number_input("Credit Score", min_value=300, max_value=850)

loan_amount = st.number_input("Loan Amount Requested", min_value=0)

monthly_expenses = st.number_input("Monthly Expenses", min_value=0)

outstanding_debt = st.number_input("Outstanding Debt", min_value=0)

loan_term = st.number_input("Loan Term (months)", min_value=1)

interest_rate = st.number_input("Interest Rate (%)", min_value=0.0)

if st.button("Predict Loan Approval"):

    data = {
        "Age": age,
        "Annual_Income": income,
        "Credit_Score": credit_score,
        "Loan_Amount_Requested": loan_amount,
        "Monthly_Expenses": monthly_expenses,
        "Outstanding_Debt": outstanding_debt,
        "Loan_Term": loan_term,
        "Interest_Rate": interest_rate
    }

    prediction, probability = predict_loan(data)

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    st.write("Approval Probability:", round(probability * 100, 2), "%")