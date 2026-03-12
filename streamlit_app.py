import streamlit as st
from src.predict import predict_loan

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>💰 Loan Approval Prediction</h1>
    <p style='text-align: center;'>AI powered credit risk evaluation</p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------- SIDEBAR INPUT ----------
st.sidebar.header("Applicant Information")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)

income = st.sidebar.number_input("Annual Income", min_value=1000, value=50000)

credit_score = st.sidebar.number_input(
    "Credit Score", min_value=300, max_value=850, value=650
)

loan_amount = st.sidebar.number_input(
    "Loan Amount Requested", min_value=1000, value=15000
)

monthly_expenses = st.sidebar.number_input(
    "Monthly Expenses", min_value=0, value=2000
)

outstanding_debt = st.sidebar.number_input(
    "Outstanding Debt", min_value=0, value=5000
)

loan_term = st.sidebar.number_input(
    "Loan Term (months)", min_value=1, value=24
)

interest_rate = st.sidebar.number_input(
    "Interest Rate (%)", min_value=0.0, value=7.0
)

predict_btn = st.sidebar.button("Predict Loan Approval")

# ---------- MAIN DASHBOARD ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Summary")

    st.write("Age:", age)
    st.write("Annual Income:", income)
    st.write("Credit Score:", credit_score)
    st.write("Loan Amount:", loan_amount)

with col2:
    st.subheader("Financial Information")

    st.write("Monthly Expenses:", monthly_expenses)
    st.write("Outstanding Debt:", outstanding_debt)
    st.write("Loan Term:", loan_term)
    st.write("Interest Rate:", interest_rate)

st.divider()

# ---------- PREDICTION ----------
if predict_btn:

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

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric(
        label="Approval Probability",
        value=f"{round(probability*100,2)}%"
    )

    st.progress(float(probability))

st.divider()

# ---------- FOOTER ----------
st.markdown(
    """
    <p style='text-align:center;'>Machine Learning Loan Risk Model</p>
    """,
    unsafe_allow_html=True
)