import math

import streamlit as st

from src.predict import predict_loan


def _build_payload(
    age: float,
    income: float,
    credit_score: float,
    loan_amount: float,
    monthly_expenses: float,
    outstanding_debt: float,
    loan_term: float,
    interest_rate: float,
) -> dict[str, float]:
    return {
        "Age": age,
        "Annual_Income": income,
        "Credit_Score": credit_score,
        "Loan_Amount_Requested": loan_amount,
        "Monthly_Expenses": monthly_expenses,
        "Outstanding_Debt": outstanding_debt,
        "Loan_Term": loan_term,
        "Interest_Rate": interest_rate,
    }


def _safe_probability(value: object) -> float | None:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(prob) or math.isinf(prob):
        return None
    return max(0.0, min(1.0, prob))


def _is_approved(prediction: object) -> bool:
    try:
        return int(float(prediction)) == 1
    except (TypeError, ValueError):
        return str(prediction).strip().lower() in {"approved", "true", "yes"}


st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide",
)

st.markdown(
    """
    <h1 style='text-align: center;'>Loan Approval Prediction</h1>
    <p style='text-align: center;'>AI powered credit risk evaluation</p>
    """,
    unsafe_allow_html=True,
)
st.divider()

st.sidebar.header("Applicant Information")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Annual Income", min_value=1000, value=50000)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=650)
loan_amount = st.sidebar.number_input("Loan Amount Requested", min_value=1000, value=15000)
monthly_expenses = st.sidebar.number_input("Monthly Expenses", min_value=0, value=2000)
outstanding_debt = st.sidebar.number_input("Outstanding Debt", min_value=0, value=5000)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=1, value=24)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, value=7.0)
predict_btn = st.sidebar.button("Predict Loan Approval")

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

if predict_btn:
    payload = _build_payload(
        age=age,
        income=income,
        credit_score=credit_score,
        loan_amount=loan_amount,
        monthly_expenses=monthly_expenses,
        outstanding_debt=outstanding_debt,
        loan_term=loan_term,
        interest_rate=interest_rate,
    )

    try:
        prediction, probability = predict_loan(payload)
    except Exception as exc:
        st.error("Prediction failed. Verify model files and configuration.")
        st.caption(str(exc))
        st.stop()

    st.subheader("Prediction Result")

    if _is_approved(prediction):
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    safe_probability = _safe_probability(probability)
    if safe_probability is None:
        st.info("Approval probability unavailable.")
    else:
        st.metric(label="Approval Probability", value=f"{safe_probability * 100:.2f}%")
        st.progress(safe_probability)

st.divider()
st.markdown(
    """
    <p style='text-align:center;'>Machine Learning Loan Risk Model</p>
    """,
    unsafe_allow_html=True,
)
