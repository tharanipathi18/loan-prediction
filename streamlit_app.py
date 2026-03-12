import streamlit as st
from src.predict import predict_loan

st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.main-title {
    font-size:40px;
    font-weight:700;
}

.card {
    padding:25px;
    border-radius:12px;
    background-color:#111827;
}

.metric-box {
    padding:20px;
    border-radius:10px;
    background-color:#1f2937;
}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown('<p class="main-title">🏦 Loan Approval AI</p>', unsafe_allow_html=True)
st.caption("Machine Learning based Loan Risk Prediction System")


# ---------- LAYOUT ----------
col1, col2 = st.columns([1,2])


# ---------- INPUT PANEL ----------
with col1:

    st.subheader("Applicant Information")

    age = st.number_input("Age", 18, 70)

    annual_income = st.number_input("Annual Income")

    credit_score = st.number_input("Credit Score")

    loan_amount = st.number_input("Loan Amount Requested")

    monthly_expenses = st.number_input("Monthly Expenses")

    outstanding_debt = st.number_input("Outstanding Debt")

    loan_term = st.number_input("Loan Term")

    interest_rate = st.number_input("Interest Rate")

    bank_history = st.slider("Bank Account History (Years)",0,20)

    default_risk = st.slider("Default Risk Score",0.0,1.0)

    transaction_freq = st.number_input("Transaction Frequency")

    city = st.selectbox("City Type",["Urban","Suburban","Rural"])

    loan_type = st.selectbox("Loan Type",["Personal","Home","Business"])

    co_applicant = st.selectbox("Co Applicant",["No","Yes"])

    predict_button = st.button("🚀 Predict Approval")


# ---------- RESULT PANEL ----------
with col2:

    st.subheader("Prediction Dashboard")

    if predict_button:

        data = {
            "Age":age,
            "Annual_Income":annual_income,
            "Credit_Score":credit_score,
            "Loan_Amount_Requested":loan_amount,
            "Monthly_Expenses":monthly_expenses,
            "Outstanding_Debt":outstanding_debt,
            "Loan_Term":loan_term,
            "Interest_Rate":interest_rate,
            "Bank_Account_History":bank_history,
            "Default_Risk":default_risk,
            "Transaction_Frequency":transaction_freq,
            "Co-Applicant":1 if co_applicant=="Yes" else 0,
            "City/Town_Urban":1 if city=="Urban" else 0,
            "City/Town_Suburban":1 if city=="Suburban" else 0,
            "Loan_Type_Personal":1 if loan_type=="Personal" else 0,
            "Loan_Type_Home":1 if loan_type=="Home" else 0,
            "Loan_Type_Business":1 if loan_type=="Business" else 0
        }

        prediction, probability = predict_loan(data)

        p1,p2 = st.columns(2)

        with p1:
            st.metric("Approval Probability",f"{probability*100:.2f}%")

        with p2:
            if prediction == 1:
                st.success("Loan Approved")
            else:
                st.error("Loan Rejected")

        st.progress(probability)