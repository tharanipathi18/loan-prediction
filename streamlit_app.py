import streamlit as st
from src.predict import predict_loan
import plotly.graph_objects as go

st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide"
)

# ---------- PROFESSIONAL CSS ----------
st.markdown("""
<style>

body{
background-color:#0f172a;
}

.big-title{
font-size:42px;
font-weight:700;
color:white;
}

.subtitle{
font-size:18px;
color:#9ca3af;
}

.card{
padding:25px;
border-radius:12px;
background-color:#1e293b;
}

.metric{
font-size:28px;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="big-title">🏦 Loan Risk Intelligence System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI powered loan approval prediction</p>', unsafe_allow_html=True)

st.divider()

# ---------- LAYOUT ----------
left, right = st.columns([1,2])

# ---------- INPUT PANEL ----------
with left:

    st.markdown("### Applicant Profile")

    age = st.number_input("Age",18,70)

    annual_income = st.number_input("Annual Income")

    credit_score = st.number_input("Credit Score")

    loan_amount = st.number_input("Loan Amount Requested")

    monthly_expenses = st.number_input("Monthly Expenses")

    outstanding_debt = st.number_input("Outstanding Debt")

    loan_term = st.number_input("Loan Term")

    interest_rate = st.number_input("Interest Rate")

    bank_history = st.slider("Bank Account History",0,20)

    default_risk = st.slider("Default Risk",0.0,1.0)

    transaction_freq = st.number_input("Transaction Frequency")

    city = st.selectbox("City Type",["Urban","Suburban","Rural"])

    loan_type = st.selectbox("Loan Type",["Personal","Home","Business"])

    co_applicant = st.selectbox("Co Applicant",["No","Yes"])

    predict_button = st.button("🚀 Run Prediction")


# ---------- DASHBOARD PANEL ----------
with right:

    st.markdown("### Risk Assessment Dashboard")

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

        col1,col2 = st.columns(2)

        with col1:
            st.metric("Approval Probability",f"{probability*100:.1f}%")

        with col2:
            if prediction == 1:
                st.success("Approved")
            else:
                st.error("Rejected")

        # ---------- GAUGE CHART ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text': "Loan Approval Probability"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "#22c55e"},
                'steps': [
                    {'range': [0,40], 'color': "#7f1d1d"},
                    {'range': [40,70], 'color': "#78350f"},
                    {'range': [70,100], 'color': "#064e3b"}
                ]
            }
        ))

        st.plotly_chart(fig,use_container_width=True)