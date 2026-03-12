import streamlit as st
from src.predict import predict_loan
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Risk AI", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>

body {
background-color: #0f172a;
}

.title {
font-size:42px;
font-weight:700;
color:white;
}

.subtitle {
color:#94a3b8;
}

.card {
background: rgba(255,255,255,0.05);
backdrop-filter: blur(12px);
padding:25px;
border-radius:16px;
border:1px solid rgba(255,255,255,0.1);
}

.metric {
font-size:28px;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="title">🏦 Loan Risk Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered credit decision system</p>', unsafe_allow_html=True)

st.divider()

# ---------- LAYOUT ----------
left, center, right = st.columns([1,2,1])

# ---------- APPLICANT PROFILE ----------
with left:

    st.markdown("### Applicant Profile")

    age = st.number_input("Age",18,70)
    annual_income = st.number_input("Annual Income")
    credit_score = st.number_input("Credit Score")
    loan_amount = st.number_input("Loan Amount")
    monthly_expenses = st.number_input("Monthly Expenses")
    outstanding_debt = st.number_input("Outstanding Debt")

    loan_term = st.number_input("Loan Term")
    interest_rate = st.number_input("Interest Rate")

    bank_history = st.slider("Bank History (Years)",0,20)
    default_risk = st.slider("Default Risk Score",0.0,1.0)
    transaction_freq = st.number_input("Transaction Frequency")

    city = st.selectbox("City Type",["Urban","Suburban","Rural"])
    loan_type = st.selectbox("Loan Type",["Personal","Home","Business"])
    co_applicant = st.selectbox("Co-Applicant",["No","Yes"])

    run = st.button("Run Risk Assessment")

# ---------- MAIN DASHBOARD ----------
with center:

    st.markdown("### Credit Decision Dashboard")

    if run:

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
                st.success("Loan Approved")
            else:
                st.error("Loan Rejected")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color':"#22c55e"},
                'steps':[
                    {'range':[0,40],'color':"#7f1d1d"},
                    {'range':[40,70],'color':"#78350f"},
                    {'range':[70,100],'color':"#064e3b"}
                ]
            }
        ))

        st.plotly_chart(fig,use_container_width=True)

# ---------- SIDE PANEL ----------
with right:

    st.markdown("### System Info")

    st.info("""
Model: Random Forest  
Accuracy: ~85%  

Signals used:

• Credit score  
• Debt ratio  
• Income stability  
• Transaction behavior  
• Default risk
""")