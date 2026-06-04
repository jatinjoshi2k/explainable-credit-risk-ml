import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="European Credit Risk AI Estimator", layout="wide")

st.title("European Credit Risk AI Estimator")
st.markdown(
    "Enter the applicant's details in the sidebar. The AI model will predict "
    "the likelihood of loan default and explain the key drivers behind its decision."
)

# ── Load model & column schema ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = XGBClassifier()
    m.load_model("loan_model.json")
    return m

@st.cache_data
def load_columns():
    df = pd.read_csv("credit_risk_cleaned.csv", nrows=0)
    return [c for c in df.columns if c != "loan_status"]

model = load_model()
feature_cols = load_columns()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Applicant Details")

age = st.sidebar.slider(
    "Age (years)", 18, 80, 30,
    help="Younger applicants (under 25) show statistically higher default rates in this dataset. "
         "Age proxies for financial maturity and career stability."
)
income = st.sidebar.slider(
    "Annual Income ($)", 5_000, 600_000, 55_000, step=1_000,
    help="The single strongest predictor in the model. Higher income sharply reduces default risk "
         "by giving the borrower more capacity to absorb repayment. The model is especially "
         "sensitive below $30,000/year."
)
emp_length = st.sidebar.slider(
    "Employment Length (years)", 0, 40, 5,
    help="Longer continuous employment signals income stability. Borrowers with under 1 year of "
         "employment carry noticeably higher risk. The marginal benefit flattens out beyond ~8 years."
)
loan_amnt = st.sidebar.slider(
    "Loan Amount ($)", 500, 35_000, 10_000, step=500,
    help="Larger loans are riskier in absolute terms, but the model weighs this mainly through the "
         "Loan-to-Income Ratio (loan amount ÷ income). A $30,000 loan for a $200,000 earner is "
         "far less risky than the same loan for a $40,000 earner."
)
int_rate = st.sidebar.slider(
    "Interest Rate (%)", 5.0, 25.0, 11.0, step=0.1,
    help="Higher interest rates increase monthly repayment burden and correlate strongly with default. "
         "Rates above 15% are a significant risk flag — they reflect both higher borrower risk and "
         "create a heavier repayment load that can tip borderline borrowers into default."
)
cred_hist = st.sidebar.slider(
    "Credit History Length (years)", 0, 30, 4,
    help="A longer credit history gives the model more signal about repayment behaviour. "
         "Very short histories (under 2 years) introduce uncertainty. Beyond ~6 years the "
         "impact on the prediction diminishes."
)

home_ownership = st.sidebar.selectbox(
    "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"],
    help="Renters carry the highest default risk in this dataset — likely because renting "
         "indicates less financial asset accumulation. MORTGAGE holders perform similarly to "
         "owners; both show lower default rates than renters."
)
loan_intent = st.sidebar.selectbox(
    "Loan Intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    help="VENTURE loans carry the highest default risk — business investments are volatile. "
         "MEDICAL and PERSONAL loans are mid-risk. EDUCATION loans tend to be lower risk "
         "as they are tied to income-building activity. DEBTCONSOLIDATION varies widely "
         "depending on the borrower's overall financial picture."
)
prior_default = st.sidebar.selectbox(
    "Prior Default on File?", ["N", "Y"],
    help="One of the strongest binary signals in the model. A prior default on the credit bureau "
         "record (Y) significantly increases predicted default probability — past behaviour is "
         "the most reliable predictor of future behaviour in credit risk modelling."
)

# ── Parameter guide expander ──────────────────────────────────────────────────
with st.sidebar.expander("Parameter Impact Guide"):
    st.markdown("""
| Parameter | Direction | Strength |
|---|---|---|
| Annual Income | Higher = safer | Very High |
| Interest Rate | Higher = riskier | High |
| Loan-to-Income Ratio | Higher = riskier | High |
| Prior Default | Y = riskier | High |
| Home Ownership | OWN/MORTGAGE < RENT | Medium |
| Employment Length | Longer = safer | Medium |
| Loan Intent | VENTURE riskiest | Medium |
| Credit History | Longer = safer | Low–Medium |
| Age | Younger = riskier | Low |

*Strengths derived from SHAP feature importance analysis on the trained model.*
""")

# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("---")
predict_clicked = st.button("Predict Default Risk", type="primary", use_container_width=True)

if predict_clicked:

    # 1. Derive computed feature
    loan_percent_income = round(loan_amnt / income, 4) if income > 0 else 0.0

    # 2. Build raw input dict (numerical)
    raw = {
        "person_age":                 age,
        "person_income":              income,
        "person_emp_length":          float(emp_length),
        "loan_amnt":                  loan_amnt,
        "loan_int_rate":              int_rate,
        "loan_percent_income":        loan_percent_income,
        "cb_person_cred_hist_length": cred_hist,
    }

    # 3. One-hot encode categoricals manually
    for val in ["MORTGAGE", "OTHER", "OWN", "RENT"]:
        raw[f"person_home_ownership_{val}"] = 1 if home_ownership == val else 0

    for val in ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]:
        raw[f"loan_intent_{val}"] = 1 if loan_intent == val else 0

    # loan_grade is lender-assigned — not known at application time, left as all zeros
    for val in ["A", "B", "C", "D", "E", "F", "G"]:
        raw[f"loan_grade_{val}"] = 0

    for val in ["N", "Y"]:
        raw[f"cb_person_default_on_file_{val}"] = 1 if prior_default == val else 0

    # 4. Align to exact training column order (fill any gaps with 0)
    input_df = pd.DataFrame([raw]).reindex(columns=feature_cols, fill_value=0)

    # 5. Predict
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # 6. Result display
    col1, col2 = st.columns([1, 1])
    with col1:
        if prediction == 0:
            st.success("## Approved")
            st.markdown(
                f"<h4 style='color:#2ecc71'>Low default risk — predicted probability: "
                f"<b>{probability:.1%}</b></h4>",
                unsafe_allow_html=True,
            )
        else:
            st.error("## High Risk of Default")
            st.markdown(
                f"<h4 style='color:#e74c3c'>Elevated default risk — predicted probability: "
                f"<b>{probability:.1%}</b></h4>",
                unsafe_allow_html=True,
            )

    with col2:
        st.metric("Default Probability",   f"{probability:.1%}")
        st.metric("Loan-to-Income Ratio",  f"{loan_percent_income:.1%}")
        st.metric("Prior Default on File", prior_default)

    # 7. SHAP waterfall explanation
    st.markdown("---")
    st.subheader("Why did the AI make this decision?")
    st.caption(
        "The waterfall chart shows which features pushed the prediction toward "
        "default (red, rightward) or away from it (blue, leftward) for this specific applicant."
    )

    with st.spinner("Calculating SHAP explanation..."):
        explainer   = shap.Explainer(model, feature_names=feature_cols)
        shap_values = explainer(input_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close()
