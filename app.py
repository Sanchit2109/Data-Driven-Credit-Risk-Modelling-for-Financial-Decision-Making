import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Page Config ----
st.set_page_config(page_title="Credit Risk Assessment", page_icon="🏦", layout="wide")

# ---- Load Model ----
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

pipeline = load_model()

# ---- Helper ----
def risk_bucket(pd_val):
    if pd_val < 0.10:
        return "Low", "🟢"
    elif pd_val < 0.25:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

# ---- Sidebar Inputs ----
st.sidebar.header("Applicant Details")

person_age = st.sidebar.slider("Age", 18, 80, 30)
person_income = st.sidebar.number_input("Annual Income ($)", 5000, 500000, 50000, step=1000)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.sidebar.slider("Employment Length (years)", 0, 40, 5)
loan_intent = st.sidebar.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount ($)", 500, 50000, 10000, step=500)
loan_int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.1)
loan_percent_income = st.sidebar.slider("Loan-to-Income Ratio", 0.0, 1.0, 0.2, step=0.01)
cb_person_default_on_file = st.sidebar.selectbox("Previous Default on File", ["N", "Y"])
cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (years)", 2, 30, 5)

# ---- Build Input DataFrame ----
input_data = pd.DataFrame([{
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": float(person_emp_length),
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
}])

# ---- Main Content ----
st.title("🏦 Credit Risk Assessment System")
st.markdown("Predict the **Probability of Default** and risk category for a loan applicant.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Summary")
    display_df = input_data.T.rename(columns={0: "Value"})
    display_df["Value"] = display_df["Value"].astype(str)
    st.table(display_df)

# ---- Prediction ----
if st.sidebar.button("Assess Risk", use_container_width=True):
    pd_val = pipeline.predict_proba(input_data)[0][1]
    risk, emoji = risk_bucket(pd_val)

    with col2:
        st.subheader("Risk Assessment Result")
        st.metric("Probability of Default", f"{pd_val:.1%}")
        st.markdown(f"### Risk Level: {emoji} **{risk}**")

        if risk == "Low":
            st.success("This applicant has a low risk of default. Likely eligible for favorable terms.")
        elif risk == "Medium":
            st.warning("Moderate risk. Consider additional verification or adjusted terms.")
        else:
            st.error("High risk of default. Exercise caution — may require collateral or higher rates.")

    # ---- SHAP Explanation ----
    st.subheader("Why this prediction? (SHAP Explanation)")

    try:
        X_transformed = pipeline.named_steps["preprocess"].transform(input_data)
        rf_model = pipeline.named_steps["model"]
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base = explainer.expected_value[1]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
            base = explainer.expected_value[1]
        else:
            sv = shap_values[0]
            base = explainer.expected_value

        if hasattr(X_transformed, "toarray"):
            data_row = X_transformed.toarray()[0]
        else:
            data_row = np.array(X_transformed)[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=base,
                data=data_row,
                feature_names=list(feature_names)
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")

else:
    with col2:
        st.info("👈 Fill in the applicant details and click **Assess Risk** to get a prediction.")
