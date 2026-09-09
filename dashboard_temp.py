
import streamlit as st
import requests
import pandas as pd

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict_fraud"

# Page config
st.set_page_config(
    page_title="Agentic AI Fraud Detection",
    layout="wide"
)

st.title("💳 Agentic AI Fraud Detection Dashboard")

st.markdown(
    "This dashboard analyzes transaction behavior using an **AI fraud detection system**."
)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Transaction Input")

transaction = {
    "TransactionAmt": st.sidebar.number_input("Transaction Amount", value=100),
    "txn_count_1h": st.sidebar.number_input("Transactions Last 1h", value=1),
    "txn_count_24h": st.sidebar.number_input("Transactions Last 24h", value=5),
    "txn_count_7d": st.sidebar.number_input("Transactions Last 7d", value=10),
    "avg_amt_1h": st.sidebar.number_input("Avg Amount 1h", value=50),
    "avg_amt_24h": st.sidebar.number_input("Avg Amount 24h", value=50),
    "avg_amt_7d": st.sidebar.number_input("Avg Amount 7d", value=50),
    "max_amt_24h": st.sidebar.number_input("Max Amount 24h", value=100),
    "amount_zscore_24h": st.sidebar.number_input("Amount Z-Score", value=1.0),
    "velocity_risk": st.sidebar.slider("Velocity Risk", 0.0, 1.0, 0.2),
    "is_night_txn": st.sidebar.selectbox("Night Transaction", [0, 1])
}

analyze_button = st.sidebar.button("Analyze Transaction")

# -------------------------
# Main Panel
# -------------------------
if analyze_button:

    st.subheader("Transaction Analysis")

    try:
        response = requests.post(API_URL, json=transaction)

        if response.status_code == 200:

            result = response.json()

            fraud_probability = result["fraud_probability"]
            decision = result["decision"]

            # -------------------------
            # Fraud Score Display
            # -------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Fraud Probability",
                    value=round(fraud_probability, 3)
                )

            with col2:

                if decision == "BLOCK":
                    st.error(f"Decision: {decision}")

                elif decision == "INVESTIGATE":
                    st.warning(f"Decision: {decision}")

                else:
                    st.success(f"Decision: {decision}")

            # -------------------------
            # Investigation Summary
            # -------------------------
            st.subheader("Investigation Summary")

            st.write(result["investigation_summary"])

            # -------------------------
            # Risk Features Chart
            # -------------------------
            if "explanations" in result and result["explanations"]:

                st.subheader("Top Risk Features")

                df = pd.DataFrame(result["explanations"])

                st.bar_chart(df.set_index("feature"))

        else:
            st.error("API returned an error.")

    except Exception as e:
        st.error("Could not connect to the Fraud Detection API.")
        st.write(e)

else:

    st.info(
        "Enter transaction details in the sidebar and click **Analyze Transaction**."
    )

    st.image(
        "https://miro.medium.com/v2/resize:fit:1400/1*8cZ0H6kF0Wn0uRlaRykWBA.png",
        caption="AI-powered fraud detection workflow",
        use_column_width=True
    )
