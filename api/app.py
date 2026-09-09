from fastapi import FastAPI
import pandas as pd
import numpy as np
import shap

from api.detection_agent import detect_fraud
from api.decision_agent import make_decision
from api.investigation_agent import investigate_transaction

from pathlib import Path
import joblib

app = FastAPI(title="Agentic Fraud Detection API")


# Load model for SHAP explanations
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_fraud_model.pkl"

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

features = [
    "TransactionAmt",
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "avg_amt_1h",
    "avg_amt_24h",
    "avg_amt_7d",
    "max_amt_24h",
    "amount_zscore_24h",
    "velocity_risk",
    "is_night_txn"
]


@app.get("/")
def home():
    return {"message": "Agentic Fraud Detection API Running"}


@app.post("/predict_fraud")
def predict_fraud(transaction: dict):

    # Convert request to dataframe
    df = pd.DataFrame([transaction])

    # --------------------------
    # Detection Agent
    # --------------------------
    fraud_score = float(detect_fraud(transaction))
    print("Fraud score:", fraud_score)

    # --------------------------
    # Decision Agent
    # --------------------------
    decision = make_decision(fraud_score)

    # Default summary
    summary = "Transaction appears normal."

    explanations = []

    # --------------------------
    # Investigation Agent (only if suspicious)
    # --------------------------
    if decision != "APPROVE":

        shap_values = explainer.shap_values(df[features])

        # Top 3 important features
        top_features = np.argsort(np.abs(shap_values[0]))[::-1][:3]

        for idx in top_features:
            explanations.append({
                "feature": features[idx],
                "impact": float(shap_values[0][idx])
            })

        summary = investigate_transaction(explanations)

    return {
        "fraud_probability": float(fraud_score),
        "decision": decision,
        "explanations": explanations,
        "investigation_summary": summary
    }
