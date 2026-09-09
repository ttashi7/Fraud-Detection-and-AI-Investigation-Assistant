from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import shap
import joblib
import logging
from pathlib import Path

from api.detection_agent import detect_fraud          # requires the rename fix
from api.decision_agent import make_decision
from api.investigation_agent import investigate_transaction

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# --- Load the model BUNDLE from notebook 06: model + features + thresholds ---
bundle = joblib.load(MODELS_DIR / "xgb_hybrid_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]                        # single source of truth
THRESHOLDS = {
    "block": bundle["block_threshold"],
    "investigate": bundle["investigate_threshold"],
    "anomaly": bundle["anomaly_threshold"],
}

explainer = shap.TreeExplainer(model)

app = FastAPI(title="Agentic Fraud Detection API")


@app.get("/")
def home():
    return {"message": "Agentic Fraud Detection API Running",
            "model_features": len(FEATURES),
            "thresholds": THRESHOLDS}


@app.post("/predict_fraud")
def predict_fraud(transaction: dict):

    # Detection agent now returns BOTH scores
    scores = detect_fraud(transaction)
    fraud_score = float(scores["fraud_probability"])
    anomaly_score = float(scores["anomaly_score"])
    logger.info("fraud_score=%.4f anomaly_score=%.4f", fraud_score, anomaly_score)

    # Decision agent applies the three-tier policy from the bundle
    decision = make_decision(fraud_score, anomaly_score, THRESHOLDS)

    explanations = []
    summary = "Transaction appears normal."

    # Investigation agent only runs on non-APPROVE (saves LLM calls)
    if decision != "APPROVE":
        df = pd.DataFrame([transaction])
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Request missing features: {missing}",
            )

        shap_values = explainer.shap_values(df[FEATURES])[0]
        top_features = np.argsort(np.abs(shap_values))[::-1][:3]
        explanations = [
            {"feature": FEATURES[i], "impact": float(shap_values[i])}
            for i in top_features
        ]
        summary = investigate_transaction(explanations)

    return {
        "fraud_probability": fraud_score,
        "anomaly_score": anomaly_score,
        "decision": decision,
        "explanations": explanations,
        "investigation_summary": summary,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
