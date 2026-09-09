from __future__ import annotations

"""Fraud Detection API: hybrid XGBoost + Isolation Forest anomaly
backstop, SHAP explainability, optional LLM investigation summary.

Run:  uvicorn api.main:app --reload   (from the repo root, .venv active)
Docs: http://127.0.0.1:8000/docs
"""

import time
import joblib
import numpy as np
import shap
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.decision_agent import make_decision
from api.detection_agent import detect_fraud, prepare_features, get_feature_vector
from api.investigation_agent import investigate_transaction

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="Fraud Detection API", version="3.0")

bundle = joblib.load(MODELS_DIR / "xgb_production_model.pkl")
model = bundle["model"]
THRESHOLDS = {
    "block": bundle["block_threshold"],
    "investigate": bundle["investigate_threshold"],
    "anomaly": bundle["anomaly_threshold"],
}
explainer = shap.TreeExplainer(model)
FEATURE_NAMES = model.get_booster().feature_names


class TransactionRequest(BaseModel):
    """Raw features only; the API computes anomaly_score internally."""
    features: dict = Field(
        ...,
        description="Raw feature dict (e.g. TransactionAmt, velocity_risk). "
                    "Do NOT send anomaly_score; the API computes it.",
    )


class FraudResponse(BaseModel):
    fraud_probability: float
    anomaly_score: float
    decision: str
    triggered_rule: str
    investigation_summary: str
    explanations: list
    processing_time_ms: float


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "model": "hybrid XGBoost + Isolation Forest anomaly backstop",
        "thresholds": THRESHOLDS,
    }


@app.post("/predict_fraud", response_model=FraudResponse)
def predict_fraud(req: TransactionRequest):
    t0 = time.time()

    try:
        scores = detect_fraud(req.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    decision, rule = make_decision(
        scores["fraud_probability"], scores["anomaly_score"], THRESHOLDS
    )

    # SHAP frame rebuilt via the same helpers, so it is guaranteed
    # identical to what the model scored (incl. derived velocity_risk).
    df = prepare_features(req.features)
    df["anomaly_score"] = scores["anomaly_score"]
    sv = explainer.shap_values(get_feature_vector(df))
    if isinstance(sv, list):                       # older SHAP/XGBoost
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv).ravel()
    order = np.argsort(-np.abs(sv))
    explanations = [
        {"feature": FEATURE_NAMES[i], "impact": float(sv[i])}
        for i in order[:5]
    ]

    summary = (
        investigate_transaction(explanations, {
            "decision": decision,
            "triggered_rule": rule,
            "fraud_probability": scores["fraud_probability"],
            "anomaly_score": scores["anomaly_score"],
        })
        if decision != "APPROVE"
        else "Transaction auto-approved: scores below alert thresholds."
    )

    return FraudResponse(
        fraud_probability=scores["fraud_probability"],
        anomaly_score=scores["anomaly_score"],
        decision=decision,
        triggered_rule=rule,
        investigation_summary=summary,
        explanations=explanations,
        processing_time_ms=round((time.time() - t0) * 1000, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)