from __future__ import annotations

"""Decision agent: three-tier fraud policy.

Thresholds come from the model bundle exported by notebook 06
(models/xgb_production_model.pkl), so the API always applies the same
policy that was evaluated on the future test window.

Tiers:
    BLOCK       -> high-confidence fraud, auto-decline
    INVESTIGATE -> mid-range fraud score, OR extreme anomaly score
                   (the anomaly backstop catches novel attack patterns
                   the supervised model has never seen)
    APPROVE     -> everything else
"""

# Fallbacks = the policy evaluated in notebooks 04/06 on the future
# test window. At runtime the model bundle's thresholds override these.
DEFAULT_THRESHOLDS = {"block": 0.90, "investigate": 0.70, "anomaly": 0.80}


def make_decision(
    fraud_score: float,
    anomaly_score: float = 0.0,
    thresholds: dict | None = None,
) -> tuple[str, str]:
    """Return (decision, triggered_rule).

    The rule name is returned because the review workflow needs to know
    WHICH signal fired: the supervised model or the anomaly backstop.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    if fraud_score >= t["block"]:
        return "BLOCK", "xgb_score >= block_threshold"

    if fraud_score >= t["investigate"]:
        return "INVESTIGATE", "xgb_score >= investigate_threshold"

    if anomaly_score >= t["anomaly"]:
        return "INVESTIGATE", "anomaly_score >= anomaly_threshold"

    return "APPROVE", "below_thresholds"