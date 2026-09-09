from __future__ import annotations
"""Investigation agent: LLM-generated review summary for analysts.

Turns model evidence (SHAP drivers, triggered rule, scores) into a
short narrative. This is an ENHANCEMENT layer: if the LLM is
unavailable (no key, network, quota), the API still returns the
decision and SHAP evidence. Fail-open, never block the verdict.

Data note: only model evidence and scores are sent to the LLM -
never raw card numbers or customer identifiers.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_client = None

FALLBACK = ("Automated summary unavailable. Review the model drivers "
            "and scores listed above manually.")


def _get_client():
    """Lazy client creation: the API starts even without a key set."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=15)
    return _client


def investigate_transaction(explanations: list, context: dict | None = None) -> str:
    context = context or {}

    evidence = "\n".join(
        f"- {item['feature']}: impact {item['impact']:+.3f} "
        f"({'raised' if item['impact'] > 0 else 'lowered'} the fraud score)"
        for item in explanations
    )

    prompt = f"""You are assisting a human fraud analyst.

Alert details:
- Decision tier: {context.get('decision', 'INVESTIGATE')}
- Triggered rule: {context.get('triggered_rule', 'unknown')}
- Model fraud probability: {context.get('fraud_probability', 0):.2f}
- Anomaly score: {context.get('anomaly_score', 0):.2f}

Top model drivers (SHAP):
{evidence}

Write 2-3 sentences for the analyst: why this transaction may be
suspicious and what to verify first. Be specific to the drivers above.
Do not invent transaction details you were not given."""

    try:
        resp = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise fraud investigation analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Investigation agent unavailable: %s", e)
        return FALLBACK
