# Fraud Detection and AI Investigation Assistant

A Python project combining supervised fraud scoring, anomaly detection, model explanations, and AI-generated review summaries through a FastAPI service and Streamlit dashboard.

The project explores how transaction history can support a fraud-review workflow: score a transaction, identify the rule that triggered an alert, explain the supervised model’s output, and present evidence for human review.

> **Project status:** The implementation has been revised and targeted code checks have passed. Full execution with regenerated model artifacts and final out-of-time performance results remain to be verified. This is a local prototype.

## Project Overview

The workflow combines:

- **XGBoost** for supervised fraud risk scoring.
- **Isolation Forest** for identifying unusual transaction behavior.
- **Decision rules** using thresholds selected on validation data.
- **SHAP** for explaining the XGBoost model’s output.
- **Optional OpenAI summaries** for translating model evidence into short review notes.
- **FastAPI** for serving predictions.
- **Streamlit** for entering transaction features and displaying results.

The current system follows a fixed workflow with optional LLM narration. It does not autonomously collect evidence, select investigation tools, or resolve fraud cases.

## Workflow

1. Prepare and validate transaction data.
2. Build historical transaction features.
3. Train supervised and anomaly-detection models.
4. Select decision thresholds using validation data.
5. Evaluate performance on a later test period.
6. Serve scores and simulated recommendations through the API.
7. Display explanations and review summaries in the dashboard.

XGBoost and Isolation Forest operate separately. Their outputs are combined in the decision engine; the anomaly score is not an input to XGBoost.

## Transaction Features

Historical features are calculated from prior transactions, excluding the current transaction from its historical window.

| Feature group | Examples | Purpose |
| --- | --- | --- |
| Current transaction | `TransactionAmt` | Transaction amount |
| Transaction frequency | `txn_count_1h`, `txn_count_24h`, `txn_count_7d` | Recent activity volume |
| Historical amounts | `avg_amt_1h`, `avg_amt_24h`, `avg_amt_7d` | Previous transaction averages |
| Historical maximum | `max_amt_24h` | Largest prior amount in the window |
| Amount deviation | `amount_zscore_24h` | Deviation from historical amounts |
| Activity concentration | `velocity_risk` | Prior 1-hour count divided by prior 24-hour count |
| Relative time | `relative_hour` | Hour within the dataset’s relative time cycle |

`relative_hour` is not a verified local clock hour. Card-profile groupings are proxies for transaction history, not verified customer identities.

The API accepts precomputed historical features. It does not retrieve history from a live transaction database.

### Data Documentation

The final dataset source, license, row count, class balance, and time coverage still need to be documented from the corrected run. No dataset size or model-performance figure is claimed until verified.

## Modeling Approach

### XGBoost

The supervised model learns from labeled transactions. Class weighting is calculated using training data only.

Its output is displayed as a **risk score**, not a calibrated probability of fraud.

### Isolation Forest

Isolation Forest identifies unusual combinations of transaction features without using fraud labels during training.

Preprocessing is fitted on historical training data and reused for later transactions. Higher raw anomaly scores indicate more unusual behavior, but an anomaly is not proof of fraud.

### Chronological Evaluation

The corrected Isolation Forest and hybrid notebooks use approximately:

- **70%** training data
- **15%** validation data
- **15%** test data

Transactions are ordered by time, and equal timestamps stay together. The hybrid notebook reuses and checks the anomaly pipeline’s split boundaries.

Thresholds are selected using validation data. The later test period is reserved for final evaluation.

### Evaluation Metrics

The workflow reports:

- ROC-AUC
- Average precision
- Alert precision
- Recall
- Alert rate
- False positives

The hybrid comparison examines whether adding anomaly alerts increases fraud coverage and review volume. It is not an equal-review-budget comparison and does not establish financial savings.

**Final results are pending a complete rerun of the corrected pipeline.**

## Decision Policy

Rules are evaluated in the following order:

| Recommendation | Condition |
| --- | --- |
| `BLOCK` | XGBoost score meets or exceeds the saved block threshold |
| `INVESTIGATE` | No block rule fired, and the raw anomaly score exceeds its saved threshold |
| `APPROVE` | Neither rule fired |

The block threshold targets **80% precision on validation data**. If no threshold meets that target, blocking is disabled.

The anomaly threshold uses an illustrative **3% validation review budget**. Tied scores and distribution changes can affect the actual alert rate.

These settings are experimental assumptions, not validated banking policies. Moderate XGBoost scores alone do not trigger investigation under the current rules.

All decisions are simulated recommendations. The application does not block payments or confirm that approved transactions are legitimate.

## Explainability

SHAP identifies features contributing to the XGBoost output:

- Positive contributions increase the model’s raw score.
- Negative contributions decrease the model’s raw score.
- Contributions are not probability-point changes or causal explanations.
- SHAP explanations do not explain the Isolation Forest signal.

The dashboard displays selected feature contributions alongside the rule that triggered the recommendation.

## Optional AI Review Summaries

The summary module receives structured evidence, including:

- Model scores
- The triggered decision rule
- Saved thresholds
- Selected XGBoost feature contributions

It generates a short review note while being instructed to avoid inventing transaction history or treating model outputs as confirmed fraud.

The LLM cannot override the decision rules.

AI summaries are disabled by default. Missing credentials, request failures, or empty responses fall back to a rule-based summary. Generated summaries still require review.

## Application Structure

| Path | Responsibility |
| --- | --- |
| `notebooks/` | Data preparation, feature engineering, training, and evaluation |
| `api/main.py` | FastAPI application and endpoints |
| `api/schemas.py` | Transaction input validation |
| `api/detection_agent.py` | Model loading, scoring, and explanations |
| `api/decision_agent.py` | Saved decision-rule application |
| `api/investigation_agent.py` | Optional AI summaries and fallback |
| `dashboard/app.py` | Streamlit interface |
| `models/hybrid_fraud_policy.pkl` | Generated model bundle used by the API |
| `requirements-app.txt` | Application dependencies |

The module names retain `agent` for project continuity; they do not imply autonomous behavior.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- FastAPI and Pydantic
- Streamlit
- joblib
- OpenAI Python SDK

## Running Locally

### 1. Install Dependencies

From the repository root, in your project environment:

```bash
python -m pip install -r requirements-app.txt
```

Use the same Python and machine-learning package versions for training and serving. The application dependency list is not a reproducibility lockfile.

### 2. Generate Model Artifacts

Run the corrected notebooks in dependency order:

1. Data preparation
2. Historical feature engineering
3. Isolation Forest
4. Hybrid model and evaluation

Run notebooks from the `notebooks/` directory so their relative paths resolve correctly.

The downstream workflow expects:

```text
data/processed/time_window_features.csv
data/processed/fraud_features_with_anomaly_iforest.csv
models/iforest_anomaly_policy.pkl
models/hybrid_fraud_policy.pkl
```

The hybrid notebook also exports test-period decisions to:

```text
data/processed/fraud_predictions.csv
```

The standalone XGBoost notebook provides a separate baseline. Its model artifact is not the one loaded by the application.

### 3. Start FastAPI

From the repository root:

```bash
python -m uvicorn api.main:app --reload
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 4. Start Streamlit

In a second terminal, from the repository root:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard calls the local FastAPI endpoint by default. Set `FRAUD_API_URL` in the dashboard process’s environment if the API runs elsewhere.

### 5. Enable AI Summaries — Optional

Create a local `.env` file in the repository root:

```dotenv
ENABLE_AI_SUMMARY=true
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Keep `.env` out of Git.

When enabled, structured model evidence and selected feature values are sent to OpenAI for summary generation. An API key is not required for local scoring and rule-based summaries.

## API Example

### Endpoint

```text
POST /predict_fraud
```

### Example Request

```json
{
  "TransactionAmt": 100.0,
  "txn_count_1h": 1,
  "txn_count_24h": 5,
  "txn_count_7d": 10,
  "avg_amt_1h": 50.0,
  "avg_amt_24h": 50.0,
  "avg_amt_7d": 50.0,
  "max_amt_24h": 100.0,
  "amount_zscore_24h": null,
  "relative_hour": 12
}
```

These values illustrate the request format, not an observed transaction.

Counts must be nonnegative and satisfy:

```text
txn_count_1h <= txn_count_24h <= txn_count_7d
```

Use null for unavailable historical statistics. The server computes `velocity_risk`.

### Response Fields

| Field | Meaning |
| --- | --- |
| `xgb_score` | Supervised model risk score |
| `anomaly_score_raw` | Isolation Forest anomaly score |
| `decision` | Simulated recommendation |
| `reason` | Rule responsible for the recommendation |
| `explanations` | Selected XGBoost feature contributions |
| `investigation_summary` | Review note |
| `summary_source` | Rule-based, AI-generated, or fallback |
| `prototype_only` | Indicates prototype status |

## Verification Status

Completed checks include:

- Python syntax validation
- Selected request-validation cases
- Decision threshold boundaries
- Disabled-blocking behavior
- Offline summary fallback

Remaining checks include:

- Complete execution of the corrected notebooks
- Model loading in the serving environment
- An API-to-dashboard prediction
- Live AI-summary behavior
- Final out-of-time performance reporting

## Limitations

- Model scores are not calibrated fraud probabilities.
- Anomalous behavior does not establish fraud.
- Transaction-history groupings may not represent individual customers.
- Historical performance may not transfer to future transactions.
- Dashboard inputs are manually supplied historical features.
- AI-generated summaries may contain errors.
- The prototype does not include authentication, production monitoring, a live history service, or case-management integration.

Keep the unauthenticated application local and load only trusted model artifacts.

## Author

**Tsering Tashi Gurung**

MBA graduate and PhD student in Information Technology with an AI emphasis.

Interests include data analytics, AI adoption, and translating model outputs into business decisions.
