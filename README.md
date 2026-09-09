# FairLens — AI Bias Detection Platform
### Hack2Skill 2026 | [Unbiased AI Decision] Track

A web app that detects and mitigates bias in ML predictions using Flask + scikit-learn, now structured for Vercel serverless deployment.

---

## Features
- Load the built-in Adult Income dataset or upload your own CSV
- Analyze bias with RandomForest or LogisticRegression on demo data (pre-trained offline)
- Compute 4 fairness metrics: Statistical Parity, Equal Opportunity, Disparate Impact, Accuracy
- Bias mitigation via per-group threshold adjustment (before vs after comparison)
- Interactive client-side charts using Chart.js
- Live prediction demo across demographic groups

---

## Setup & Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train models offline (required before first deploy/run)
```bash
python train_offline.py
```
This generates:
- `model/trained_model.pkl`
- `model/precomputed_metrics.json`
- `model/precomputed_metrics.pkl`

### 3) Run locally
```bash
flask --app api.index run --debug
```

Open:
```
http://localhost:5000
```

---

## Serverless Notes (Vercel)
- Demo analysis uses models loaded once at cold start from `model/trained_model.pkl`.
- No live retraining of the full 48K demo dataset inside request handlers.
- Uploaded CSV analysis is capped to **8,000 rows** to keep request-time LogisticRegression training within serverless execution limits.
- Any temporary file writes must use `/tmp` (via Python `tempfile`).

---

## File Structure
```
Hack2Skill/
├── api/
│   └── index.py
├── model/
│   ├── trained_model.pkl
│   ├── precomputed_metrics.json
│   └── precomputed_metrics.pkl
├── templates/
│   └── index.html
├── static/
├── train_offline.py
├── vercel.json
├── requirements.txt
├── adult_income_dataset.csv
└── README.md
```

---

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, Flask |
| ML | scikit-learn (RandomForest, LogisticRegression) |
| Fairness | Custom metrics (Statistical Parity, Equal Opportunity, Disparate Impact) |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Dataset | Adult Income CSV |

---

## Fairness Metrics
| Metric | Formula | Fair Threshold |
|--------|---------|---------------|
| Statistical Parity Diff | max(P(ŷ=1\|A=a)) − min(P(ŷ=1\|A=a)) | < 0.1 |
| Equal Opportunity Diff | max(TPR_a) − min(TPR_a) | < 0.1 |
| Disparate Impact Ratio | min(P(ŷ=1\|A=a)) / P(ŷ=1) | ≥ 0.8 |
| Accuracy | Correct predictions / total predictions | Higher is better |
