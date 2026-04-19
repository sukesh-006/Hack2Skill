# FairLens — AI Bias Detection Platform
### Hack2Skill 2026 | [Unbiased AI Decision] Track

A fully working web application that detects and mitigates bias in AI/ML models.
Built with Python + Flask + scikit-learn.

---

## Features
- Load the built-in UCI Adult Income dataset OR upload your own CSV
- Train Random Forest or Logistic Regression classifier
- Compute 4 fairness metrics: Statistical Parity, Equal Opportunity, Disparate Impact, Accuracy
- Interactive charts showing bias across groups (gender, race, etc.)
- Bias mitigation via threshold adjustment (post-processing)
- Before vs After comparison with radar chart
- Live prediction demo — see how predictions change across demographic groups

---

## Setup & Run (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
python app.py
```

### Step 3 — Open browser
```
http://localhost:5000
```

---

## How to Demo (for judges)

1. **Setup tab** → Click "Load Demo Dataset" (48K records loads in seconds)
2. **Analyze tab** → Select `gender` as sensitive attribute → Click "Run Bias Analysis"
3. **Results tab** → Show the bias alert, metric cards, and group comparison charts
4. **Mitigation tab** → Show before/after bars and radar chart improvement
5. **Predict tab** → Enter a profile, predict income, then change only "Gender" and re-predict to visually show bias

### Key talking points for judges:
- Statistical Parity Diff > 0.1 means the model is unfair
- Disparate Impact < 0.8 violates the "80% rule" used in US employment law
- Our mitigation reduces bias while keeping accuracy above 80%

---

## File Structure
```
bias_detector/
├── app.py              ← Flask backend + ML pipeline
├── requirements.txt    ← Dependencies
├── README.md
└── templates/
    └── index.html      ← Full frontend (single file)
```

---

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, Flask |
| ML | scikit-learn (RandomForest, LogisticRegression) |
| Fairness | Custom metrics (Statistical Parity, Equal Opportunity, Disparate Impact) |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Dataset | UCI Adult Income (auto-downloaded, no setup needed) |

---

## Fairness Metrics Explained

| Metric | Formula | Fair Threshold |
|--------|---------|---------------|
| Statistical Parity Diff | max(P(ŷ=1\|A=a)) − min(P(ŷ=1\|A=a)) | < 0.1 |
| Equal Opportunity Diff | max(TPR_a) − min(TPR_a) | < 0.1 |
| Disparate Impact Ratio | min(P(ŷ=1\|A=a)) / P(ŷ=1) | ≥ 0.8 |
| Equalized Odds | TPR and FPR equal across groups | Visualized |
