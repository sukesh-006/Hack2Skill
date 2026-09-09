import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "adult_income_dataset.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "trained_model.pkl"
METRICS_JSON_PATH = MODEL_DIR / "precomputed_metrics.json"
METRICS_PKL_PATH = MODEL_DIR / "precomputed_metrics.pkl"

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def compute_fairness_metrics(df, predictions, sensitive_col, label_col="income"):
    groups = df[sensitive_col].unique()
    metrics = {}
    overall_pos_rate = predictions.mean()

    for group in groups:
        mask = df[sensitive_col] == group
        group_preds = predictions[mask]
        group_true = df[label_col].values[mask]

        tp = ((group_preds == 1) & (group_true == 1)).sum()
        fp = ((group_preds == 1) & (group_true == 0)).sum()
        tn = ((group_preds == 0) & (group_true == 0)).sum()
        fn = ((group_preds == 0) & (group_true == 1)).sum()

        pos_rate = group_preds.mean()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        acc = (tp + tn) / len(group_preds) if len(group_preds) > 0 else 0

        metrics[str(group)] = {
            "count": int(mask.sum()),
            "positive_rate": round(float(pos_rate), 4),
            "true_positive_rate": round(float(tpr), 4),
            "false_positive_rate": round(float(fpr), 4),
            "accuracy": round(float(acc), 4),
            "disparate_impact": round(float(pos_rate / overall_pos_rate), 4)
            if overall_pos_rate > 0
            else 0,
        }

    pos_rates = [v["positive_rate"] for v in metrics.values()]
    tprs = [v["true_positive_rate"] for v in metrics.values()]
    di_values = [v["disparate_impact"] for v in metrics.values()]

    bias_summary = {
        "statistical_parity_diff": round(float(max(pos_rates) - min(pos_rates)), 4),
        "equal_opportunity_diff": round(float(max(tprs) - min(tprs)), 4),
        "disparate_impact_ratio": round(float(min(di_values)), 4),
        "is_biased": (max(pos_rates) - min(pos_rates)) > 0.1 or min(di_values) < 0.8,
    }
    return metrics, bias_summary


def compute_reweighed_predictions(df, model, X_test, test_idx, sensitive_col, label_col="income"):
    groups = df.iloc[test_idx][sensitive_col]
    proba = model.predict_proba(X_test)[:, 1]
    group_thresholds = {}

    for group in groups.unique():
        mask = groups == group
        group_proba = proba[mask.values]
        target_rate = proba.mean()
        sorted_p = np.sort(group_proba)[::-1]
        n_positive = int(target_rate * len(sorted_p))
        threshold = (
            sorted_p[min(n_positive, len(sorted_p) - 1)]
            if n_positive < len(sorted_p)
            else 0.0
        )
        group_thresholds[group] = float(threshold)

    mitigated = np.zeros(len(proba), dtype=int)
    for i, (prob, group) in enumerate(zip(proba, groups)):
        mitigated[i] = 1 if prob >= group_thresholds[group] else 0
    return mitigated


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1, 0: 0, 1: 1})
    return df


def encode_dataset(df):
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


def train_models(df_enc, label_col="income"):
    feature_cols = [c for c in df_enc.columns if c != label_col]
    X = df_enc[feature_cols].values
    y = df_enc[label_col].values
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        np.arange(len(df_enc)),
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic": LogisticRegression(max_iter=500, random_state=42),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained, feature_cols, X_test, y_test, test_idx


def build_precomputed_results(df, trained_models, feature_cols, X_test, y_test, test_idx):
    test_df = df.iloc[test_idx].reset_index(drop=True)
    sensitive_columns = [col for col in ["gender", "race"] if col in df.columns]
    results = {}

    for model_name, model in trained_models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        feat_imp = {}
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for fname, imp in zip(feature_cols, importances):
                feat_imp[fname] = round(float(imp), 4)
            feat_imp = dict(sorted(feat_imp.items(), key=lambda x: -x[1])[:10])

        model_results = {}
        for sensitive_col in sensitive_columns:
            orig_metrics, orig_bias = compute_fairness_metrics(
                test_df, preds, sensitive_col, "income"
            )
            mit_preds = compute_reweighed_predictions(
                df,
                model,
                X_test,
                test_idx,
                sensitive_col,
                "income",
            )
            mit_metrics, mit_bias = compute_fairness_metrics(
                test_df, mit_preds, sensitive_col, "income"
            )
            mit_acc = accuracy_score(y_test, mit_preds)

            model_results[sensitive_col] = {
                "success": True,
                "accuracy": round(float(acc), 4),
                "mitigated_accuracy": round(float(mit_acc), 4),
                "original": {
                    "group_metrics": orig_metrics,
                    "bias_summary": orig_bias,
                },
                "mitigated": {
                    "group_metrics": mit_metrics,
                    "bias_summary": mit_bias,
                },
                "feature_importance": feat_imp,
                "sensitive_col": sensitive_col,
                "model_type": model_name,
            }
        results[model_name] = model_results
    return results


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df_enc, encoders = encode_dataset(df)
    trained_models, feature_cols, X_test, y_test, test_idx = train_models(df_enc)
    precomputed = build_precomputed_results(
        df,
        trained_models,
        feature_cols,
        X_test,
        y_test,
        test_idx,
    )

    artifact = {
        "models": trained_models,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "label_col": "income",
        "precomputed_results": precomputed,
    }

    joblib.dump(artifact, MODEL_PATH)
    with METRICS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(precomputed, f, indent=2)
    joblib.dump(precomputed, METRICS_PKL_PATH)

    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {METRICS_JSON_PATH}")
    print(f"Saved: {METRICS_PKL_PATH}")


if __name__ == "__main__":
    main()
