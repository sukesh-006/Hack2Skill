from pathlib import Path
import tempfile
import time
import warnings

from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "model" / "trained_model.pkl"
MAX_UPLOAD_ROWS = 8000
UPLOAD_SOFT_TIMEOUT_SECONDS = 8.0
TMP_DIR = tempfile.gettempdir()

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "templates"),
    static_folder=str(ROOT_DIR / "static"),
)


# ── Fairness Metrics ─────────────────────────────────────────────────────────
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


def load_adult_dataset():
    local_path = ROOT_DIR / "adult_income_dataset.csv"
    if local_path.exists():
        df = pd.read_csv(local_path)
        df["income"] = df["income"].map({"<=50K": 0, ">50K": 1, 0: 0, 1: 1})
        return df
    raise FileNotFoundError("adult_income_dataset.csv not found.")


def check_timeout(start_time, phase):
    if time.monotonic() - start_time > UPLOAD_SOFT_TIMEOUT_SECONDS:
        raise TimeoutError(
            f"Upload analysis timed out during {phase}. Reduce rows and retry (max {MAX_UPLOAD_ROWS})."
        )


def encode_dataframe(df):
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


def train_upload_logistic(df, sensitive_col, label_col):
    start = time.monotonic()
    df_enc, encoders = encode_dataframe(df)
    check_timeout(start, "encoding")

    if label_col not in df_enc.columns or sensitive_col not in df.columns:
        raise ValueError("Selected label/sensitive column was not found in dataset.")

    feature_cols = [c for c in df_enc.columns if c != label_col]
    X = df_enc[feature_cols].values
    y = df_enc[label_col].values

    if len(np.unique(y)) < 2:
        raise ValueError("Label column must contain at least 2 classes.")

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        np.arange(len(df)),
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    check_timeout(start, "split")

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    check_timeout(start, "training")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    test_df = df.iloc[test_idx].reset_index(drop=True)
    orig_metrics, orig_bias = compute_fairness_metrics(test_df, preds, sensitive_col, label_col)

    mit_preds = compute_reweighed_predictions(df, model, X_test, test_idx, sensitive_col, label_col)
    mit_metrics, mit_bias = compute_fairness_metrics(test_df, mit_preds, sensitive_col, label_col)
    mit_acc = accuracy_score(y_test, mit_preds)

    check_timeout(start, "fairness evaluation")

    result = {
        "success": True,
        "accuracy": round(float(acc), 4),
        "mitigated_accuracy": round(float(mit_acc), 4),
        "original": {"group_metrics": orig_metrics, "bias_summary": orig_bias},
        "mitigated": {"group_metrics": mit_metrics, "bias_summary": mit_bias},
        "feature_importance": {},
        "sensitive_col": sensitive_col,
        "model_type": "logistic",
    }

    return model, encoders, feature_cols, label_col, result


def load_artifact_once():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained artifact missing at {MODEL_PATH}. Run train_offline.py first."
        )
    artifact = joblib.load(MODEL_PATH)
    return artifact


ARTIFACT = None
ARTIFACT_ERROR = None
try:
    ARTIFACT = load_artifact_once()
except Exception as exc:
    ARTIFACT_ERROR = str(exc)

state = {"tmp_dir": TMP_DIR}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load_demo", methods=["POST"])
def load_demo():
    try:
        if ARTIFACT_ERROR:
            return jsonify({"success": False, "error": ARTIFACT_ERROR})

        df = load_adult_dataset()
        state["df"] = df
        state["source"] = "demo"

        return jsonify(
            {
                "success": True,
                "rows": len(df),
                "columns": list(df.columns),
                "sensitive_cols": [c for c in ["gender", "race"] if c in df.columns],
                "preview": df.head(5).to_dict(orient="records"),
                "stats": {
                    "gender_dist": df["gender"].value_counts().to_dict()
                    if "gender" in df.columns
                    else {},
                    "race_dist": df["race"].value_counts().to_dict()
                    if "race" in df.columns
                    else {},
                    "income_dist": df["income"].value_counts().to_dict()
                    if "income" in df.columns
                    else {},
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"success": False, "error": "No file uploaded."})

        df = pd.read_csv(file)
        original_rows = len(df)
        capped = False
        if original_rows > MAX_UPLOAD_ROWS:
            df = df.head(MAX_UPLOAD_ROWS).copy()
            capped = True

        state["df"] = df
        state["source"] = "upload"

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        return jsonify(
            {
                "success": True,
                "rows": len(df),
                "original_rows": original_rows,
                "capped": capped,
                "max_rows": MAX_UPLOAD_ROWS,
                "columns": list(df.columns),
                "numeric_cols": num_cols,
                "categorical_cols": cat_cols,
                "preview": df.head(5).to_dict(orient="records"),
                "tmp_dir": TMP_DIR,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json or {}
        df = state.get("df")
        if df is None:
            return jsonify({"success": False, "error": "No dataset loaded."})

        sensitive_col = data.get("sensitive_col", "gender")
        label_col = data.get("label_col", "income")
        requested_model = data.get("model", "random_forest")

        source = state.get("source", "demo")
        if source == "upload":
            model, encoders, feature_cols, out_label_col, result = train_upload_logistic(
                df, sensitive_col, label_col
            )
            state["model"] = model
            state["encoders"] = encoders
            state["feature_cols"] = feature_cols
            state["label_col"] = out_label_col
            if requested_model != "logistic":
                result["note"] = (
                    "Uploaded CSV analysis defaults to LogisticRegression for serverless runtime safety."
                )
            return jsonify(result)

        if ARTIFACT_ERROR:
            return jsonify({"success": False, "error": ARTIFACT_ERROR})

        model_key = "logistic" if requested_model == "logistic" else "random_forest"
        precomputed = ARTIFACT.get("precomputed_results", {})
        model_results = precomputed.get(model_key, {})
        if sensitive_col not in model_results:
            return jsonify(
                {
                    "success": False,
                    "error": f"Sensitive column '{sensitive_col}' is not available in precomputed demo metrics.",
                }
            )

        state["model"] = ARTIFACT["models"][model_key]
        state["encoders"] = ARTIFACT.get("encoders", {})
        state["feature_cols"] = ARTIFACT.get("feature_cols", [])
        state["label_col"] = ARTIFACT.get("label_col", "income")

        return jsonify(model_results[sensitive_col])
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/predict", methods=["POST"])
def predict_single():
    try:
        data = request.json or {}
        model = state.get("model")
        if model is None:
            return jsonify({"success": False, "error": "Run analysis first."})

        encoders = state.get("encoders", {})
        feature_cols = state.get("feature_cols")
        label_col = state.get("label_col", "income")
        df = state.get("df")

        if not feature_cols:
            if df is None:
                return jsonify({"success": False, "error": "No features available."})
            feature_cols = [c for c in df.columns if c != label_col]

        input_row = {}
        for col in feature_cols:
            default_val = None
            if df is not None and col in df.columns and not df[col].isna().all():
                default_val = df[col].mode(dropna=True)[0]

            val = data.get(col, default_val)
            if col in encoders:
                encoder = encoders[col]
                val_str = str(val)
                if val_str in encoder.classes_:
                    val = int(encoder.transform([val_str])[0])
                else:
                    val = 0
            else:
                val = float(val) if val is not None else 0.0

            input_row[col] = [float(val)]

        X_input = pd.DataFrame(input_row)[feature_cols].values
        prob = model.predict_proba(X_input)[0][1]
        pred = int(prob >= 0.5)

        return jsonify(
            {
                "success": True,
                "prediction": ">50K" if pred == 1 else "<=50K",
                "probability": round(float(prob), 4),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
