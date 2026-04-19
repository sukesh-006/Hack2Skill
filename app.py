from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import json
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Adult Income dataset (built-in, no download needed) ──────────────────────
def load_adult_dataset():
    import os
    # 1. Try local CSV in same folder (always works, no internet needed)
    local_path = os.path.join(os.path.dirname(__file__), 'adult_income_dataset.csv')
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, 0: 0, 1: 1})
        return df

    # 2. Try downloading from UCI (needs internet)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    try:
        df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
        df.dropna(inplace=True)
        df['income'] = df['income'].str.strip().map({'<=50K': 0, '>50K': 1})
        df['gender'] = df['gender'].str.strip()
        df['race'] = df['race'].str.strip()
        return df
    except Exception:
        pass

    # 3. Last resort: generate synthetic
    return generate_synthetic_dataset()

def generate_synthetic_dataset():
    np.random.seed(42)
    n = 5000
    genders = np.random.choice(['Male', 'Female'], n, p=[0.67, 0.33])
    races = np.random.choice(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], n, p=[0.85, 0.095, 0.032, 0.01, 0.013])
    ages = np.random.randint(18, 90, n)
    edu = np.random.randint(1, 17, n)
    hours = np.random.randint(10, 100, n)

    # Biased income probability
    prob = 0.1 + 0.3 * (edu / 16) + 0.1 * (hours / 100)
    prob += np.where(genders == 'Male', 0.15, 0.0)
    prob += np.where(races == 'White', 0.05, 0.0)
    prob = np.clip(prob, 0, 1)
    income = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'age': ages, 'workclass': 'Private', 'fnlwgt': 100000,
        'education': 'Bachelors', 'education_num': edu,
        'marital_status': 'Never-married', 'occupation': 'Tech-support',
        'relationship': 'Not-in-family', 'race': races, 'gender': genders,
        'capital_gain': 0, 'capital_loss': 0, 'hours_per_week': hours,
        'native_country': 'United-States', 'income': income
    })
    return df

# ── Fairness Metrics ─────────────────────────────────────────────────────────
def compute_fairness_metrics(df, predictions, sensitive_col, label_col='income'):
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
            'count': int(mask.sum()),
            'positive_rate': round(float(pos_rate), 4),
            'true_positive_rate': round(float(tpr), 4),
            'false_positive_rate': round(float(fpr), 4),
            'accuracy': round(float(acc), 4),
            'disparate_impact': round(float(pos_rate / overall_pos_rate), 4) if overall_pos_rate > 0 else 0
        }

    # Bias scores
    pos_rates = [v['positive_rate'] for v in metrics.values()]
    tprs = [v['true_positive_rate'] for v in metrics.values()]
    di_values = [v['disparate_impact'] for v in metrics.values()]

    bias_summary = {
        'statistical_parity_diff': round(float(max(pos_rates) - min(pos_rates)), 4),
        'equal_opportunity_diff': round(float(max(tprs) - min(tprs)), 4),
        'disparate_impact_ratio': round(float(min(di_values)), 4),
        'is_biased': (max(pos_rates) - min(pos_rates)) > 0.1 or min(di_values) < 0.8
    }
    return metrics, bias_summary

def compute_reweighed_predictions(df, model, X_test, test_idx, sensitive_col, label_col='income'):
    """Simple threshold adjustment per group as mitigation."""
    groups = df.iloc[test_idx][sensitive_col]
    proba = model.predict_proba(X_test)[:, 1]
    group_thresholds = {}

    for group in groups.unique():
        mask = groups == group
        group_proba = proba[mask.values]
        # Adjust threshold to equalize positive rates
        target_rate = proba.mean()
        sorted_p = np.sort(group_proba)[::-1]
        n_positive = int(target_rate * len(sorted_p))
        threshold = sorted_p[min(n_positive, len(sorted_p)-1)] if n_positive < len(sorted_p) else 0.0
        group_thresholds[group] = float(threshold)

    mitigated = np.zeros(len(proba), dtype=int)
    for i, (prob, group) in enumerate(zip(proba, groups)):
        mitigated[i] = 1 if prob >= group_thresholds[group] else 0
    return mitigated

# ── Global state ─────────────────────────────────────────────────────────────
state = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_demo', methods=['POST'])
def load_demo():
    try:
        df = load_adult_dataset()
        state['df'] = df
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': list(df.columns),
            'sensitive_cols': ['gender', 'race'],
            'preview': df.head(5).to_dict(orient='records'),
            'stats': {
                'gender_dist': df['gender'].value_counts().to_dict(),
                'race_dist': df['race'].value_counts().to_dict(),
                'income_dist': df['income'].value_counts().to_dict()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        state['df'] = df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': list(df.columns),
            'numeric_cols': num_cols,
            'categorical_cols': cat_cols,
            'preview': df.head(5).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        df = state.get('df')
        if df is None:
            return jsonify({'success': False, 'error': 'No dataset loaded.'})

        sensitive_col = data.get('sensitive_col', 'gender')
        label_col = data.get('label_col', 'income')
        model_type = data.get('model', 'random_forest')

        # Encode features
        df_enc = df.copy()
        le = LabelEncoder()
        for col in df_enc.select_dtypes(include='object').columns:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))

        feature_cols = [c for c in df_enc.columns if c != label_col]
        X = df_enc[feature_cols].values
        y = df_enc[label_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        test_idx = list(range(len(X) - len(X_test), len(X)))

        if model_type == 'logistic':
            model = LogisticRegression(max_iter=500, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        test_df = df.iloc[len(X_train):].reset_index(drop=True)

        # Original metrics
        orig_metrics, orig_bias = compute_fairness_metrics(test_df, preds, sensitive_col, label_col)

        # Mitigated metrics
        mit_preds = compute_reweighed_predictions(df, model, X_test, list(range(len(X_train), len(X))), sensitive_col, label_col)
        mit_metrics, mit_bias = compute_fairness_metrics(test_df, mit_preds, sensitive_col, label_col)
        mit_acc = accuracy_score(y_test, mit_preds)

        # Feature importance
        feat_imp = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for fname, imp in zip(feature_cols, importances):
                feat_imp[fname] = round(float(imp), 4)
            feat_imp = dict(sorted(feat_imp.items(), key=lambda x: -x[1])[:10])

        state['model'] = model
        state['test_df'] = test_df
        state['sensitive_col'] = sensitive_col

        return jsonify({
            'success': True,
            'accuracy': round(float(acc), 4),
            'mitigated_accuracy': round(float(mit_acc), 4),
            'original': {'group_metrics': orig_metrics, 'bias_summary': orig_bias},
            'mitigated': {'group_metrics': mit_metrics, 'bias_summary': mit_bias},
            'feature_importance': feat_imp,
            'sensitive_col': sensitive_col,
            'model_type': model_type
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/predict', methods=['POST'])
def predict_single():
    try:
        data = request.json
        df = state.get('df')
        model = state.get('model')
        if df is None or model is None:
            return jsonify({'success': False, 'error': 'Run analysis first.'})

        df_enc = df.copy()
        le = LabelEncoder()
        encoders = {}
        for col in df_enc.select_dtypes(include='object').columns:
            encoders[col] = le.fit(df_enc[col].astype(str))
            df_enc[col] = encoders[col].transform(df_enc[col].astype(str))

        label_col = 'income'
        feature_cols = [c for c in df_enc.columns if c != label_col]

        input_row = {}
        for col in feature_cols:
            val = data.get(col, df[col].mode()[0])
            if col in encoders:
                try:
                    val = encoders[col].transform([str(val)])[0]
                except:
                    val = 0
            input_row[col] = [float(val)]

        X_input = pd.DataFrame(input_row)[feature_cols].values
        prob = model.predict_proba(X_input)[0][1]
        pred = int(prob >= 0.5)

        return jsonify({
            'success': True,
            'prediction': '>50K' if pred == 1 else '<=50K',
            'probability': round(float(prob), 4)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
