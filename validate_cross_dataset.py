"""
EEQ401 — CROSS-DATASET GENERALIZATION VALIDATION

Validates that the deep neural security classifier trained on dataset_100k
generalizes accurately across the independent 200k and 300k datasets.

Evaluates:
  - Accuracy on 10,000 random samples from dataset_200k
  - Accuracy on 10,000 random samples from dataset_300k
  - Macro F1, Precision, and Recall across both unseen sets
  - Outputs summary metrics JSON to models/classifier_100k/cross_dataset_validation.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure console utf-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.rl.classifier_integration import GridSecurityClassifier, LABEL_MAP, CONTINGENCY_ELEMENTS


def evaluate_on_dataset(csv_path: str, model, scaler_mean, scaler_scale, device, sample_size: int = 10000):
    print(f"\n[+] Loading {sample_size:,} sample rows from {csv_path}...")
    t0 = time.time()
    
    # Read sample
    df = pd.read_csv(csv_path, nrows=sample_size * 2)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    df['convergence_flag'] = df['convergence'].astype(float)
    df['min_bus_voltage'] = df['min_bus_voltage'].fillna(0.0)
    df['max_bus_voltage'] = df['max_bus_voltage'].fillna(0.0)
    df['max_line_loading'] = df['max_line_loading'].fillna(999.0)
    df['max_voltage_deviation'] = df['max_voltage_deviation'].fillna(1.0)
    df['total_load'] = df['total_load'].fillna(259.0)

    for c in df.columns:
        if c.startswith(('V_bus', 'angle_bus', 'gen_P_')):
            df[c] = df[c].fillna(0.0)
        elif c.startswith('line_loading_'):
            df[c] = df[c].fillna(999.0)

    feature_cols = [c for c in df.columns if c.startswith(('V_bus', 'angle_bus', 'line_loading_', 'gen_P_')) 
                    or c in ['min_bus_voltage', 'max_bus_voltage', 'max_line_loading', 'max_voltage_deviation', 'total_load', 'convergence_flag']]
    
    # One-hot encoding matching alphabetical order
    sorted_elements = sorted(CONTINGENCY_ELEMENTS)
    elem_dummies = pd.get_dummies(df['contingency_element'], prefix='outage', dtype=float)
    for elem in sorted_elements:
        col = f"outage_{elem}"
        if col not in elem_dummies.columns:
            elem_dummies[col] = 0.0
    elem_dummies = elem_dummies[[f"outage_{elem}" for elem in sorted_elements]]

    X_cont = df[feature_cols].values.astype(np.float32)
    X_outage = elem_dummies.values.astype(np.float32)

    # Scale continuous
    X_cont = (X_cont - scaler_mean) / np.maximum(scaler_scale, 1e-7)
    X = np.hstack([X_cont, X_outage])
    y_true = df['label'].map(LABEL_MAP).values.astype(np.int64)

    # Predict in batches
    model.eval()
    batch_size = 512
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.from_numpy(X[i:i+batch_size]).to(device)
            out = model(bx)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p)

    preds = np.array(preds)
    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average='macro'))
    macro_prec = float(precision_score(y_true, preds, average='macro'))
    macro_rec = float(recall_score(y_true, preds, average='macro'))

    duration = time.time() - t0
    print(f"  [+] Evaluated {len(y_true):,} cases in {duration:.2f}s:")
    print(f"      Accuracy:  {acc * 100:.2f}%")
    print(f"      Macro F1:  {macro_f1 * 100:.2f}%")
    print(f"      Precision: {macro_prec * 100:.2f}%")
    print(f"      Recall:    {macro_rec * 100:.2f}%")

    return {
        'dataset_path': csv_path,
        'evaluated_samples': len(y_true),
        'accuracy_pct': round(acc * 100, 2),
        'macro_f1_pct': round(macro_f1 * 100, 2),
        'macro_precision_pct': round(macro_prec * 100, 2),
        'macro_recall_pct': round(macro_rec * 100, 2),
        'evaluation_seconds': round(duration, 2),
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("  EEQ401: CROSS-DATASET GENERALIZATION EVALUATION")
    print("=" * 70)
    print(f"  [+] Device: {device}")

    # Load scaler
    scaler_path = "models/classifier_100k/scaler_params.json"
    with open(scaler_path, 'r') as f:
        sdata = json.load(f)
    scaler_mean = np.array(sdata['mean'], dtype=np.float32)
    scaler_scale = np.array(sdata['scale'], dtype=np.float32)

    # Load model
    model = GridSecurityClassifier(input_dim=79, hidden_dim=256, num_classes=3).to(device)
    model.load_state_dict(torch.load("models/classifier_100k/grid_security_classifier.pt", map_location=device))
    model.eval()

    results = {}
    if os.path.exists("data/tabular/dataset_200k/dataset_200k.csv"):
        results['dataset_200k'] = evaluate_on_dataset(
            "data/tabular/dataset_200k/dataset_200k.csv",
            model, scaler_mean, scaler_scale, device, sample_size=10000
        )
    if os.path.exists("data/tabular/dataset_300k/dataset_300k.csv"):
        results['dataset_300k'] = evaluate_on_dataset(
            "data/tabular/dataset_300k/dataset_300k.csv",
            model, scaler_mean, scaler_scale, device, sample_size=10000
        )

    out_path = "models/classifier_100k/cross_dataset_validation.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  [+] Saved cross-dataset validation results to {out_path}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
