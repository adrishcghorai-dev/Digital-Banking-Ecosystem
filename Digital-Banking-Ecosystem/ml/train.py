"""
Training Pipeline
=================
End-to-end script that:

1. Loads real data from behavior.json
2. Generates synthetic data to augment
3. Extracts features via the feature engineering pipeline
4. Trains all four models (BotDetector, AnomalyDetector, UserVerifier, RiskScorer)
5. Evaluates with train/test split, prints classification reports
6. Saves trained models + scalers to ml/saved_models/
"""

import os
import sys
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

# ── Add project root to path so we can import ml package ──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.feature_engineering import extract_features_batch, NUM_FEATURES, FEATURE_NAMES
from ml.synthetic_data import generate_dataset
from ml.models import BotDetector, AnomalyDetector, UserVerifier, RiskScorer

# ── Config ────────────────────────────────────────────────────────────────────

BEHAVIOR_FILE = os.path.join(_PROJECT_ROOT, "logs", "behavior.json")
N_HUMAN = 1000
N_BOT = 500
N_FRAUD = 500
N_USERS = 10
TEST_SIZE = 0.2
SEED = 42


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def load_real_data() -> list:
    """Load existing behavior.json records."""
    if not os.path.exists(BEHAVIOR_FILE):
        print("[INFO] No behavior.json found – using synthetic data only.")
        return []
    with open(BEHAVIOR_FILE, "r") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} real records from behavior.json")
    return data


def main():
    separator("STEP 1 — Data Preparation")

    # 1a. Load real data
    real_records = load_real_data()

    # 1b. Generate synthetic data
    print(f"[INFO] Generating synthetic data: {N_HUMAN} human, {N_BOT} bot, {N_FRAUD} fraud")
    syn_records, bot_labels, anomaly_labels = generate_dataset(
        n_human=N_HUMAN, n_bot=N_BOT, n_fraud=N_FRAUD,
        n_users=N_USERS, seed=SEED,
    )

    # 1c. Merge real data as human/normal (labels: bot=0, anomaly=0)
    for rec in real_records:
        rec["_label_bot"] = 0
        rec["_label_anomaly"] = 0
        rec["_label_user_id"] = rec.get("session_id", "real_user")[:20]

    all_records = real_records + syn_records
    all_bot_labels = np.concatenate([
        np.zeros(len(real_records), dtype=np.int32),
        bot_labels,
    ])
    all_anomaly_labels = np.concatenate([
        np.zeros(len(real_records), dtype=np.int32),
        anomaly_labels,
    ])
    all_user_ids = np.array(
        [r.get("_label_user_id", "unknown") for r in all_records]
    )

    print(f"[INFO] Total dataset: {len(all_records)} records")
    print(f"       Humans:   {int((all_bot_labels == 0).sum())}")
    print(f"       Bots:     {int((all_bot_labels == 1).sum())}")
    print(f"       Anomalies:{int((all_anomaly_labels == 1).sum())}")

    # ── Extract features ──────────────────────────────────────────────────
    separator("STEP 2 — Feature Extraction")
    X = extract_features_batch(all_records)
    print(f"[INFO] Feature matrix shape: {X.shape}  ({NUM_FEATURES} features)")
    print(f"[INFO] Any NaN: {np.isnan(X).any()}, Any Inf: {np.isinf(X).any()}")

    # Replace any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Train/Test split ──────────────────────────────────────────────────
    X_train, X_test, \
    y_bot_train, y_bot_test, \
    y_ano_train, y_ano_test, \
    uid_train, uid_test = train_test_split(
        X, all_bot_labels, all_anomaly_labels, all_user_ids,
        test_size=TEST_SIZE, random_state=SEED, stratify=all_bot_labels,
    )
    print(f"[INFO] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN: Bot Detector
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 3a — Bot Detector (RandomForest)")
    bot_det = BotDetector()
    bot_det.fit(X_train, y_bot_train)

    # Evaluate
    preds = [bot_det.predict(x.reshape(1, -1))["is_bot"] for x in X_test]
    preds_int = np.array([int(p) for p in preds])

    print("Classification Report:")
    print(classification_report(y_bot_test, preds_int,
                                target_names=["Human", "Bot"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_bot_test, preds_int))
    try:
        probas = np.array([bot_det.predict(x.reshape(1, -1))["confidence"]
                           for x in X_test])
        auc = roc_auc_score(y_bot_test, probas)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        pass

    print("\nTop 10 Feature Importances:")
    imp = bot_det.feature_importances()
    for i, (name, val) in enumerate(list(imp.items())[:10]):
        print(f"  {i+1:2d}. {name:30s} {val:.4f}")

    bot_det.save()
    print("[OK] Bot Detector saved.")

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN: Anomaly Detector
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 3b — Anomaly Detector (IsolationForest)")
    X_normal_train = X_train[y_ano_train == 0]
    print(f"[INFO] Training on {X_normal_train.shape[0]} normal samples")

    ano_det = AnomalyDetector()
    ano_det.fit(X_normal_train)

    # Evaluate on test set
    ano_preds = []
    for x in X_test:
        result = ano_det.predict(x.reshape(1, -1))
        ano_preds.append(1 if result["is_anomaly"] else 0)
    ano_preds = np.array(ano_preds)

    print("Classification Report:")
    print(classification_report(y_ano_test, ano_preds,
                                target_names=["Normal", "Anomaly"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_ano_test, ano_preds))

    ano_det.save()
    print("[OK] Anomaly Detector saved.")

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN: User Verifier
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 3c — User Verifier (One-Class SVM)")
    X_human_train = X_train[y_bot_train == 0]
    uid_human_train = uid_train[y_bot_train == 0]

    user_ver = UserVerifier()
    user_ver.fit(X_human_train, uid_human_train, min_samples=5)
    print(f"[INFO] Trained models for {len(user_ver.user_models)} users: "
          f"{list(user_ver.user_models.keys())}")

    # Evaluate: for each human test sample, verify against correct & wrong user
    correct = 0
    wrong = 0
    total = 0
    human_test_mask = y_bot_train == 0  # use only human labels from test
    X_human_test = X_test[y_bot_test == 0]
    uid_human_test = uid_test[y_bot_test == 0]

    for x, uid in zip(X_human_test, uid_human_test):
        if uid not in user_ver.user_models:
            continue
        result = user_ver.verify(x.reshape(1, -1), uid)
        if result["is_verified"]:
            correct += 1
        total += 1

        # Also test against a wrong user
        wrong_uid = [u for u in user_ver.user_models if u != uid]
        if wrong_uid:
            result_wrong = user_ver.verify(x.reshape(1, -1), wrong_uid[0])
            if not result_wrong["is_verified"]:
                wrong += 1

    if total > 0:
        print(f"[INFO] Correct user verified:  {correct}/{total} = {correct/total:.2%}")
        print(f"[INFO] Wrong user rejected:    {wrong}/{total} = {wrong/total:.2%}")
    else:
        print("[WARN] Not enough test data for user verification evaluation")

    user_ver.save()
    print("[OK] User Verifier saved.")

    # ══════════════════════════════════════════════════════════════════════
    # VERIFY: Risk Scorer (Ensemble)
    # ══════════════════════════════════════════════════════════════════════
    separator("STEP 4 — Risk Scorer (Ensemble Verification)")
    scorer = RiskScorer()
    scorer.load_models()

    # Score a few test samples
    print("Sample risk scores:")
    print(f"  {'Type':8s} {'Score':>5s} {'Level':10s} {'Bot':>6s} {'Anomaly':>8s}")
    print(f"  {'-'*45}")

    indices = np.random.RandomState(42).choice(len(X_test), size=min(10, len(X_test)), replace=False)
    for idx in indices:
        # Reconstruct record from all_records
        global_idx = len(X_train) + idx  # approximate
        rec = all_records[min(global_idx, len(all_records)-1)]
        uid = rec.get("_label_user_id", "unknown")
        result = scorer.score(rec, claimed_user_id=uid)
        label_type = "BOT" if all_bot_labels[min(global_idx, len(all_bot_labels)-1)] == 1 else "HUMAN"
        print(f"  {label_type:8s} {result['risk_score']:5d} {result['risk_level']:10s} "
              f"{result['breakdown']['bot_score']:6.3f} "
              f"{result['breakdown']['anomaly_score']:8.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    separator("TRAINING COMPLETE")
    print("Saved models:")
    model_dir = os.path.join(_HERE, "saved_models")
    if os.path.exists(model_dir):
        for f in sorted(os.listdir(model_dir)):
            try:
                size = os.path.getsize(os.path.join(model_dir, f))
                print(f"  {f:40s} {size/1024:.1f} KB")
            except OSError:
                print(f"  {f:40s} (file in use)")
    print("\nAll models trained and saved successfully!")
    print("Use the Flask API endpoint POST /api/ml/analyze to score live sessions.")


if __name__ == "__main__":
    main()
