"""
ML Models for Behavioral Intelligence
======================================
Four model classes:

1. **BotDetector**      – RandomForest binary classifier (bot vs human)
2. **AnomalyDetector**  – IsolationForest unsupervised anomaly detector
3. **UserVerifier**     – One-Class SVM per-user behavioral verifier
4. **RiskScorer**       – Weighted ensemble producing 0–100 risk score
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from .feature_engineering import extract_features, FEATURE_NAMES, NUM_FEATURES

# ── Paths ─────────────────────────────────────────────────────────────────────

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def _ensure_model_dir():
    os.makedirs(_MODEL_DIR, exist_ok=True)


def _model_path(name: str) -> str:
    return os.path.join(_MODEL_DIR, name)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Bot Detector
# ══════════════════════════════════════════════════════════════════════════════

class BotDetector:
    """RandomForest classifier: 0 = human, 1 = bot."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> dict:
        if not self._fitted:
            raise RuntimeError("BotDetector not trained – call fit() or load()")
        X = np.atleast_2d(X)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        results = []
        for p in proba:
            bot_prob = float(p[1]) if len(p) > 1 else float(p[0])
            results.append({
                "is_bot": bool(bot_prob >= 0.5),
                "confidence": round(bot_prob, 4),
            })
        return results if len(results) > 1 else results[0]

    def feature_importances(self) -> dict:
        if not self._fitted:
            return {}
        imp = self.model.feature_importances_
        pairs = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
        return {name: round(float(v), 4) for name, v in pairs}

    def save(self, prefix="bot_detector"):
        _ensure_model_dir()
        joblib.dump(self.model, _model_path(f"{prefix}.pkl"))
        joblib.dump(self.scaler, _model_path(f"{prefix}_scaler.pkl"))

    def load(self, prefix="bot_detector"):
        self.model = joblib.load(_model_path(f"{prefix}.pkl"))
        self.scaler = joblib.load(_model_path(f"{prefix}_scaler.pkl"))
        self._fitted = True
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 2. Anomaly Detector
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """IsolationForest trained on normal data; anomalies score < 0."""

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X_normal: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_normal)
        self.model.fit(X_scaled)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> dict:
        if not self._fitted:
            raise RuntimeError("AnomalyDetector not trained")
        X = np.atleast_2d(X)
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        preds = self.model.predict(X_scaled)
        results = []
        for score, pred in zip(scores, preds):
            results.append({
                "is_anomaly": bool(pred == -1),
                "anomaly_score": round(float(score), 4),
            })
        return results if len(results) > 1 else results[0]

    def save(self, prefix="anomaly_detector"):
        _ensure_model_dir()
        joblib.dump(self.model, _model_path(f"{prefix}.pkl"))
        joblib.dump(self.scaler, _model_path(f"{prefix}_scaler.pkl"))

    def load(self, prefix="anomaly_detector"):
        self.model = joblib.load(_model_path(f"{prefix}.pkl"))
        self.scaler = joblib.load(_model_path(f"{prefix}_scaler.pkl"))
        self._fitted = True
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 3. User Verifier
# ══════════════════════════════════════════════════════════════════════════════

class UserVerifier:
    """One-Class SVM per user for behavioral biometric verification."""

    def __init__(self):
        self.user_models = {}
        self.user_scalers = {}
        self.global_scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, user_ids: np.ndarray, min_samples: int = 5):
        self.global_scaler.fit(X)
        unique_users = set(user_ids)

        for uid in unique_users:
            mask = user_ids == uid
            X_user = X[mask]
            if len(X_user) < min_samples:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_user)

            svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
            svm.fit(X_scaled)

            self.user_models[uid] = svm
            self.user_scalers[uid] = scaler

        self._fitted = True
        return self

    def verify(self, X: np.ndarray, claimed_user_id: str) -> dict:
        if not self._fitted:
            raise RuntimeError("UserVerifier not trained")

        X = np.atleast_2d(X)

        if claimed_user_id not in self.user_models:
            return {
                "is_verified": False,
                "similarity_score": 0.0,
                "reason": f"No model for user '{claimed_user_id}'"
            }

        scaler = self.user_scalers[claimed_user_id]
        svm = self.user_models[claimed_user_id]

        X_scaled = scaler.transform(X)
        score = float(svm.decision_function(X_scaled)[0])
        pred = int(svm.predict(X_scaled)[0])

        similarity = 1.0 / (1.0 + np.exp(-score))

        return {
            "is_verified": bool(pred == 1),
            "similarity_score": round(float(similarity), 4),
        }

    def save(self, prefix="user_verifier"):
        _ensure_model_dir()
        joblib.dump({
            "user_models": self.user_models,
            "user_scalers": self.user_scalers,
            "global_scaler": self.global_scaler,
        }, _model_path(f"{prefix}.pkl"))

    def load(self, prefix="user_verifier"):
        data = joblib.load(_model_path(f"{prefix}.pkl"))
        self.user_models = data["user_models"]
        self.user_scalers = data["user_scalers"]
        self.global_scaler = data["global_scaler"]
        self._fitted = True
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 4. Risk Scorer (Ensemble)
# ══════════════════════════════════════════════════════════════════════════════

class RiskScorer:
    """
    Combines BotDetector, AnomalyDetector, UserVerifier, and rule-based
    flags into a single 0–100 risk score.

    Score breakdown:
        40% bot probability
        30% anomaly score
        20% identity mismatch
        10% rule-based flags
    """

    @staticmethod
    def _rule_flags(record: dict) -> float:
        flags = 0
        total = 6

        if (record.get("session_duration_ms") or 0) < 2000:
            flags += 1

        if (record.get("mouse_sample_count") or 0) == 0:
            flags += 1

        page = record.get("page", "")
        if page in ("/login", "/register") and (record.get("keystroke_count") or 0) == 0:
            flags += 1

        if (record.get("typing_speed_cps") or 0) > 25:
            flags += 1

        dur = record.get("session_duration_ms") or 0
        if dur > 10000 and (record.get("idle_ratio") or 0) < 0.01:
            flags += 1

        ms_mean = record.get("mouse_speed_mean") or 0
        ms_std = record.get("mouse_speed_std") or 0
        if ms_mean > 500 and ms_std < 50:
            flags += 1

        return flags / total

    def __init__(self):
        self.bot_detector = BotDetector()
        self.anomaly_detector = AnomalyDetector()
        self.user_verifier = UserVerifier()
        self._loaded = False

    def load_models(self):
        try:
            self.bot_detector.load()
            self.anomaly_detector.load()
            self.user_verifier.load()
            self._loaded = True
        except FileNotFoundError as e:
            print(f"[RiskScorer] Could not load all models: {e}")
            self._loaded = False
        return self

    def score(self, record: dict, claimed_user_id: str = None) -> dict:
        if not self._loaded:
            self.load_models()

        features = extract_features(record)
        features_2d = features.reshape(1, -1)

        bot_result = self.bot_detector.predict(features_2d)
        bot_score = bot_result["confidence"]

        ano_result = self.anomaly_detector.predict(features_2d)
        raw_ano = ano_result["anomaly_score"]
        anomaly_score = 1.0 / (1.0 + np.exp(raw_ano * 5))

        if claimed_user_id and self.user_verifier._fitted:
            id_result = self.user_verifier.verify(features_2d, claimed_user_id)
            identity_mismatch = 1.0 - id_result["similarity_score"]
        else:
            id_result = {"is_verified": True, "similarity_score": 1.0,
                         "reason": "No user claim provided"}
            identity_mismatch = 0.0

        rule_score = self._rule_flags(record)

        composite = (
            0.40 * bot_score +
            0.30 * anomaly_score +
            0.20 * identity_mismatch +
            0.10 * rule_score
        )
        risk_score = int(round(min(max(composite * 100, 0), 100)))

        if risk_score >= 75:
            level = "critical"
        elif risk_score >= 50:
            level = "high"
        elif risk_score >= 25:
            level = "medium"
        else:
            level = "low"

        return {
            "risk_score": risk_score,
            "risk_level": level,
            "breakdown": {
                "bot_score": round(bot_score, 4),
                "anomaly_score": round(float(anomaly_score), 4),
                "identity_mismatch": round(identity_mismatch, 4),
                "rule_flags": round(rule_score, 4),
            },
            "details": {
                "bot_detection": bot_result,
                "anomaly_detection": ano_result,
                "identity_verification": id_result,
            }
        }
