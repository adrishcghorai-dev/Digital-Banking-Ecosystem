"""
ML API Blueprint
================
Flask Blueprint that exposes the trained ML models via REST endpoints.
"""

import os
import json
import traceback
import subprocess
import sys

from flask import Blueprint, request, jsonify, session

ml_bp = Blueprint("ml_api", __name__)

_risk_scorer = None
_models_loaded = False


def _get_scorer():
    global _risk_scorer, _models_loaded
    if _risk_scorer is None:
        from ml.models import RiskScorer
        _risk_scorer = RiskScorer()
        try:
            _risk_scorer.load_models()
            _models_loaded = True
        except Exception as e:
            print(f"[ML API] Failed to load models: {e}")
            _models_loaded = False
    return _risk_scorer


@ml_bp.route("/api/ml/analyze", methods=["POST"])
def analyze():
    try:
        record = request.get_json(force=True, silent=True)
        if not record:
            return jsonify({"error": "No JSON body provided"}), 400

        claimed_user = request.args.get("user_id", None)
        scorer = _get_scorer()

        if not _models_loaded:
            return jsonify({
                "error": "Models not trained yet.",
                "status": "models_not_found",
            }), 503

        result = scorer.score(record, claimed_user_id=claimed_user)

        return jsonify({
            "bot_detection": result["details"]["bot_detection"],
            "anomaly_detection": result["details"]["anomaly_detection"],
            "identity_verification": result["details"]["identity_verification"],
            "risk_score": {
                "score": result["risk_score"],
                "level": result["risk_level"],
                "breakdown": result["breakdown"],
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ml_bp.route("/api/ml/analyze/batch", methods=["POST"])
def analyze_batch():
    try:
        records = request.get_json(force=True, silent=True)
        if not records or not isinstance(records, list):
            return jsonify({"error": "Expected a JSON array"}), 400

        claimed_user = request.args.get("user_id", None)
        scorer = _get_scorer()

        if not _models_loaded:
            return jsonify({"error": "Models not trained yet."}), 503

        results = []
        for rec in records:
            result = scorer.score(rec, claimed_user_id=claimed_user)
            results.append({
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "is_bot": result["details"]["bot_detection"]["is_bot"],
                "is_anomaly": result["details"]["anomaly_detection"]["is_anomaly"],
            })

        return jsonify({"results": results, "count": len(results)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ml_bp.route("/api/ml/retrain", methods=["POST"])
def retrain():
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized – admin access required"}), 403

    try:
        train_script = os.path.join(os.path.dirname(__file__), "train.py")
        result = subprocess.run(
            [sys.executable, train_script],
            capture_output=True, text=True, timeout=120,
        )

        global _risk_scorer, _models_loaded
        _risk_scorer = None
        _models_loaded = False

        return jsonify({
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Training timed out (120s limit)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_bp.route("/api/ml/status", methods=["GET"])
def status():
    scorer = _get_scorer()
    model_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    saved_files = []
    if os.path.exists(model_dir):
        saved_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

    from ml.feature_engineering import FEATURE_NAMES, NUM_FEATURES

    return jsonify({
        "models_loaded": _models_loaded,
        "num_features": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "saved_model_files": saved_files,
        "user_verifier_users": (
            list(scorer.user_verifier.user_models.keys())
            if _models_loaded else []
        ),
    })
