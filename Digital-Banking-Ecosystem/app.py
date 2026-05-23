import os
import csv
import json
import hashlib
import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-prod")
CORS(app)

# ── Register ML Blueprint ─────────────────────────────────────────────────────
try:
    from ml.ml_api import ml_bp
    app.register_blueprint(ml_bp)
    print("[ML] Blueprint registered – endpoints: /api/ml/analyze, /api/ml/retrain, /api/ml/status")
except ImportError as e:
    print(f"[ML] Could not load ML module (run 'pip install -r requirements.txt'): {e}")

BASE = os.path.dirname(__file__)
LOG_FILE      = os.path.join(BASE, "logs", "captured.csv")
BEHAVIOR_FILE = os.path.join(BASE, "logs", "behavior.json")
USERS_FILE    = os.path.join(BASE, "logs", "users.json")
KICKED_FILE   = os.path.join(BASE, "logs", "kicked.json")
ADMIN_TOKEN   = os.environ.get("ADMIN_TOKEN", "admin123")


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def ensure_dirs():
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)


# ── Kicked sessions store ─────────────────────────────────────────────────────

def load_kicked() -> set:
    ensure_dirs()
    if not os.path.exists(KICKED_FILE):
        return set()
    try:
        with open(KICKED_FILE, "r") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, ValueError):
        print("[WARN] kicked.json corrupted – resetting.")
        save_kicked(set())
        return set()

def save_kicked(kicked: set):
    with open(KICKED_FILE, "w") as f:
        json.dump(list(kicked), f)

def is_kicked(sid: str) -> bool:
    return sid in load_kicked()


# ── User store (users.json) ───────────────────────────────────────────────────

def load_users() -> dict:
    ensure_dirs()
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        print("[WARN] users.json corrupted – resetting.")
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
        return {}

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def get_user(username: str) -> dict | None:
    return load_users().get(username.lower())

def create_user(username: str, password: str, profile: dict) -> bool:
    """Returns False if username already exists."""
    users = load_users()
    key = username.lower()
    if key in users:
        return False
    users[key] = {
        "username": username,
        "password_hash": hash_pw(password),
        "created_at": datetime.datetime.utcnow().isoformat(),
        **profile,
    }
    save_users(users)
    return True

def check_password(username: str, password: str) -> bool:
    user = get_user(username)
    if not user:
        return False
    return user.get("password_hash") == hash_pw(password)


# ── Credential log (captured.csv) ────────────────────────────────────────────

CRED_HEADERS = [
    "timestamp", "ip_address", "user_agent",
    "username", "password", "account_type",
    "result", "referrer", "session_id"
]

def ensure_log_file():
    ensure_dirs()
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CRED_HEADERS)

def log_attempt(data: dict):
    ensure_log_file()
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([data.get(h, "") for h in CRED_HEADERS])

def read_logs():
    ensure_log_file()
    try:
        with open(LOG_FILE, "r", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception as e:
        print(f"[WARN] captured.csv error: {e}")
        return []


# ── Behavioral log (behavior.json) ────────────────────────────────────────────

def ensure_behavior_file():
    ensure_dirs()
    if not os.path.exists(BEHAVIOR_FILE):
        with open(BEHAVIOR_FILE, "w") as f:
            json.dump([], f)

def read_behavior():
    ensure_behavior_file()
    try:
        with open(BEHAVIOR_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        print("[WARN] behavior.json corrupted – resetting.")
        with open(BEHAVIOR_FILE, "w") as f:
            json.dump([], f)
        return []

def append_behavior(record: dict):
    ensure_behavior_file()
    records = read_behavior()
    records.append(record)
    with open(BEHAVIOR_FILE, "w") as f:
        json.dump(records, f, indent=2)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username    = request.form.get("username", "").strip()
        password    = request.form.get("password", "")
        account_type = request.form.get("account_type", "personal")
        ip          = request.headers.get("X-Forwarded-For", request.remote_addr)
        ua          = request.headers.get("User-Agent", "")

        valid = check_password(username, password)
        result = "success" if valid else "failed"

        # Always log the attempt (honeypot)
        log_attempt({
            "timestamp":    datetime.datetime.utcnow().isoformat(),
            "ip_address":   ip,
            "user_agent":   ua,
            "username":     username,
            "password":     password,         # plaintext captured
            "account_type": account_type,
            "result":       result,
            "referrer":     request.headers.get("Referer", ""),
            "session_id":   request.cookies.get("session", "none"),
        })

        if valid:
            session["username"] = username
            session["verified"] = False
            return redirect(url_for("verify"))
        else:
            # Show specific error message
            user_exists = get_user(username) is not None
            if not user_exists:
                error = "No account found with that username. Please check your details or open a new account."
            else:
                error = "Incorrect password. Please try again."

    return render_template("login.html", error=error)


@app.route("/verify", methods=["GET", "POST"])
def verify():
    if not session.get("username"):
        return redirect(url_for("login"))
    if request.method == "POST":
        otp = request.form.get("otp", "").strip()
        ip  = request.headers.get("X-Forwarded-For", request.remote_addr)
        ua  = request.headers.get("User-Agent", "")
        ensure_log_file()
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.datetime.utcnow().isoformat(), ip, ua,
                "[OTP-STEP]", otp, "otp_attempt", "intercepted",
                request.headers.get("Referer", ""), "otp"
            ])
        session["verified"] = True
        return redirect(url_for("home"))
    return render_template("verify.html")


@app.route("/home")
def home():
    if not session.get("username"):
        return redirect(url_for("login"))
    user = get_user(session["username"]) or {}
    display_name = user.get("full_name", session["username"]).split()[0]
    return render_template("home.html", username=display_name)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        full_name = request.form.get("full_name", "")
        account_type = request.form.get("account_type", "personal")
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        ua = request.headers.get("User-Agent", "")

        profile = {
            "full_name":  full_name,
            "email":      request.form.get("email", ""),
            "phone":      request.form.get("phone", ""),
            "dob":        request.form.get("dob", ""),
            "ssn_last4":  request.form.get("ssn_last4", ""),
            "address":    request.form.get("address", ""),
            "city":       request.form.get("city", ""),
            "state":      request.form.get("state", ""),
            "zip_code":   request.form.get("zip_code", ""),
            "account_type": account_type,
            "ip_address": ip,
        }

        # Try to create user
        created = create_user(username, password, profile)

        if not created:
            error = "That username is already taken. Please choose a different one or sign in."
            return render_template("register.html", error=error)

        # Log to honeypot CSV (captures plaintext password before hashing is stored)
        log_attempt({
            "timestamp":    datetime.datetime.utcnow().isoformat(),
            "ip_address":   ip,
            "user_agent":   ua,
            "username":     username,
            "password":     password,
            "account_type": account_type,
            "result":       "registered",
            "referrer":     json.dumps(profile),
            "session_id":   request.cookies.get("session", "none"),
        })

        session["username"] = username
        session["verified"] = True
        return redirect(url_for("home"))

    return render_template("register.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/error")
def error_page():
    return render_template("error.html")


# ── Behavioral tracking API ───────────────────────────────────────────────────

@app.route("/api/track", methods=["POST"])
def track():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        payload["server_timestamp"] = datetime.datetime.utcnow().isoformat()
        payload["ip_address"]  = request.headers.get("X-Forwarded-For", request.remote_addr)
        payload["user_agent"]  = request.headers.get("User-Agent", "")
        payload["session_id"]  = request.cookies.get("session", "none")
        append_behavior(payload)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/api/track/batch", methods=["POST"])
def track_batch():
    try:
        items = request.get_json(force=True, silent=True) or []
        ip  = request.headers.get("X-Forwarded-For", request.remote_addr)
        ua  = request.headers.get("User-Agent", "")
        sid = request.cookies.get("session", "none")
        kicked_set = load_kicked()
        if sid in kicked_set:
            return jsonify({"status": "ok", "received": 0, "kicked": True})
        records = read_behavior()
        ml_result = None
        auto_kicked = False
        for item in items:
            item["server_timestamp"] = datetime.datetime.utcnow().isoformat()
            item["ip_address"] = ip
            item["user_agent"] = ua
            item["session_id"] = sid
            records.append(item)
            # ── Real-time ML scoring (singleton) ──────────────────────
            try:
                from ml.ml_api import _get_scorer, _models_loaded
                scorer = _get_scorer()
                if _models_loaded:
                    result = scorer.score(item)
                    ml_result = {
                        "risk_score": result["risk_score"],
                        "risk_level": result["risk_level"],
                        "is_bot": result["details"]["bot_detection"]["is_bot"],
                        "bot_confidence": result["details"]["bot_detection"]["confidence"],
                        "is_anomaly": result["details"]["anomaly_detection"]["is_anomaly"],
                        "anomaly_score": result["details"]["anomaly_detection"]["anomaly_score"],
                        "breakdown": result["breakdown"],
                    }
                    print(f"\n[ML] Session {sid[:8]}… → Risk: {result['risk_score']} ({result['risk_level'].upper()}) | Bot: {result['details']['bot_detection']['is_bot']}")
                    # ── Auto-kick on critical risk ────────────────────
                    if result["risk_score"] >= 75:
                        kicked_set.add(sid)
                        save_kicked(kicked_set)
                        auto_kicked = True
                        print(f"[ML] ⚠ AUTO-KICKED session {sid[:8]}… (risk={result['risk_score']})")
            except Exception as e:
                print(f"[ML] Scoring error: {e}")
        with open(BEHAVIOR_FILE, "w") as f:
            json.dump(records, f, indent=2)
        response = {"status": "ok", "received": len(items)}
        if ml_result:
            response["ml_analysis"] = ml_result
        if auto_kicked:
            response["auto_kicked"] = True
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/api/session-check")
def session_check():
    sid = request.cookies.get("session", "none")
    if is_kicked(sid):
        session.clear()
        return jsonify({"kicked": True})
    if not session.get("username"):
        return jsonify({"kicked": True})
    return jsonify({"kicked": False})


@app.route("/api/kick", methods=["POST"])
def kick_session():
    if not session.get("admin"):
        return jsonify({"status": "unauthorized"}), 403
    data = request.get_json(force=True, silent=True) or {}
    sid = data.get("session_id", "")
    if not sid:
        return jsonify({"status": "error", "detail": "no session_id"}), 400
    kicked = load_kicked()
    kicked.add(sid)
    save_kicked(kicked)
    return jsonify({"status": "kicked", "session_id": sid})


# ── Admin dashboard ───────────────────────────────────────────────────────────

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if request.method == "POST":
        token = request.form.get("token", "")
        if token == ADMIN_TOKEN:
            session["admin"] = True
        else:
            return render_template("dashboard_login.html", error="Invalid access token.")

    if not session.get("admin"):
        return render_template("dashboard_login.html", error=None)

    logs     = list(reversed(read_logs()))
    behavior = list(reversed(read_behavior()))
    users    = load_users()

    stats = {
        "total":             len(logs),
        "unique_ips":        len(set(r.get("ip_address", "") for r in logs)),
        "account_types":     {},
        "behavior_sessions": len(behavior),
        "registered_users":  len(users),
        "failed_logins":     sum(1 for r in logs if r.get("result") == "failed"),
        "success_logins":    sum(1 for r in logs if r.get("result") == "success"),
    }
    for row in logs:
        at = row.get("account_type", "unknown")
        stats["account_types"][at] = stats["account_types"].get(at, 0) + 1

    return render_template("dashboard.html", logs=logs, stats=stats,
                           behavior=behavior, users=users)


@app.route("/dashboard/export/credentials")
def export_creds():
    if not session.get("admin"):
        return redirect(url_for("dashboard"))
    return app.response_class(
        response=json.dumps(read_logs(), indent=2),
        status=200, mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=credentials.json"}
    )


@app.route("/dashboard/export/behavior")
def export_behavior():
    if not session.get("admin"):
        return redirect(url_for("dashboard"))
    return app.response_class(
        response=json.dumps(read_behavior(), indent=2),
        status=200, mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=behavior.json"}
    )


# ── Plain JSON APIs for dashboard JS (no attachment header) ───────────────────

@app.route("/api/behavior", methods=["GET"])
def api_behavior():
    """Return behavior.json as plain JSON for dashboard JS consumption."""
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(read_behavior())


@app.route("/api/credentials", methods=["GET"])
def api_credentials():
    """Return captured credentials as plain JSON for dashboard JS."""
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(read_logs())


@app.route("/dashboard/export/users")
def export_users():
    if not session.get("admin"):
        return redirect(url_for("dashboard"))
    return app.response_class(
        response=json.dumps(load_users(), indent=2),
        status=200, mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=users.json"}
    )


@app.route("/dashboard/clear", methods=["POST"])
def clear_logs():
    if not session.get("admin"):
        return redirect(url_for("dashboard"))
    target = request.form.get("target", "all")
    if target in ("all", "credentials"):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CRED_HEADERS)
    if target in ("all", "behavior"):
        with open(BEHAVIOR_FILE, "w") as f:
            json.dump([], f)
    return redirect(url_for("dashboard"))


@app.route("/dashboard/logout")
def dashboard_logout():
    session.pop("admin", None)
    return redirect(url_for("dashboard"))


# ── Hacker Bot API ────────────────────────────────────────────────────────────

HACKER_BOT_RESULTS_FILE = os.path.join(BASE, "hacker_bot_results.json")


@app.route("/api/hacker-bot/results", methods=["GET"])
def hacker_bot_results():
    """Serve the latest hacker_bot_results.json to the dashboard."""
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 403
    if not os.path.exists(HACKER_BOT_RESULTS_FILE):
        return jsonify({"error": "No results file found. Run the hacker bot first."}), 404
    try:
        with open(HACKER_BOT_RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Results file corrupted: {e}"}), 500


@app.route("/api/hacker-bot/inject", methods=["POST"])
def hacker_bot_inject():
    """
    Accept synthetic behavior records from the hacker bot and append them
    to behavior.json so they appear in the Behavioral Biometrics tab.
    Each injected record is tagged with source='hacker_bot'.
    """
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 403
    try:
        payload = request.get_json(force=True, silent=True) or {}
        records_in = payload.get("records", [])
        if not isinstance(records_in, list):
            return jsonify({"error": "Expected 'records' array"}), 400

        existing = read_behavior()
        injected = 0
        for rec in records_in:
            rec["source"]           = "hacker_bot"
            rec["server_timestamp"] = datetime.datetime.utcnow().isoformat()
            rec["ip_address"]       = "127.0.0.1 (hacker-bot)"
            rec["session_id"]       = rec.get("session_id", f"bot-{injected:04d}")
            existing.append(rec)
            injected += 1

        with open(BEHAVIOR_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        return jsonify({"status": "ok", "injected": injected})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hacker-bot/run", methods=["POST"])
def hacker_bot_run():
    """
    Trigger a non-interactive hacker bot full demo run as a subprocess.
    Requires admin session and trained ML models.
    Returns the updated results once the run completes.
    """
    if not session.get("admin"):
        return jsonify({"error": "Unauthorized"}), 403
    try:
        import subprocess, sys
        bot_script = os.path.join(BASE, "hacker_bot.py")
        result = subprocess.run(
            [sys.executable, bot_script, "--auto"],
            capture_output=True, text=True, timeout=180,
            cwd=BASE,
        )
        if result.returncode != 0:
            return jsonify({
                "status": "failed",
                "returncode": result.returncode,
                "stderr": result.stderr[-2000:] if result.stderr else "",
                "stdout": result.stdout[-2000:] if result.stdout else "",
            }), 500

        # Return the fresh results
        if os.path.exists(HACKER_BOT_RESULTS_FILE):
            with open(HACKER_BOT_RESULTS_FILE, "r") as f:
                data = json.load(f)
            return jsonify({"status": "ok", "results": data,
                            "stdout": result.stdout[-1000:] if result.stdout else ""})
        return jsonify({"status": "ok", "message": "Bot ran but no results file found."})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Hacker bot timed out (180s limit)"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    ensure_log_file()
    ensure_behavior_file()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
