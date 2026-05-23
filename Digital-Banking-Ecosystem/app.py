import os
import csv
import json
import hashlib
import datetime
import urllib.request
import urllib.error
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
GEMINI_API_KEY = os.environ.get("AIzaSyDUrmton9H_fm3jfA0rkvSA94iWdM3agN0")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-1.5-mini")
GEMINI_URL     = f"https://gemini.googleapis.com/v1/models/{GEMINI_MODEL}:generateText"


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def mask_password(password: str) -> str:
    if not password:
        return ""
    visible_length = min(10, len(password))
    return "*" * visible_length + ("..." if len(password) > visible_length else "")


def parse_iso_timestamp(value: str):
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None


def is_recent(entry: dict, keys: list[str], cutoff: datetime.datetime) -> bool:
    for key in keys:
        ts = parse_iso_timestamp(entry.get(key, ""))
        if ts is not None:
            return ts >= cutoff
    return True


def prune_old_data(retention_hours: int = 24) -> tuple[list[dict], list[dict]]:
    recent_cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=retention_hours)
    logs = read_logs()
    behavior = read_behavior()

    filtered_logs = [row for row in logs if is_recent(row, ["timestamp"], recent_cutoff)]
    filtered_behavior = [item for item in behavior if is_recent(item, ["server_timestamp"], recent_cutoff)]

    if len(filtered_logs) != len(logs):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CRED_HEADERS)
            writer.writerows([[row.get(h, "") for h in CRED_HEADERS] for row in filtered_logs])

    if len(filtered_behavior) != len(behavior):
        save_behavior(filtered_behavior)

    return filtered_logs, filtered_behavior


def ensure_dirs():
    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)


# ── Kicked sessions store ─────────────────────────────────────────────────────

def load_kicked() -> set:
    ensure_dirs()
    if not os.path.exists(KICKED_FILE):
        return set()
    with open(KICKED_FILE, "r") as f:
        return set(json.load(f))

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
    with open(USERS_FILE, "r") as f:
        return json.load(f)

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
    with open(LOG_FILE, "r", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]

def _gemini_response_text(response_json: dict) -> str:
    if not response_json:
        return ""
    if isinstance(response_json, dict) and response_json.get("candidates"):
        return "\n".join(str(c.get("output", "")) for c in response_json.get("candidates", []) if c.get("output"))
    if isinstance(response_json, dict) and response_json.get("output"):
        output = response_json.get("output")
        if isinstance(output, dict):
            if isinstance(output.get("content"), list):
                return "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in output.get("content", [])
                )
            return str(output.get("content", ""))
        return str(output)
    return json.dumps(response_json)


def _call_gemini(prompt: str, timeout: int = 15) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    body = json.dumps({
        "prompt": {"text": prompt},
        "temperature": 0.0,
        "max_output_tokens": 800,
    }).encode("utf-8")
    request_obj = urllib.request.Request(
        GEMINI_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}",
        },
        method="POST"
    )
    with urllib.request.urlopen(request_obj, timeout=timeout) as response:
        response_json = json.load(response)
    return _gemini_response_text(response_json)


def _parse_gemini_json_array(text: str) -> list:
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return []


def _find_credential_anomalies_fallback(logs: list) -> list:
    anomalies = []
    failed_by_ip = {}
    failed_by_user = {}
    for idx, row in enumerate(logs, start=1):
        ip = row.get("ip_address", "unknown")
        user = row.get("username", "unknown")
        result = (row.get("result") or "").lower()
        user_agent = (row.get("user_agent") or "").lower()

        if user == "[OTP-STEP]":
            anomalies.append({
                "row_index": idx,
                "username": user,
                "reason": "OTP intercept step captured in credentials log.",
            })
            continue

        if result == "failed":
            failed_by_ip[ip] = failed_by_ip.get(ip, 0) + 1
            failed_by_user[user] = failed_by_user.get(user, 0) + 1
            if failed_by_ip[ip] >= 4:
                anomalies.append({
                    "row_index": idx,
                    "username": user,
                    "reason": f"Multiple failed login attempts from the same IP ({ip}).",
                })
                continue
            if failed_by_user[user] >= 4:
                anomalies.append({
                    "row_index": idx,
                    "username": user,
                    "reason": f"Repeated failed attempts for user '{user}'.",
                })
                continue

        if any(needle in user_agent for needle in ["bot", "curl", "python", "scrapy", "selenium", "wget"]):
            anomalies.append({
                "row_index": idx,
                "username": user,
                "reason": "Suspicious user agent suggests automated login activity.",
            })
    return anomalies


def analyze_credential_logs_with_gemini(logs: list) -> list:
    if not logs:
        return []
    if not GEMINI_API_KEY:
        return _find_credential_anomalies_fallback(logs)

    sample = logs[:25]
    prompt_lines = [
        "You are a security analyst reviewing captured login attempts.",
        "Identify anomalous or suspicious records and return only a JSON array of objects.",
        "Each object should include row_index, username, result, ip_address, and a short reason.",
        "Use row_index values matching the display order in the dashboard (1 = first shown row).",
        "Return [] if there are no suspicious records.",
        "",
        "Records:",
    ]
    for idx, row in enumerate(sample, start=1):
        prompt_lines.append(
            f"{idx}. timestamp={row.get('timestamp','')}, ip_address={row.get('ip_address','')}, username={row.get('username','')}, result={row.get('result','')}, account_type={row.get('account_type','')}, user_agent={row.get('user_agent','')}"
        )
    prompt_lines.append("\nJSON:")
    prompt = "\n".join(prompt_lines)

    try:
        response_text = _call_gemini(prompt)
        parsed = _parse_gemini_json_array(response_text)
        anomalies = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            row_index = item.get("row_index")
            try:
                row_index = int(row_index)
            except Exception:
                continue
            anomalies.append({
                "row_index": row_index,
                "username": str(item.get("username", "")),
                "reason": str(item.get("reason", "")).strip(),
            })
        if anomalies:
            return anomalies
    except Exception:
        pass

    return _find_credential_anomalies_fallback(logs)

# ── Behavioral log (behavior.json) ────────────────────────────────────────────

def ensure_behavior_file():
    ensure_dirs()
    if not os.path.exists(BEHAVIOR_FILE):
        with open(BEHAVIOR_FILE, "w") as f:
            json.dump([], f)

def read_behavior():
    ensure_behavior_file()
    with open(BEHAVIOR_FILE, "r") as f:
        return json.load(f)


def save_behavior(records: list):
    ensure_behavior_file()
    with open(BEHAVIOR_FILE, "w") as f:
        json.dump(records, f, indent=2)


def merge_behavior_record(records: list, record: dict):
    session_id = record.get("session_id")
    if session_id:
        for idx, existing in enumerate(records):
            if existing.get("session_id") == session_id:
                records[idx] = record
                return
    records.append(record)


def append_behavior(record: dict):
    ensure_behavior_file()
    records = read_behavior()
    merge_behavior_record(records, record)
    save_behavior(records)


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
        payload["session_id"]  = payload.get("session_id") or request.cookies.get("session", "none")
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
        item_sids = [(item.get("session_id") or sid) for item in items]
        if any(s in kicked_set for s in item_sids):
            return jsonify({"status": "ok", "received": 0, "kicked": True})
        records = read_behavior()
        ml_result = None
        auto_kicked = False
        for item in items:
            item["server_timestamp"] = datetime.datetime.utcnow().isoformat()
            item["ip_address"] = ip
            item["user_agent"] = ua
            item["session_id"] = item.get("session_id") or sid
            merge_behavior_record(records, item)
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
                    print(f"\n[ML] Session {item['session_id'][:8]}… → Risk: {result['risk_score']} ({result['risk_level'].upper()}) | Bot: {result['details']['bot_detection']['is_bot']}")
                    # ── Auto-kick on high or critical risk ─────────────────
                    if result["risk_score"] >= 50 or str(result.get("risk_level", "")).lower() in ("high", "critical"):
                        kicked_set.add(item["session_id"])
                        save_kicked(kicked_set)
                        auto_kicked = True
                        print(f"[ML] ⚠ AUTO-KICKED session {item['session_id'][:8]}… (risk={result['risk_score']})")
            except Exception as e:
                print(f"[ML] Scoring error: {e}")
        save_behavior(records)
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
    sid = request.args.get("session_id") or request.cookies.get("session", "none")
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

    logs, behavior = prune_old_data(retention_hours=24)
    logs = list(reversed(logs))
    behavior = list(reversed(behavior))
    users = load_users()

    for row in logs:
        row["password"] = mask_password(row.get("password", ""))

    credential_anomalies = analyze_credential_logs_with_gemini(logs)
    anomaly_map = {item["row_index"]: item["reason"] for item in credential_anomalies}
    for idx, row in enumerate(logs, start=1):
        if idx in anomaly_map:
            row["_anomaly_reason"] = anomaly_map[idx]

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

    return render_template(
        "dashboard.html",
        logs=logs,
        stats=stats,
        behavior=behavior,
        users=users,
        credential_anomalies=credential_anomalies,
        gemini_available=bool(GEMINI_API_KEY),
    )


@app.route("/api/dashboard/status")
def dashboard_status():
    if not session.get("admin"):
        return jsonify({"status": "unauthorized"}), 403
    logs, behavior = prune_old_data(retention_hours=24)
    return jsonify({
        "status": "ok",
        "log_count": len(logs),
        "behavior_count": len(behavior),
        "kicked_count": len(load_kicked()),
    })


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


if __name__ == "__main__":
    ensure_log_file()
    ensure_behavior_file()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
