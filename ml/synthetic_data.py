"""
Synthetic Data Generator
========================
Generates realistic labeled behavioral data for training ML models when
real data is scarce.  Produces three profiles:

- **human**  – natural, variable behaviour
- **bot**    – unnaturally uniform / fast / robotic
- **fraud**  – erratic, suspiciously fast, atypical patterns

Each sample is a dict mimicking one behavior_snapshot from behavior.json.
"""

import random
import math
import numpy as np
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(42)


def _clamp(v, lo=0.0, hi=1e9):
    return max(lo, min(v, hi))


def _rand_normal(mu, sigma, lo=0.0, hi=1e9):
    return _clamp(_RNG.normal(mu, sigma), lo, hi)


def _rand_uniform(lo, hi):
    return _RNG.uniform(lo, hi)


def _make_heatmap(n_cells: int, grid_cols=32, grid_rows=18) -> list:
    """Generate a random sparse heatmap."""
    cells = []
    used = set()
    for _ in range(n_cells):
        col = _RNG.integers(0, grid_cols)
        row = _RNG.integers(0, grid_rows)
        if (col, row) not in used:
            used.add((col, row))
            cells.append({"col": int(col), "row": int(row),
                          "count": int(_RNG.integers(1, 30))})
    return cells


def _make_dwell(n_elements: int, scale: float) -> list:
    """Generate dwell_by_element entries."""
    elements = [
        "main.app-main", "section.greeting-bar", "div.card-label",
        "span.balance-amount", "account-savings", "account-credit",
        "account-checking", "div.user-pill", "btn-wire",
        "section.quick-actions-section", "div.txn-meta", "a.nav-link",
        "h2.section-title", "div.card-footer", "button.icon-btn",
        "span.balance-change", "div.qa-icon", "txn-payroll",
    ]
    chosen = random.sample(elements, min(n_elements, len(elements)))
    return [{"element": e, "dwell_ms": int(_rand_normal(scale, scale * 0.5, lo=5))}
            for e in chosen]


def _make_clicks(n_clicks: int, session_ms: int) -> list:
    clicks = []
    for i in range(n_clicks):
        t = int(_rand_uniform(500, session_ms))
        clicks.append({
            "x": int(_rand_uniform(50, 1500)),
            "y": int(_rand_uniform(30, 700)),
            "t": t,
            "target": random.choice(["main.app-main", "div.user-pill",
                                      "a.nav-link", "button.icon-btn"]),
            "button": 0,
        })
    clicks.sort(key=lambda c: c["t"])
    return clicks


def _make_fingerprint(profile: str) -> dict:
    """Return a plausible browser fingerprint dict."""
    if profile == "bot":
        # Bots often have headless / unusual fingerprints
        return {
            "screen_width": random.choice([1920, 1024, 800]),
            "screen_height": random.choice([1080, 768, 600]),
            "viewport_width": random.choice([1920, 1024, 800]),
            "viewport_height": random.choice([1080, 768, 600]),
            "color_depth": 24,
            "pixel_ratio": 1.0,
            "timezone": "UTC",
            "timezone_offset": 0,
            "language": "en-US",
            "languages": ["en-US"],
            "platform": random.choice(["Win32", "Linux x86_64"]),
            "hardware_concurrency": random.choice([2, 4]),
            "touch_points": 0,
            "cookie_enabled": random.choice([True, False]),
            "do_not_track": None,
            "online": True,
            "plugins_count": random.choice([0, 1]),
            "webgl_vendor": "Google Inc.",
            "webgl_renderer": "SwiftShader",
            "referrer": "",
            "user_agent": "Mozilla/5.0 (compatible; Bot/1.0)",
        }
    else:
        # Human-like (desktop)
        sw = random.choice([1366, 1440, 1536, 1920, 2560])
        sh = random.choice([768, 900, 864, 1080, 1440])
        return {
            "screen_width": sw,
            "screen_height": sh,
            "viewport_width": sw - _RNG.integers(0, 20),
            "viewport_height": sh - _RNG.integers(100, 200),
            "color_depth": 24 if random.random() < 0.3 else 32,
            "pixel_ratio": random.choice([1.0, 1.25, 1.5, 2.0]),
            "timezone": random.choice(["Asia/Calcutta", "America/New_York",
                                        "Europe/London", "Asia/Tokyo"]),
            "timezone_offset": random.choice([-330, -300, 0, -540]),
            "language": "en-US",
            "languages": ["en-US", "en"],
            "platform": "Win32",
            "hardware_concurrency": random.choice([4, 8, 12, 16]),
            "touch_points": 0,
            "cookie_enabled": True,
            "do_not_track": None,
            "online": True,
            "plugins_count": _RNG.integers(3, 8),
            "webgl_vendor": "Google Inc. (Intel)",
            "webgl_renderer": "ANGLE (Intel, Intel(R) UHD Graphics)",
            "referrer": "http://192.168.31.35:5001/login",
            "user_agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Profile generators
# ─────────────────────────────────────────────────────────────────────────────

def _generate_human(user_id: str = "user_default") -> dict:
    """Generate one human behavior snapshot."""
    session_ms = int(_rand_normal(25000, 15000, lo=3000, hi=120000))
    mouse_speed_mean = _rand_normal(800, 400, lo=50, hi=3000)
    n_clicks = int(_rand_normal(4, 3, lo=0, hi=20))
    n_scroll = int(_rand_normal(80, 60, lo=0, hi=500))
    n_keys = int(_rand_normal(15, 20, lo=0, hi=100))

    clicks = _make_clicks(n_clicks, session_ms)

    return {
        "type": "behavior_snapshot",
        "is_final": random.choice([True, False]),
        "session_id": f"human_{user_id}_{_RNG.integers(1000, 9999)}",
        "page": random.choice(["/home", "/login", "/register", "/verify"]),
        "session_duration_ms": session_ms,
        "mouse_sample_count": int(_rand_normal(60, 40, lo=5, hi=300)),
        "mouse_speed_mean": mouse_speed_mean,
        "mouse_speed_std": _rand_normal(mouse_speed_mean * 0.8, 200, lo=10),
        "mouse_speed_max": _rand_normal(mouse_speed_mean * 3, 1000, lo=mouse_speed_mean),
        "mouse_acceleration_mean": _rand_normal(8000, 5000, lo=100),
        "mouse_acceleration_std": _rand_normal(10000, 6000, lo=100),
        "heatmap_grid_cols": 32,
        "heatmap_grid_rows": 18,
        "heatmap": _make_heatmap(int(_rand_normal(20, 15, lo=3, hi=100))),
        "dwell_by_element": _make_dwell(int(_rand_normal(10, 4, lo=2, hi=18)), scale=400),
        "click_count": n_clicks,
        "clicks": clicks,
        "scroll_event_count": n_scroll,
        "scroll_speed_mean": _rand_normal(1200, 600, lo=50) if n_scroll > 0 else None,
        "scroll_speed_max": _rand_normal(8000, 5000, lo=500) if n_scroll > 0 else None,
        "max_scroll_depth_px": _rand_normal(500, 300, lo=0) if n_scroll > 0 else 0,
        "keystroke_count": n_keys,
        "char_count": max(n_keys - int(_rand_normal(3, 2, lo=0)), 0),
        "typing_speed_cps": _rand_normal(5, 2, lo=0.5, hi=15) if n_keys > 0 else 0,
        "key_dwell_mean_ms": _rand_normal(100, 30, lo=30, hi=300) if n_keys > 0 else None,
        "key_dwell_std_ms": _rand_normal(30, 15, lo=5, hi=100) if n_keys > 0 else None,
        "key_flight_mean_ms": _rand_normal(150, 50, lo=30, hi=500) if n_keys > 0 else None,
        "key_flight_std_ms": _rand_normal(60, 25, lo=10, hi=200) if n_keys > 0 else None,
        "total_idle_ms": int(_rand_normal(3000, 3000, lo=0, hi=session_ms * 0.6)),
        "idle_ratio": _rand_normal(0.15, 0.1, lo=0, hi=0.8),
        "geolocation": {},
        "fingerprint": _make_fingerprint("human"),
        "client_timestamp": "2026-01-01T12:00:00.000Z",
        "server_timestamp": "2026-01-01T12:00:00.000000",
        "ip_address": f"192.168.{_RNG.integers(1,255)}.{_RNG.integers(1,255)}",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        # ── Labels ──
        "_label_bot": 0,
        "_label_anomaly": 0,
        "_label_user_id": user_id,
    }


def _generate_bot() -> dict:
    """Generate one bot behavior snapshot."""
    session_ms = int(_rand_normal(4000, 2000, lo=500, hi=15000))
    # Bots: unnaturally consistent mouse speeds, or zero mouse activity
    has_mouse = random.random() < 0.5
    mouse_speed_mean = _rand_normal(2500, 500, lo=1500, hi=5000) if has_mouse else 0.0

    return {
        "type": "behavior_snapshot",
        "is_final": random.choice([True, False]),
        "session_id": f"bot_{_RNG.integers(10000, 99999)}",
        "page": random.choice(["/home", "/login"]),
        "session_duration_ms": session_ms,
        "mouse_sample_count": int(_rand_normal(10, 5, lo=0, hi=50)) if has_mouse else 0,
        "mouse_speed_mean": mouse_speed_mean,
        "mouse_speed_std": _rand_normal(50, 20, lo=0, hi=100) if has_mouse else 0.0,  # very low variance
        "mouse_speed_max": mouse_speed_mean * _rand_normal(1.1, 0.05, lo=1.01, hi=1.3) if has_mouse else 0.0,
        "mouse_acceleration_mean": _rand_normal(500, 200, lo=0) if has_mouse else 0.0,
        "mouse_acceleration_std": _rand_normal(100, 50, lo=0) if has_mouse else 0.0,
        "heatmap_grid_cols": 32,
        "heatmap_grid_rows": 18,
        "heatmap": _make_heatmap(int(_rand_normal(3, 2, lo=0, hi=8))),
        "dwell_by_element": _make_dwell(int(_rand_normal(2, 1, lo=0, hi=5)), scale=50),
        "click_count": int(_rand_normal(1, 1, lo=0, hi=5)),
        "clicks": _make_clicks(int(_rand_normal(1, 1, lo=0, hi=3)), session_ms),
        "scroll_event_count": int(_rand_normal(5, 5, lo=0, hi=30)),
        "scroll_speed_mean": _rand_normal(3000, 500, lo=2000) if random.random() < 0.3 else None,
        "scroll_speed_max": _rand_normal(3500, 500, lo=2500) if random.random() < 0.3 else None,
        "max_scroll_depth_px": _rand_normal(100, 80, lo=0) if random.random() < 0.3 else 0,
        "keystroke_count": 0,  # bots rarely type naturally
        "char_count": 0,
        "typing_speed_cps": 0,
        "key_dwell_mean_ms": None,
        "key_dwell_std_ms": None,
        "key_flight_mean_ms": None,
        "key_flight_std_ms": None,
        "total_idle_ms": 0,
        "idle_ratio": _rand_normal(0.01, 0.01, lo=0, hi=0.05),
        "geolocation": {},
        "fingerprint": _make_fingerprint("bot"),
        "client_timestamp": "2026-01-01T12:00:00.000Z",
        "server_timestamp": "2026-01-01T12:00:00.000000",
        "ip_address": f"10.{_RNG.integers(0,255)}.{_RNG.integers(0,255)}.{_RNG.integers(1,255)}",
        "user_agent": "Mozilla/5.0 (compatible; Bot/1.0)",
        # ── Labels ──
        "_label_bot": 1,
        "_label_anomaly": 1,
        "_label_user_id": "bot",
    }


def _generate_fraud(user_id: str = "fraud_user") -> dict:
    """
    Generate one fraud behavior snapshot.
    Fraud sessions look human-ish but have erratic/anomalous patterns.
    """
    session_ms = int(_rand_normal(6000, 3000, lo=1000, hi=20000))
    mouse_speed_mean = _rand_normal(2000, 800, lo=200, hi=6000)
    n_clicks = int(_rand_normal(10, 5, lo=2, hi=30))  # unusually many clicks
    n_keys = int(_rand_normal(40, 15, lo=5, hi=80))

    return {
        "type": "behavior_snapshot",
        "is_final": random.choice([True, False]),
        "session_id": f"fraud_{user_id}_{_RNG.integers(1000, 9999)}",
        "page": random.choice(["/home", "/home", "/home"]),  # stuck on one page
        "session_duration_ms": session_ms,
        "mouse_sample_count": int(_rand_normal(30, 20, lo=3, hi=100)),
        "mouse_speed_mean": mouse_speed_mean,
        "mouse_speed_std": _rand_normal(mouse_speed_mean * 1.5, 500, lo=100),  # very high variance
        "mouse_speed_max": _rand_normal(mouse_speed_mean * 5, 2000, lo=mouse_speed_mean * 2),
        "mouse_acceleration_mean": _rand_normal(20000, 8000, lo=5000),  # very high
        "mouse_acceleration_std": _rand_normal(25000, 10000, lo=5000),
        "heatmap_grid_cols": 32,
        "heatmap_grid_rows": 18,
        "heatmap": _make_heatmap(int(_rand_normal(5, 3, lo=1, hi=15))),  # focused on few cells
        "dwell_by_element": _make_dwell(int(_rand_normal(4, 2, lo=1, hi=8)), scale=100),  # short dwell
        "click_count": n_clicks,
        "clicks": _make_clicks(n_clicks, session_ms),
        "scroll_event_count": int(_rand_normal(200, 100, lo=50, hi=600)),  # excessive scrolling
        "scroll_speed_mean": _rand_normal(3000, 1000, lo=1000),
        "scroll_speed_max": _rand_normal(20000, 5000, lo=8000),
        "max_scroll_depth_px": _rand_normal(1500, 500, lo=500),
        "keystroke_count": n_keys,
        "char_count": n_keys,  # exact match (copy-paste or autofill)
        "typing_speed_cps": _rand_normal(20, 5, lo=12, hi=40),  # impossibly fast
        "key_dwell_mean_ms": _rand_normal(30, 10, lo=10, hi=60),  # very short dwell
        "key_dwell_std_ms": _rand_normal(5, 3, lo=1, hi=15),  # very consistent
        "key_flight_mean_ms": _rand_normal(30, 10, lo=10, hi=50),  # very fast transitions
        "key_flight_std_ms": _rand_normal(5, 3, lo=1, hi=15),
        "total_idle_ms": int(_rand_normal(200, 200, lo=0, hi=1000)),
        "idle_ratio": _rand_normal(0.02, 0.02, lo=0, hi=0.1),
        "geolocation": {},
        "fingerprint": _make_fingerprint("human"),  # tries to look human
        "client_timestamp": "2026-01-01T03:00:00.000Z",  # unusual hour
        "server_timestamp": "2026-01-01T03:00:00.000000",
        "ip_address": f"45.{_RNG.integers(0,255)}.{_RNG.integers(0,255)}.{_RNG.integers(1,255)}",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        # ── Labels ──
        "_label_bot": 0,
        "_label_anomaly": 1,
        "_label_user_id": user_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    n_human: int = 1000,
    n_bot: int = 500,
    n_fraud: int = 500,
    n_users: int = 10,
    seed: int = 42,
) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """
    Generate a full synthetic dataset.

    Parameters
    ----------
    n_human : int   – Number of normal human samples.
    n_bot : int     – Number of bot samples.
    n_fraud : int   – Number of fraud samples.
    n_users : int   – Number of distinct simulated users.
    seed : int      – Random seed for reproducibility.

    Returns
    -------
    records : list[dict]
        Raw behavior snapshots (like behavior.json entries) with _label_* keys.
    bot_labels : np.ndarray of shape (n_total,)
        0 = human, 1 = bot.
    anomaly_labels : np.ndarray of shape (n_total,)
        0 = normal, 1 = anomalous (bot + fraud).
    """
    global _RNG
    _RNG = np.random.default_rng(seed)
    random.seed(seed)

    user_ids = [f"user_{i}" for i in range(n_users)]
    records = []

    # Human samples
    for i in range(n_human):
        uid = user_ids[i % n_users]
        records.append(_generate_human(uid))

    # Bot samples
    for _ in range(n_bot):
        records.append(_generate_bot())

    # Fraud samples
    for i in range(n_fraud):
        uid = f"fraud_{i % 5}"
        records.append(_generate_fraud(uid))

    # Shuffle
    combined = list(range(len(records)))
    random.shuffle(combined)
    records = [records[i] for i in combined]

    bot_labels = np.array([r["_label_bot"] for r in records], dtype=np.int32)
    anomaly_labels = np.array([r["_label_anomaly"] for r in records], dtype=np.int32)

    return records, bot_labels, anomaly_labels


if __name__ == "__main__":
    records, bot_lbl, ano_lbl = generate_dataset()
    print(f"Generated {len(records)} samples")
    print(f"  Humans:   {int((bot_lbl == 0).sum())}")
    print(f"  Bots:     {int((bot_lbl == 1).sum())}")
    print(f"  Anomalies:{int((ano_lbl == 1).sum())}")
