"""
Feature Engineering Pipeline
============================
Extracts 42 numerical features from a single behavior_snapshot JSON record.
All features are designed to be safe for scikit-learn (no NaN explosions).
"""

import math
import numpy as np

# ── Ordered feature names (used everywhere for consistency) ────────────────────

FEATURE_NAMES = [
    # Mouse dynamics (6)
    "mouse_speed_mean",
    "mouse_speed_std",
    "mouse_speed_max",
    "mouse_acceleration_mean",
    "mouse_acceleration_std",
    "mouse_sample_count",
    # Heatmap statistics (5)
    "heatmap_entropy",
    "heatmap_spread",
    "heatmap_concentration",
    "heatmap_total_count",
    "heatmap_coverage_ratio",
    # Click behavior (3)
    "click_count",
    "mean_click_interval",
    "click_rate",
    # Scroll behavior (4)
    "scroll_event_count",
    "scroll_speed_mean",
    "scroll_speed_max",
    "max_scroll_depth_px",
    # Keystroke dynamics (7)
    "keystroke_count",
    "char_count",
    "typing_speed_cps",
    "key_dwell_mean_ms",
    "key_dwell_std_ms",
    "key_flight_mean_ms",
    "key_flight_std_ms",
    # Idle behavior (2)
    "total_idle_ms",
    "idle_ratio",
    # Session metadata (2)
    "session_duration_ms",
    "is_final",
    # Element dwell (5)
    "num_elements_dwelled",
    "total_dwell_ms",
    "mean_dwell_ms",
    "max_dwell_ms",
    "dwell_entropy",
    # Fingerprint (8)
    "screen_area",
    "viewport_area",
    "pixel_ratio",
    "hardware_concurrency",
    "touch_points",
    "plugins_count",
    "color_depth",
    "is_mobile",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 42


# ── Helper utilities ──────────────────────────────────────────────────────────

def _safe(val, default=0.0):
    """Return *val* as float, falling back to *default* for None / NaN."""
    if val is None:
        return float(default)
    try:
        v = float(val)
        return v if math.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _entropy(counts):
    """Shannon entropy of a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


# ── Main extraction function ─────────────────────────────────────────────────

def extract_features(record: dict) -> np.ndarray:
    """
    Convert one behavior_snapshot dict into a 1-D numpy array of 42 floats.

    Parameters
    ----------
    record : dict
        A single entry from behavior.json.

    Returns
    -------
    np.ndarray of shape (42,)
    """
    f = {}

    # ── Mouse dynamics ────────────────────────────────────────────────────
    f["mouse_speed_mean"]        = _safe(record.get("mouse_speed_mean"))
    f["mouse_speed_std"]         = _safe(record.get("mouse_speed_std"))
    f["mouse_speed_max"]         = _safe(record.get("mouse_speed_max"))
    f["mouse_acceleration_mean"] = _safe(record.get("mouse_acceleration_mean"))
    f["mouse_acceleration_std"]  = _safe(record.get("mouse_acceleration_std"))
    f["mouse_sample_count"]      = _safe(record.get("mouse_sample_count"))

    # ── Heatmap statistics ────────────────────────────────────────────────
    heatmap = record.get("heatmap", [])
    hm_counts = [cell.get("count", 0) for cell in heatmap]
    grid_cols = _safe(record.get("heatmap_grid_cols"), 32)
    grid_rows = _safe(record.get("heatmap_grid_rows"), 18)
    total_cells = max(grid_cols * grid_rows, 1)

    f["heatmap_entropy"]        = _entropy(hm_counts)
    f["heatmap_spread"]         = float(len(hm_counts))
    f["heatmap_total_count"]    = float(sum(hm_counts)) if hm_counts else 0.0
    f["heatmap_concentration"]  = (
        max(hm_counts) / f["heatmap_total_count"]
        if f["heatmap_total_count"] > 0 else 0.0
    )
    f["heatmap_coverage_ratio"] = f["heatmap_spread"] / total_cells

    # ── Click behavior ────────────────────────────────────────────────────
    clicks = record.get("clicks", [])
    f["click_count"] = _safe(record.get("click_count"), 0)

    if len(clicks) >= 2:
        times = sorted(c.get("t", 0) for c in clicks)
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        f["mean_click_interval"] = float(np.mean(intervals))
    else:
        f["mean_click_interval"] = 0.0

    dur_sec = max(_safe(record.get("session_duration_ms"), 1) / 1000.0, 0.001)
    f["click_rate"] = f["click_count"] / dur_sec

    # ── Scroll behavior ───────────────────────────────────────────────────
    f["scroll_event_count"] = _safe(record.get("scroll_event_count"))
    f["scroll_speed_mean"]  = _safe(record.get("scroll_speed_mean"))
    f["scroll_speed_max"]   = _safe(record.get("scroll_speed_max"))
    f["max_scroll_depth_px"]= _safe(record.get("max_scroll_depth_px"))

    # ── Keystroke dynamics ────────────────────────────────────────────────
    f["keystroke_count"]    = _safe(record.get("keystroke_count"))
    f["char_count"]         = _safe(record.get("char_count"))
    f["typing_speed_cps"]   = _safe(record.get("typing_speed_cps"))
    f["key_dwell_mean_ms"]  = _safe(record.get("key_dwell_mean_ms"))
    f["key_dwell_std_ms"]   = _safe(record.get("key_dwell_std_ms"))
    f["key_flight_mean_ms"] = _safe(record.get("key_flight_mean_ms"))
    f["key_flight_std_ms"]  = _safe(record.get("key_flight_std_ms"))

    # ── Idle behavior ─────────────────────────────────────────────────────
    f["total_idle_ms"] = _safe(record.get("total_idle_ms"))
    f["idle_ratio"]    = _safe(record.get("idle_ratio"))

    # ── Session metadata ──────────────────────────────────────────────────
    f["session_duration_ms"] = _safe(record.get("session_duration_ms"))
    f["is_final"]            = 1.0 if record.get("is_final") else 0.0

    # ── Element dwell ─────────────────────────────────────────────────────
    dwell_list = record.get("dwell_by_element", [])
    dwell_times = [d.get("dwell_ms", 0) for d in dwell_list]

    f["num_elements_dwelled"] = float(len(dwell_times))
    f["total_dwell_ms"]       = float(sum(dwell_times))
    f["mean_dwell_ms"]        = float(np.mean(dwell_times)) if dwell_times else 0.0
    f["max_dwell_ms"]         = float(max(dwell_times)) if dwell_times else 0.0
    f["dwell_entropy"]        = _entropy(dwell_times)

    # ── Fingerprint ───────────────────────────────────────────────────────
    fp = record.get("fingerprint", {})
    sw = _safe(fp.get("screen_width"), 1920)
    sh = _safe(fp.get("screen_height"), 1080)
    vw = _safe(fp.get("viewport_width"), 1920)
    vh = _safe(fp.get("viewport_height"), 1080)

    f["screen_area"]          = sw * sh
    f["viewport_area"]        = vw * vh
    f["pixel_ratio"]          = _safe(fp.get("pixel_ratio"), 1.0)
    f["hardware_concurrency"] = _safe(fp.get("hardware_concurrency"), 4)
    f["touch_points"]         = _safe(fp.get("touch_points"), 0)
    f["plugins_count"]        = _safe(fp.get("plugins_count"), 0)
    f["color_depth"]          = _safe(fp.get("color_depth"), 24)

    f["is_mobile"] = 1.0 if (fp.get("touch_points", 0) > 0 and sw < 800) else 0.0

    # ── Assemble vector in canonical order ────────────────────────────────
    return np.array([f[name] for name in FEATURE_NAMES], dtype=np.float64)


def extract_features_batch(records: list) -> np.ndarray:
    """
    Extract features for a list of behavior snapshots.

    Returns
    -------
    np.ndarray of shape (n_records, 42)
    """
    if not records:
        return np.empty((0, NUM_FEATURES), dtype=np.float64)
    return np.vstack([extract_features(r) for r in records])
