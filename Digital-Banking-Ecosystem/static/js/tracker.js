/**
 * SecureBank Behavioral Biometrics Tracker
 * Collects: mouse movement, clicks, scroll, keyboard dynamics,
 *           dwell time, heatmap, geolocation, device fingerprint
 * Sends to /api/track/batch on interval + page unload
 */
(function () {
  'use strict';

  const SESSION_ID = Math.random().toString(36).slice(2) + Date.now();
  window.TRACKER_SESSION_ID = SESSION_ID;
  const PAGE = window.location.pathname;
  const T0 = Date.now();

  // ── Storage ──────────────────────────────────────────────────────────────
  const events = [];          // raw event queue (flushed on send)
  const mousePositions = [];  // {x,y,t} sampled
  let lastMouse = null;
  let lastMouseT = T0;
  const velocities = [];
  const accelerations = [];
  let lastVel = 0;

  // Heatmap: 32×18 grid (normalized to viewport)
  const HMAP_COLS = 32, HMAP_ROWS = 18;
  const heatmap = new Array(HMAP_COLS * HMAP_ROWS).fill(0);
  let heatmapTick = null;

  // Dwell: element → accumulated ms
  const dwell = {};
  let dwellTarget = null, dwellStart = null;

  // Keyboard dynamics
  const keyDownTimes = {};
  const keystrokes = []; // {key, dwell_ms, flight_ms}
  let lastKeyUpT = null;
  let charCount = 0;

  // Clicks
  const clicks = [];

  // Scroll
  const scrollEvents = [];
  let lastScrollY = 0, lastScrollT = T0;

  // Idle detection
  let lastActivityT = T0;
  let totalIdleMs = 0;
  let idleStart = null;
  const IDLE_THRESHOLD = 3000;

  // Geolocation (request once)
  let geoData = null;
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      pos => {
        geoData = {
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
          altitude: pos.coords.altitude,
        };
      },
      err => { geoData = { error: err.message }; },
      { timeout: 8000 }
    );
  }

  // ── Helpers ──────────────────────────────────────────────────────────────
  function now() { return Date.now(); }
  function elapsed() { return now() - T0; }

  function heatmapIdx(x, y) {
    const col = Math.min(HMAP_COLS - 1, Math.floor((x / window.innerWidth) * HMAP_COLS));
    const row = Math.min(HMAP_ROWS - 1, Math.floor((y / window.innerHeight) * HMAP_ROWS));
    return row * HMAP_COLS + col;
  }

  function trackLabel(el) {
    if (!el) return 'unknown';
    const dt = el.dataset && el.dataset.track;
    if (dt) return dt;
    if (el.id) return '#' + el.id;
    if (el.tagName) return el.tagName.toLowerCase() + (el.className ? '.' + String(el.className).split(' ')[0] : '');
    return 'unknown';
  }

  function markActivity() {
    const t = now();
    if (idleStart !== null) {
      totalIdleMs += t - idleStart;
      idleStart = null;
    }
    lastActivityT = t;
  }

  function checkIdle() {
    if (now() - lastActivityT > IDLE_THRESHOLD && idleStart === null) {
      idleStart = now();
    }
  }

  // ── Mouse movement ────────────────────────────────────────────────────────
  document.addEventListener('mousemove', e => {
    const t = now();
    const x = e.clientX, y = e.clientY;
    markActivity();

    // Sample every 50 ms
    if (t - lastMouseT >= 50) {
      mousePositions.push({ x, y, t: elapsed() });

      if (lastMouse) {
        const dx = x - lastMouse.x, dy = y - lastMouse.y;
        const dt = (t - lastMouseT) / 1000 || 0.001;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const vel = dist / dt;
        velocities.push(vel);
        const acc = Math.abs(vel - lastVel) / dt;
        accelerations.push(acc);
        lastVel = vel;
      }
      lastMouse = { x, y };
      lastMouseT = t;
    }

    // Heatmap accumulation (debounced via RAF)
    if (heatmapTick === null) {
      heatmapTick = requestAnimationFrame(() => {
        heatmap[heatmapIdx(x, y)]++;
        heatmapTick = null;
      });
    }
  }, { passive: true });

  // ── Dwell time ────────────────────────────────────────────────────────────
  document.addEventListener('mouseover', e => {
    const t = now();
    if (dwellTarget !== null && dwellStart !== null) {
      const key = trackLabel(dwellTarget);
      dwell[key] = (dwell[key] || 0) + (t - dwellStart);
    }
    dwellTarget = e.target;
    dwellStart = t;
  }, { passive: true });

  document.addEventListener('mouseout', () => {
    if (dwellTarget !== null && dwellStart !== null) {
      const key = trackLabel(dwellTarget);
      dwell[key] = (dwell[key] || 0) + (now() - dwellStart);
    }
    dwellTarget = null;
    dwellStart = null;
  }, { passive: true });

  // ── Clicks ────────────────────────────────────────────────────────────────
  document.addEventListener('click', e => {
    markActivity();
    clicks.push({
      x: e.clientX,
      y: e.clientY,
      t: elapsed(),
      target: trackLabel(e.target),
      button: e.button,
    });
  });

  // ── Keyboard dynamics ────────────────────────────────────────────────────
  document.addEventListener('keydown', e => {
    markActivity();
    keyDownTimes[e.code] = now();
  });

  document.addEventListener('keyup', e => {
    const t = now();
    const downT = keyDownTimes[e.code];
    if (downT !== undefined) {
      const dwellMs = t - downT;
      const flightMs = lastKeyUpT !== null ? downT - lastKeyUpT : null;
      keystrokes.push({
        key: e.key.length === 1 ? e.key : e.code, // mask special keys
        dwell_ms: dwellMs,
        flight_ms: flightMs,
        t: elapsed(),
      });
      if (e.key.length === 1) charCount++;
      delete keyDownTimes[e.code];
    }
    lastKeyUpT = t;
  });

  // ── Scroll ────────────────────────────────────────────────────────────────
  document.addEventListener('scroll', () => {
    markActivity();
    const t = now();
    const y = window.scrollY;
    const dy = y - lastScrollY;
    const dt = (t - lastScrollT) / 1000 || 0.001;
    scrollEvents.push({ y, dy, speed: Math.abs(dy / dt), t: elapsed() });
    lastScrollY = y;
    lastScrollT = t;
  }, { passive: true });

  // ── Idle check ────────────────────────────────────────────────────────────
  setInterval(checkIdle, 1000);

  // ── Device fingerprint ────────────────────────────────────────────────────
  function fingerprint() {
    const nav = window.navigator;
    return {
      screen_width: window.screen.width,
      screen_height: window.screen.height,
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
      color_depth: window.screen.colorDepth,
      pixel_ratio: window.devicePixelRatio,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      timezone_offset: new Date().getTimezoneOffset(),
      language: nav.language,
      languages: nav.languages ? Array.from(nav.languages) : [],
      platform: nav.platform,
      hardware_concurrency: nav.hardwareConcurrency,
      device_memory: nav.deviceMemory,
      touch_points: nav.maxTouchPoints,
      cookie_enabled: nav.cookieEnabled,
      do_not_track: nav.doNotTrack,
      online: nav.onLine,
      plugins_count: nav.plugins ? nav.plugins.length : 0,
      webgl_vendor: (function () {
        try {
          const c = document.createElement('canvas');
          const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
          const ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
          return ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : null;
        } catch { return null; }
      })(),
      webgl_renderer: (function () {
        try {
          const c = document.createElement('canvas');
          const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
          const ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
          return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null;
        } catch { return null; }
      })(),
      referrer: document.referrer,
      user_agent: nav.userAgent,
    };
  }

  // ── Statistical helpers ────────────────────────────────────────────────────
  function mean(arr) {
    if (!arr.length) return null;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }
  function std(arr) {
    if (arr.length < 2) return null;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
  }
  function max(arr) { return arr.length ? Math.max(...arr) : null; }

  // ── Build snapshot ────────────────────────────────────────────────────────
  function buildSnapshot(isFinal) {
    const sessionDuration = elapsed();
    const typingDurationSec = sessionDuration / 1000;

    // Finalize current dwell
    if (dwellTarget !== null && dwellStart !== null) {
      const key = trackLabel(dwellTarget);
      dwell[key] = (dwell[key] || 0) + (now() - dwellStart);
      dwellStart = now(); // reset so we don't double-count
    }

    // Top dwell elements
    const dwellSorted = Object.entries(dwell)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([el, ms]) => ({ element: el, dwell_ms: Math.round(ms) }));

    // Heatmap — only non-zero cells
    const heatmapSparse = [];
    for (let i = 0; i < heatmap.length; i++) {
      if (heatmap[i] > 0) {
        heatmapSparse.push({
          col: i % HMAP_COLS,
          row: Math.floor(i / HMAP_COLS),
          count: heatmap[i],
        });
      }
    }

    // Scroll stats
    const scrollSpeeds = scrollEvents.map(s => s.speed);
    const maxScrollDepth = scrollEvents.length ? Math.max(...scrollEvents.map(s => s.y)) : 0;

    return {
      type: 'behavior_snapshot',
      is_final: isFinal,
      session_id: SESSION_ID,
      page: PAGE,
      session_duration_ms: sessionDuration,

      // Mouse features
      mouse_sample_count: mousePositions.length,
      mouse_speed_mean: mean(velocities),
      mouse_speed_std: std(velocities),
      mouse_speed_max: max(velocities),
      mouse_acceleration_mean: mean(accelerations),
      mouse_acceleration_std: std(accelerations),

      // Heatmap
      heatmap_grid_cols: HMAP_COLS,
      heatmap_grid_rows: HMAP_ROWS,
      heatmap: heatmapSparse,

      // Dwell
      dwell_by_element: dwellSorted,

      // Clicks
      click_count: clicks.length,
      clicks: clicks.slice(-50), // last 50

      // Scroll
      scroll_event_count: scrollEvents.length,
      scroll_speed_mean: mean(scrollSpeeds),
      scroll_speed_max: max(scrollSpeeds),
      max_scroll_depth_px: maxScrollDepth,

      // Keyboard
      keystroke_count: keystrokes.length,
      char_count: charCount,
      typing_speed_cps: typingDurationSec > 0 ? charCount / typingDurationSec : null,
      key_dwell_mean_ms: mean(keystrokes.map(k => k.dwell_ms)),
      key_dwell_std_ms: std(keystrokes.map(k => k.dwell_ms)),
      key_flight_mean_ms: mean(keystrokes.filter(k => k.flight_ms !== null).map(k => k.flight_ms)),
      key_flight_std_ms: std(keystrokes.filter(k => k.flight_ms !== null).map(k => k.flight_ms)),
      keystrokes: keystrokes.slice(-100),

      // Idle
      total_idle_ms: totalIdleMs,
      idle_ratio: sessionDuration > 0 ? totalIdleMs / sessionDuration : null,

      // Geolocation
      geolocation: geoData,

      // Device
      fingerprint: fingerprint(),

      // Timestamp
      client_timestamp: new Date().toISOString(),
    };
  }

  // ── Send ──────────────────────────────────────────────────────────────────
  function send(isFinal) {
    const snapshot = buildSnapshot(isFinal);
    const payload = JSON.stringify([snapshot]);

    if (navigator.sendBeacon && isFinal) {
      navigator.sendBeacon('/api/track/batch', new Blob([payload], { type: 'application/json' }));
    } else {
      fetch('/api/track/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        keepalive: isFinal,
      }).then(r => r.json()).then(data => {
        if (data && data.kicked) {
          window.location.href = '/login?kicked=1';
          return;
        }
        // ── Real-time ML HUD update ──────────────────────────────
        if (data && data.ml_analysis) {
          if (typeof window.updateRiskHUD === 'function') {
            window.updateRiskHUD(data.ml_analysis);
          }
        }
        // ── Auto-kicked by ML ────────────────────────────────────
        if (data && data.auto_kicked) {
          if (typeof window.showAutoKicked === 'function') {
            window.showAutoKicked();
          } else {
            window.location.href = '/login?kicked=1';
          }
        }
      }).catch(() => {});
    }

    const statusEl = document.getElementById('tracker-status');
    if (statusEl) {
      statusEl.textContent =
        `[tracker] sent snapshot — ${snapshot.mouse_sample_count} pts, ${snapshot.click_count} clicks, ${snapshot.keystroke_count} keys`;
    }
  }

  // Send every 5 seconds for faster real-time updates and kick detection
  setInterval(() => send(false), 5000);
  window.addEventListener('beforeunload', () => send(true));
  window.addEventListener('pagehide', () => send(true));

  // Initial send after 3 s to capture device fingerprint quickly
  setTimeout(() => send(false), 3000);

})();
