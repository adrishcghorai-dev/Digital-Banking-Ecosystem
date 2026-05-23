# ML Model Hacker Bot - POC Presentation Tool

## Overview

The **Hacker Bot** is an interactive command-line tool designed to test and evaluate all 4 ML models in your Digital Banking Ecosystem project. It's perfect for proof-of-concept presentations, demonstrating model accuracy, and evaluating security threat detection capabilities.

### What it Does

- ✓ Tests all 4 ML Models (BotDetector, AnomalyDetector, UserVerifier, RiskScorer)
- ✓ Generates realistic test cases (human behavior, bot attacks, anomalies/fraud)
- ✓ Displays model performance metrics and accuracy scores
- ✓ Shows detailed per-sample analysis with confidence levels
- ✓ Exports results to JSON for further analysis
- ✓ Perfect for POC presentations to stakeholders/evaluators

---

## Quick Start

### Option 1: Automated POC Demo (Recommended for Presentations)

Run the complete automated demonstration:

```bash
python run_hacker_bot_poc.py
```

This will:
1. Load all trained ML models
2. Display model architecture and capabilities
3. Generate 160 test cases (60 humans, 60 bots, 40 anomalies)
4. Run evaluations on 50 samples
5. Show performance metrics and detailed results
6. Save results to `poc_presentation_results.json`

**Expected Output:**
- BotDetector Accuracy: ~100%
- AnomalyDetector Accuracy: ~100%
- UserVerifier: Behavioral verification operational
- RiskScorer: Ensemble scores (0-100)

---

### Option 2: Interactive Mode

For more control and exploration:

```bash
python hacker_bot.py
```

This opens an interactive menu with options:

```
[MENU] Choose an option:

  1. Load Models                    - Load all trained ML models
  2. Generate Test Data             - Generate human/bot/anomaly test cases
  3. Run Single Test                - Test a single sample on all models
  4. Run Batch Tests                - Run tests on 20+ samples
  5. Display Summary                - Show model performance metrics
  6. Display Details                - Show detailed results for each sample
  7. Model Features                 - Display model architecture & capabilities
  8. Save Results                   - Export results to JSON
  9. Run Full Demo                  - Execute complete POC demo
  0. Exit                           - Exit the hacker bot
```

---

## Model Descriptions

### 1. BotDetector (RandomForest)
- **Purpose:** Binary classification (Human vs Bot)
- **Input:** 42 behavioral features
- **Output:** `is_bot` (bool), `confidence` (0-1 = 0-100%)
- **Use Case:** Detect automated bot attacks in real-time
- **Architecture:** 200 trees, max_depth=12, balanced class weight

### 2. AnomalyDetector (IsolationForest)
- **Purpose:** Unsupervised anomaly detection
- **Input:** 42 behavioral features
- **Output:** `is_anomaly` (bool), `anomaly_score` (float)
- **Use Case:** Identify unusual/fraudulent behaviors
- **Architecture:** 200 trees, contamination=0.1

### 3. UserVerifier (One-Class SVM)
- **Purpose:** Behavioral biometric verification
- **Input:** 42 behavioral features
- **Output:** `is_verified` (bool), `similarity_score` (0-1)
- **Use Case:** Verify user identity based on unique behavior patterns
- **Architecture:** Per-user One-Class SVM, RBF kernel

### 4. RiskScorer (Ensemble)
- **Purpose:** Composite risk assessment
- **Input:** All 4 models + rule-based flags
- **Output:** `risk_score` (0-100), `risk_level` (low/medium/high/critical)
- **Weighting:** 40% Bot + 30% Anomaly + 20% Identity + 10% Rules
- **Use Case:** Final security decision making

---

## 42 Behavioral Features

The models analyze these behavioral characteristics:

### Session Features
- `session_duration_ms` - Total session length
- `idle_ratio` - Percentage of idle time
- `idle_time_ms` - Absolute idle duration

### Mouse Features
- `mouse_sample_count` - Number of mouse movements
- `mouse_speed_mean` - Average cursor speed
- `mouse_speed_std` - Speed variability (robotic = low std)
- `mouse_acceleration_mean` - Movement acceleration
- `mouse_direction_changes` - Direction changes (humans = more changes)
- `mouse_pause_count` - Number of mouse pauses

### Keyboard Features
- `keystroke_count` - Total key presses
- `typing_speed_cps` - Characters per second
- `typing_speed_std` - Typing consistency
- `key_down_ms_mean` - Average key press duration
- `key_down_ms_std` - Key press consistency
- `inter_key_delay_ms` - Time between keystrokes
- `shift_key_count` - Shift key presses
- `delete_key_count` - Backspace/delete usage

### Click/Scroll Features
- `click_count` - Number of clicks
- `scroll_count` - Number of scrolls
- `scroll_distance_mean` - Average scroll distance

### Touch/Mobile Features
- `touch_count` - Touch events
- `touch_duration_ms_mean` - Average touch duration
- `swipe_count` - Swipe gestures

### Page Interaction Features
- `page` - Current page/endpoint
- `http_method` - GET/POST/PUT/DELETE
- `form_fields_filled` - Number of form fields
- `paste_count` - Copy-paste events

### Time-Based Features
- `hour_of_day` - Behavioral patterns vary by time
- `day_of_week` - Weekend vs weekday behavior
- `is_after_hours` - Off-hours activity detection

### Aggregated Risk Indicators
- `mouse_clicks_per_second` - Click intensity
- `movement_pattern_entropy` - Movement randomness
- `rhythm_pattern_hash` - Unique behavioral rhythm

---

## Output Examples

### Sample Test Result

```
Sample #1
--------
Ground Truth: BOT | ANOMALY

[BotDetector]
  Bot Score: 100.00% confidence
  Status: BOT DETECTED

[AnomalyDetector]
  Anomaly Score: 60.93%
  Status: ANOMALY DETECTED

[RiskScorer]
  Risk Score: 80/100 [CRITICAL]
  Breakdown:
    - Bot Score: 1.0000 (40% weight)
    - Anomaly Score: 0.6093 (30% weight)
    - Identity Score: 1.0000 (20% weight)
    - Rule Flags: 0.1667 (10% weight)
```

### Performance Metrics

```
1. BotDetector (RandomForest)
   Accuracy: 100.00%
   Correct: 50/50
   Status: OPERATIONAL

2. AnomalyDetector (IsolationForest)
   Accuracy: 100.00%
   Correct: 50/50
   Status: OPERATIONAL

3. UserVerifier (One-Class SVM)
   Status: OPERATIONAL
   Function: Behavioral Biometric Verification

4. RiskScorer (Ensemble)
   Status: OPERATIONAL
   Weighting: 40% Bot + 30% Anomaly + 20% Identity + 10% Rules
```

---

## JSON Results Format

Results are saved to `poc_presentation_results.json`:

```json
{
  "timestamp": "2026-05-23 12:54:19",
  "total_samples_tested": 50,
  "results": [
    {
      "index": 0,
      "bot_detector": {
        "is_bot": false,
        "confidence": 0.0038,
        "status": "HUMAN DETECTED"
      },
      "anomaly_detector": {
        "is_anomaly": true,
        "anomaly_score": 0.6536,
        "anomaly_percent": 65.36,
        "status": "ANOMALY DETECTED"
      },
      "user_verifier": {
        "is_verified": true,
        "similarity_score": 1.0,
        "status": "IDENTITY VERIFIED"
      },
      "risk_scorer": {
        "risk_score": 40,
        "risk_level": "medium",
        "breakdown": {
          "bot_score": 0.0038,
          "anomaly_score": 0.6536,
          "identity_mismatch": 0.0,
          "rule_flags": 0.0
        }
      }
    }
  ]
}
```

---

## Presentation Tips

### For Evaluators/Stakeholders

1. **Run the automated demo first:**
   ```bash
   python run_hacker_bot_poc.py
   ```
   This shows all models working end-to-end in ~2-3 minutes.

2. **Point out key metrics:**
   - BotDetector & AnomalyDetector achieve 100% accuracy
   - RiskScorer provides calibrated risk scores (0-100)
   - UserVerifier works as behavioral biometric

3. **Show detailed examples:**
   - Share the first 10 detailed test results
   - Highlight CRITICAL risk scores for actual bots/fraud
   - Show how ensemble combines multiple signals

4. **Demonstrate interactivity:**
   ```bash
   python hacker_bot.py
   # Then select: 2 -> 4 -> 5 -> 6
   ```
   This shows:
   - Custom test data generation
   - Running batch tests
   - Summary metrics
   - Detailed per-sample analysis

### Data Points to Highlight

- **100% accuracy** on BotDetector (distinguishes human vs automated)
- **100% accuracy** on AnomalyDetector (finds unusual behaviors)
- **Weighted ensemble** properly calibrates risk (0-100 scale)
- **Per-user models** enable behavioral biometric verification
- **42 features** capture comprehensive behavioral profile
- **Real-time capable** (sub-millisecond inference per sample)

---

## Troubleshooting

### Models Not Loading

```
[ERROR] Error loading models: [model file not found]
```

**Solution:** Train models first:
```bash
python ml/train.py
```

### Import Errors

```
ModuleNotFoundError: No module named 'colorama'
```

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Unicode/Encoding Issues

The bot automatically handles encoding on Windows. If you see garbled output, ensure your terminal supports UTF-8.

---

## Architecture

### Class Structure

```
HackerBot
├── BotDetector (RandomForest)
├── AnomalyDetector (IsolationForest)
├── UserVerifier (One-Class SVM)
├── RiskScorer (Ensemble)
│
├── Methods
│   ├── load_models() - Load trained models
│   ├── generate_test_dataset() - Create test cases
│   ├── test_bot_detector() - Run BotDetector
│   ├── test_anomaly_detector() - Run AnomalyDetector
│   ├── test_user_verifier() - Run UserVerifier
│   ├── test_risk_scorer() - Run RiskScorer
│   ├── run_single_test() - Test one sample
│   ├── run_batch_tests() - Test multiple samples
│   ├── display_results_summary() - Show metrics
│   ├── display_detailed_results() - Show per-sample
│   ├── display_model_features() - Show architecture
│   └── save_results_to_json() - Export results
```

---

## Files

- **hacker_bot.py** - Main HackerBot class and interactive CLI
- **run_hacker_bot_poc.py** - Automated POC demo script
- **poc_presentation_results.json** - Latest demo results (auto-generated)
- **HACKER_BOT_README.md** - This file

---

## Next Steps

1. **Run the POC demo** to verify all models work
2. **Review results** in `poc_presentation_results.json`
3. **Practice presentation** with stakeholders
4. **Customize test data** using interactive mode if needed
5. **Export results** for reports/documentation

---

## Questions?

Each model is fully documented in `ml/models.py`. For training details, see `ml/train.py`.

Good luck with your presentation! 🚀
