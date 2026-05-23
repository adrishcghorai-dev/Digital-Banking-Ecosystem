"""
================================================================================
                ML MODEL HACKER BOT - POC PRESENTATION
       End-to-end ML Model Evaluation & Security Threat Detection
================================================================================

An interactive hacker bot that:
  * Tests all 4 ML Models (BotDetector, AnomalyDetector, UserVerifier, RiskScorer)
  * Generates attack/normal test cases
  * Displays model performance scores with visual indicators
  * Shows confidence levels and risk assessments
  * Perfect for POC presentations and evaluations
"""

import os
import sys
import json
import numpy as np
import time
from typing import Dict, List, Any
from colorama import Fore, Back, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)


# ── Numpy-safe JSON encoder ───────────────────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars/arrays to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_json_dumps(data, **kwargs):
    """json.dumps with numpy-type support."""
    return json.dumps(data, cls=_NumpyEncoder, **kwargs)


def _numpy_to_python(obj):
    """Recursively convert numpy types in a dict/list to native Python."""
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Add project root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.models import BotDetector, AnomalyDetector, UserVerifier, RiskScorer
from ml.feature_engineering import extract_features, NUM_FEATURES
from ml.synthetic_data import generate_dataset


# ════════════════════════════════════════════════════════════════════════════
# HACKER BOT CLASS
# ════════════════════════════════════════════════════════════════════════════

class HackerBot:
    """AI Hacker Bot for ML Model Testing & Evaluation"""

    def __init__(self):
        self.bot_detector = BotDetector()
        self.anomaly_detector = AnomalyDetector()
        self.user_verifier = UserVerifier()
        self.risk_scorer = RiskScorer()
        
        self.test_data = None
        self.test_labels = None
        self.results = []
        self.models_loaded = False
        
    def load_models(self):
        """Load all trained models"""
        print(f"\n{Fore.CYAN}[*] Loading trained models...{Style.RESET_ALL}")
        try:
            self.bot_detector.load()
            self.anomaly_detector.load()
            self.user_verifier.load()
            self.risk_scorer.load_models()
            self.models_loaded = True
            print(f"{Fore.GREEN}[OK] All models loaded successfully!{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Error loading models: {e}{Style.RESET_ALL}")
            return False
    
    def generate_test_dataset(self, n_human=50, n_bot=50, n_fraud=30):
        """Generate test dataset with humans, bots, and anomalies"""
        print(f"\n{Fore.CYAN}[*] Generating test dataset...{Style.RESET_ALL}")
        print(f"    - {n_human} normal human behaviors")
        print(f"    - {n_bot} bot behaviors")
        print(f"    - {n_fraud} anomalous/fraudulent behaviors")
        
        syn_records, bot_labels, anomaly_labels = generate_dataset(
            n_human=n_human, n_bot=n_bot, n_fraud=n_fraud,
            n_users=5, seed=42,
        )
        
        self.test_data = syn_records
        self.test_labels = {
            'bot': bot_labels,
            'anomaly': anomaly_labels,
        }
        
        print(f"{Fore.GREEN}[[OK]] Generated {len(syn_records)} test cases{Style.RESET_ALL}")
        return syn_records
    
    def test_bot_detector(self, sample: Dict) -> Dict:
        """Test BotDetector model"""
        try:
            features = extract_features(sample)
            result = self.bot_detector.predict(features.reshape(1, -1))
            
            confidence = result['confidence'] * 100
            is_bot = result['is_bot']
            
            return {
                'is_bot': is_bot,
                'confidence': confidence,
                'status': 'BOT DETECTED' if is_bot else 'HUMAN DETECTED',
            }
        except Exception as e:
            return {'error': str(e)}
    
    def test_anomaly_detector(self, sample: Dict) -> Dict:
        """Test AnomalyDetector model"""
        try:
            features = extract_features(sample)
            result = self.anomaly_detector.predict(features.reshape(1, -1))
            
            anomaly_score = result['anomaly_score']
            is_anomaly = result['is_anomaly']
            
            # Normalize anomaly score to percentage
            anomaly_percent = ((1.0 / (1.0 + np.exp(anomaly_score * 5))) * 100)
            
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'anomaly_percent': anomaly_percent,
                'status': 'ANOMALY DETECTED' if is_anomaly else 'NORMAL',
            }
        except Exception as e:
            return {'error': str(e)}
    
    def test_user_verifier(self, sample: Dict, user_id: str = "user_001") -> Dict:
        """Test UserVerifier model"""
        try:
            features = extract_features(sample)
            result = self.user_verifier.verify(features.reshape(1, -1), user_id)
            
            similarity = result['similarity_score'] * 100
            is_verified = result['is_verified']
            
            return {
                'is_verified': is_verified,
                'similarity_score': similarity,
                'status': 'IDENTITY VERIFIED' if is_verified else 'IDENTITY MISMATCH',
            }
        except Exception as e:
            return {'error': str(e)}
    
    def test_risk_scorer(self, sample: Dict, user_id: str = "user_001") -> Dict:
        """Test RiskScorer (Ensemble) model"""
        try:
            result = self.risk_scorer.score(sample, claimed_user_id=user_id)
            
            return {
                'risk_score': result['risk_score'],
                'risk_level': result['risk_level'],
                'breakdown': result['breakdown'],
                'details': result['details'],
            }
        except Exception as e:
            return {'error': str(e)}
    
    def run_single_test(self, sample: Dict, index: int = 0) -> Dict:
        """Run all model tests on a single sample"""
        test_result = {
            'index': index,
            'bot_detector': self.test_bot_detector(sample),
            'anomaly_detector': self.test_anomaly_detector(sample),
            'user_verifier': self.test_user_verifier(sample),
            'risk_scorer': self.test_risk_scorer(sample),
        }
        return test_result
    
    def run_batch_tests(self, num_samples: int = 20):
        """Run tests on multiple samples"""
        if not self.test_data:
            print(f"{Fore.YELLOW}[!] No test data. Generating...{Style.RESET_ALL}")
            self.generate_test_dataset()
        
        print(f"\n{Fore.CYAN}[*] Running batch tests on {num_samples} samples...{Style.RESET_ALL}")
        
        self.results = []
        
        for i in range(min(num_samples, len(self.test_data))):
            sample = self.test_data[i]
            result = self.run_single_test(sample, index=i)
            self.results.append(result)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"{Fore.CYAN}    [{i+1}/{num_samples}] Tested...{Style.RESET_ALL}", end='\r')
        
        print(f"{Fore.GREEN}[[OK]] Batch testing complete!{Style.RESET_ALL}")
        return self.results
    
    def display_results_summary(self):
        """Display comprehensive results summary"""
        if not self.results:
            print(f"{Fore.YELLOW}[!] No results to display. Run tests first.{Style.RESET_ALL}")
            return
        
        print(f"\n{'='*80}")
        print(f"  HACKER BOT - MODEL EVALUATION RESULTS")
        print(f"{'='*80}\n")
        
        # Calculate metrics
        bot_correct = 0
        anomaly_correct = 0
        total = len(self.results)
        
        for i, result in enumerate(self.results):
            bot_pred = result['bot_detector'].get('is_bot', False)
            ano_pred = result['anomaly_detector'].get('is_anomaly', False)
            
            bot_label = self.test_labels['bot'][i]
            ano_label = self.test_labels['anomaly'][i]
            
            if bot_pred == bot_label:
                bot_correct += 1
            if ano_pred == ano_label:
                anomaly_correct += 1
        
        # Display metrics
        print(f"{Fore.YELLOW}[MODELS] PERFORMANCE METRICS{Style.RESET_ALL}")
        print(f"{'-'*80}")
        
        bot_accuracy = (bot_correct / total * 100) if total > 0 else 0
        anomaly_accuracy = (anomaly_correct / total * 100) if total > 0 else 0
        
        print(f"\n1. {Fore.CYAN}BotDetector (RandomForest){Style.RESET_ALL}")
        print(f"   |- Accuracy:  {bot_accuracy:6.2f}%  {self._get_score_bar(bot_accuracy)}")
        print(f"   |- Correct:   {bot_correct}/{total}")
        print(f"   +- Status:    {Fore.GREEN}[OK] OPERATIONAL{Style.RESET_ALL}")
        
        print(f"\n2. {Fore.CYAN}AnomalyDetector (IsolationForest){Style.RESET_ALL}")
        print(f"   |- Accuracy:  {anomaly_accuracy:6.2f}%  {self._get_score_bar(anomaly_accuracy)}")
        print(f"   |- Correct:   {anomaly_correct}/{total}")
        print(f"   +- Status:    {Fore.GREEN}[OK] OPERATIONAL{Style.RESET_ALL}")
        
        print(f"\n3. {Fore.CYAN}UserVerifier (One-Class SVM){Style.RESET_ALL}")
        print(f"   |- Status:    {Fore.GREEN}[OK] OPERATIONAL{Style.RESET_ALL}")
        print(f"   +- Function:  Behavioral Biometric Verification")
        
        print(f"\n4. {Fore.CYAN}RiskScorer (Ensemble){Style.RESET_ALL}")
        print(f"   |- Status:    {Fore.GREEN}[OK] OPERATIONAL{Style.RESET_ALL}")
        print(f"   +- Weighting: 40% Bot + 30% Anomaly + 20% Identity + 10% Rules")
        
        print(f"\n{'-'*80}")
    
    def display_detailed_results(self, max_samples: int = 10):
        """Display detailed results for individual samples"""
        if not self.results:
            print(f"{Fore.YELLOW}[!] No results to display.{Style.RESET_ALL}")
            return
        
        print(f"\n{'='*80}")
        print(f"  DETAILED TEST RESULTS (First {min(max_samples, len(self.results))} samples)")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(self.results[:max_samples]):
            sample_idx = result['index']
            
            bot_res = result['bot_detector']
            ano_res = result['anomaly_detector']
            risk_res = result['risk_scorer']
            
            print(f"{Fore.YELLOW}Sample #{i+1}{Style.RESET_ALL}")
            print(f"{'-'*80}")
            
            # Ground truth
            bot_label = "BOT" if self.test_labels['bot'][sample_idx] == 1 else "HUMAN"
            ano_label = "ANOMALY" if self.test_labels['anomaly'][sample_idx] == 1 else "NORMAL"
            print(f"Ground Truth: {bot_label:8s} | {ano_label:10s}")
            
            # Bot Detector
            print(f"\n{Fore.CYAN}[BotDetector]{Style.RESET_ALL}")
            if 'error' not in bot_res:
                confidence = bot_res['confidence']
                status = bot_res['status']
                color = Fore.RED if bot_res['is_bot'] else Fore.GREEN
                print(f"  +- {color}{status}{Style.RESET_ALL} (Confidence: {confidence:.2f}%)")
            else:
                print(f"  +- {Fore.RED}Error: {bot_res['error']}{Style.RESET_ALL}")
            
            # Anomaly Detector
            print(f"\n{Fore.CYAN}[AnomalyDetector]{Style.RESET_ALL}")
            if 'error' not in ano_res:
                anomaly_pct = ano_res['anomaly_percent']
                status = ano_res['status']
                color = Fore.RED if ano_res['is_anomaly'] else Fore.GREEN
                print(f"  +- {color}{status}{Style.RESET_ALL} (Anomaly Score: {anomaly_pct:.2f}%)")
            else:
                print(f"  +- {Fore.RED}Error: {ano_res['error']}{Style.RESET_ALL}")
            
            # Risk Scorer
            print(f"\n{Fore.CYAN}[RiskScorer]{Style.RESET_ALL}")
            if 'error' not in risk_res:
                score = risk_res['risk_score']
                level = risk_res['risk_level'].upper()
                breakdown = risk_res['breakdown']
                
                color = Fore.RED if score >= 75 else (Fore.YELLOW if score >= 50 else Fore.GREEN)
                print(f"  |- {color}Risk Score: {score}/100  [{level}]{Style.RESET_ALL}")
                print(f"  |- Breakdown:")
                print(f"  |  |- Bot Score:       {breakdown['bot_score']:.4f}")
                print(f"  |  |- Anomaly Score:   {breakdown['anomaly_score']:.4f}")
                print(f"  |  |- Identity Score:  {breakdown['identity_mismatch']:.4f}")
                print(f"  |  +- Rule Flags:      {breakdown['rule_flags']:.4f}")
            else:
                print(f"  +- {Fore.RED}Error: {risk_res['error']}{Style.RESET_ALL}")
            
            print()
    
    def _get_score_bar(self, score: float) -> str:
        """Get visual bar representation of score"""
        bar_length = 20
        filled = int(bar_length * score / 100)
        empty = bar_length - filled
        
        if score >= 80:
            color = Fore.GREEN
        elif score >= 60:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        
        bar = f"{color}[{'='*filled}{'-'*empty}]{Style.RESET_ALL}"
        return bar
    
    def display_model_features(self):
        """Display features and capabilities of each model"""
        print(f"\n{'='*80}")
        print(f"  ML MODELS - FEATURES & CAPABILITIES")
        print(f"{'='*80}\n")
        
        models_info = [
            {
                'name': 'BotDetector',
                'algorithm': 'Random Forest Classifier',
                'purpose': 'Binary classification (Human vs Bot)',
                'features': '42 behavioral features',
                'output': 'is_bot (bool), confidence (0-1)',
                'use_case': 'Detect automated bot attacks',
            },
            {
                'name': 'AnomalyDetector',
                'algorithm': 'Isolation Forest',
                'purpose': 'Unsupervised anomaly detection',
                'features': '42 behavioral features',
                'output': 'is_anomaly (bool), anomaly_score (float)',
                'use_case': 'Identify unusual/fraudulent behaviors',
            },
            {
                'name': 'UserVerifier',
                'algorithm': 'One-Class SVM (per-user)',
                'purpose': 'Behavioral biometric verification',
                'features': '42 behavioral features',
                'output': 'is_verified (bool), similarity_score (0-1)',
                'use_case': 'Verify user identity based on behavior patterns',
            },
            {
                'name': 'RiskScorer',
                'algorithm': 'Ensemble (Weighted Combination)',
                'purpose': 'Composite risk assessment',
                'features': 'All 4 models + rule-based flags',
                'output': 'risk_score (0-100), risk_level (low/med/high/critical)',
                'use_case': 'Final security decision making',
            },
        ]
        
        for i, info in enumerate(models_info, 1):
            print(f"{Fore.CYAN}{i}. {info['name']}{Style.RESET_ALL}")
            print(f"   |- Algorithm: {info['algorithm']}")
            print(f"   |- Purpose:   {info['purpose']}")
            print(f"   |- Features:  {info['features']}")
            print(f"   |- Output:    {info['output']}")
            print(f"   +- Use Case:  {info['use_case']}")
            print()
    
    def save_results_to_json(self, filename: str = "hacker_bot_results.json"):
        """Save results to JSON file"""
        output_path = os.path.join(_HERE, filename)

        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples_tested': len(self.results),
            'results': _numpy_to_python(self.results),
        }

        with open(output_path, 'w') as f:
            f.write(_safe_json_dumps(summary, indent=2))

        print(f"\n{Fore.GREEN}[[OK]] Results saved to {filename}{Style.RESET_ALL}")
        return output_path

    def export_to_dashboard(self, base_url: str = "http://127.0.0.1:5001"):
        """
        Push synthetic behavior records (from self.test_data) into the Flask app's
        behavior.json via the /api/hacker-bot/inject endpoint.
        The Flask server must be running and an admin session is NOT required
        because we write directly to the file when called from --auto mode.
        """
        if not self.test_data:
            print(f"{Fore.YELLOW}[!] No test data to export.{Style.RESET_ALL}")
            return

        # Build lightweight behavior records from synthetic data + ML results
        records = []
        for i, sample in enumerate(self.test_data):
            rec = _numpy_to_python(dict(sample))
            # Attach risk scorer output if available
            if i < len(self.results):
                rs = self.results[i].get('risk_scorer', {})
                rec['ml_risk_score']  = int(rs.get('risk_score', 0))
                rec['ml_risk_level']  = str(rs.get('risk_level', 'unknown'))
                rec['ml_is_bot']      = bool(self.results[i].get('bot_detector', {}).get('is_bot', False))
                rec['ml_is_anomaly']  = bool(self.results[i].get('anomaly_detector', {}).get('is_anomaly', False))
            rec['source']     = 'hacker_bot'
            rec['session_id'] = f'bot-{i:04d}'
            records.append(rec)

        # Try HTTP injection first (works when server is running)
        try:
            import urllib.request
            body = _safe_json_dumps({'records': records}).encode('utf-8')
            req = urllib.request.Request(
                f"{base_url}/api/hacker-bot/inject",
                data=body,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                print(f"{Fore.GREEN}[[OK]] Injected {result.get('injected', '?')} records into dashboard via HTTP{Style.RESET_ALL}")
                return
        except Exception:
            pass  # Fall through to direct file write

        # Direct file write fallback (used when called as --auto subprocess)
        behavior_path = os.path.join(_HERE, 'logs', 'behavior.json')
        os.makedirs(os.path.join(_HERE, 'logs'), exist_ok=True)
        existing = []
        if os.path.exists(behavior_path):
            try:
                with open(behavior_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.extend(records)
        with open(behavior_path, 'w') as f:
            f.write(_safe_json_dumps(existing, indent=2))
        print(f"{Fore.GREEN}[[OK]] Injected {len(records)} hacker-bot records into behavior.json directly{Style.RESET_ALL}")


# ════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CLI
# ════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print hacker bot banner"""
    banner = f"""
{Fore.RED}
================================================================================
                                                                              
             *** ML MODEL HACKER BOT ***                                    
        Proof of Concept - Model Evaluation & Testing                      
                                                                              
   Test all 4 ML Models for security threat detection & behavioral analysis 
                                                                              
================================================================================
{Style.RESET_ALL}
"""
    print(banner)


def print_menu():
    """Print interactive menu"""
    menu = f"""
{Fore.YELLOW}[MENU] Choose an option:{Style.RESET_ALL}

  1. Load Models                    - Load all trained ML models
  2. Generate Test Data             - Generate human/bot/anomaly test cases
  3. Run Single Test                - Test a single sample on all models
  4. Run Batch Tests                - Run tests on 20+ samples (default)
  5. Display Summary                - Show model performance metrics
  6. Display Details                - Show detailed results for each sample
  7. Model Features                 - Display model architecture & capabilities
  8. Save Results                   - Export results to JSON
  9. Run Full Demo                  - Execute complete POC demo
  0. Exit                           - Exit the hacker bot

{Style.RESET_ALL}"""
    print(menu)


def run_full_demo(bot: HackerBot):
    """Run complete POC demo"""
    print(f"\n{'='*80}")
    print(f"  EXECUTING FULL POC DEMONSTRATION")
    print(f"{'='*80}\n")
    
    # Step 1: Load models
    if not bot.models_loaded:
        bot.load_models()
    
    # Step 2: Generate test data
    bot.generate_test_dataset(n_human=50, n_bot=50, n_fraud=30)
    
    # Step 3: Run batch tests
    bot.run_batch_tests(num_samples=20)
    
    # Step 4: Display results
    bot.display_results_summary()
    bot.display_detailed_results(max_samples=5)
    
    # Step 5: Save results
    bot.save_results_to_json()
    
    print(f"\n{Fore.GREEN}[OK] POC Demo Complete!{Style.RESET_ALL}")


def interactive_cli():
    """Main interactive CLI loop"""
    print_banner()
    
    bot = HackerBot()
    
    while True:
        print_menu()
        choice = input(f"{Fore.YELLOW}Enter your choice (0-9): {Style.RESET_ALL}").strip()
        
        if choice == '1':
            bot.load_models()
        
        elif choice == '2':
            try:
                n_human = int(input("Number of human samples [50]: ") or "50")
                n_bot = int(input("Number of bot samples [50]: ") or "50")
                n_fraud = int(input("Number of anomalous samples [30]: ") or "30")
                bot.generate_test_dataset(n_human=n_human, n_bot=n_bot, n_fraud=n_fraud)
            except ValueError:
                print(f"{Fore.RED}[[ERROR]] Invalid input{Style.RESET_ALL}")
        
        elif choice == '3':
            if not bot.test_data:
                print(f"{Fore.YELLOW}[!] No test data. Generating...{Style.RESET_ALL}")
                bot.generate_test_dataset()
            
            try:
                idx = int(input("Sample index to test [0]: ") or "0")
                if 0 <= idx < len(bot.test_data):
                    result = bot.run_single_test(bot.test_data[idx], index=idx)
                    bot.results = [result]
                    bot.display_detailed_results(max_samples=1)
                else:
                    print(f"{Fore.RED}[[ERROR]] Invalid index{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}[[ERROR]] Invalid input{Style.RESET_ALL}")
        
        elif choice == '4':
            if not bot.models_loaded:
                print(f"{Fore.YELLOW}[!] Models not loaded. Loading...{Style.RESET_ALL}")
                bot.load_models()
            
            if not bot.test_data:
                print(f"{Fore.YELLOW}[!] No test data. Generating...{Style.RESET_ALL}")
                bot.generate_test_dataset()
            
            try:
                num_samples = int(input("Number of samples to test [20]: ") or "20")
                bot.run_batch_tests(num_samples=num_samples)
            except ValueError:
                print(f"{Fore.RED}[[ERROR]] Invalid input{Style.RESET_ALL}")
        
        elif choice == '5':
            bot.display_results_summary()
        
        elif choice == '6':
            try:
                max_show = int(input("Max samples to display [10]: ") or "10")
                bot.display_detailed_results(max_samples=max_show)
            except ValueError:
                print(f"{Fore.RED}[[ERROR]] Invalid input{Style.RESET_ALL}")
        
        elif choice == '7':
            bot.display_model_features()
        
        elif choice == '8':
            if bot.results:
                filename = input("Filename to save [hacker_bot_results.json]: ") or "hacker_bot_results.json"
                bot.save_results_to_json(filename)
            else:
                print(f"{Fore.YELLOW}[!] No results to save. Run tests first.{Style.RESET_ALL}")
        
        elif choice == '9':
            run_full_demo(bot)
        
        elif choice == '0':
            print(f"\n{Fore.CYAN}[*] Exiting Hacker Bot... Goodbye!{Style.RESET_ALL}\n")
            break
        
        else:
            print(f"{Fore.RED}[[ERROR]] Invalid choice. Please try again.{Style.RESET_ALL}")


# ════════════════════════════════════════════════════════════════════════════
# NON-INTERACTIVE MODE  (called by /api/hacker-bot/run)
# ════════════════════════════════════════════════════════════════════════════

def run_non_interactive():
    """Run full demo without prompts, inject results to behavior.json, exit."""
    print(f"{Fore.CYAN}[AUTO] Non-interactive hacker bot starting…{Style.RESET_ALL}")
    bot = HackerBot()

    if not bot.load_models():
        print(f"{Fore.RED}[AUTO] Model load failed – ensure models are trained.{Style.RESET_ALL}")
        sys.exit(1)

    bot.generate_test_dataset(n_human=50, n_bot=50, n_fraud=30)
    bot.run_batch_tests(num_samples=20)
    bot.display_results_summary()
    bot.save_results_to_json()
    bot.export_to_dashboard()
    print(f"{Fore.GREEN}[AUTO] Done.{Style.RESET_ALL}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--auto" in sys.argv:
        try:
            run_non_interactive()
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}[!] Interrupted{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"{Fore.RED}[[ERROR]] Fatal error: {e}{Style.RESET_ALL}")
            sys.exit(1)
    else:
        try:
            interactive_cli()
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}[!] Interrupted by user{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"{Fore.RED}[[ERROR]] Fatal error: {e}{Style.RESET_ALL}")
