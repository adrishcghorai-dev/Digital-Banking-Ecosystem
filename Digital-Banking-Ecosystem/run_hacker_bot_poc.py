#!/usr/bin/env python
"""
Automated POC Demo Runner
Runs the hacker bot in demo mode for presentations
"""

import os
import sys

# Add project root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from hacker_bot import HackerBot, print_banner
from colorama import Fore, Back, Style

def run_poc_demo():
    """Run automated POC demonstration"""
    print_banner()
    
    print(f"{Fore.YELLOW}[*] Initializing POC Presentation Mode...{Style.RESET_ALL}\n")
    
    bot = HackerBot()
    
    # Step 1: Load models
    print(f"{'='*80}")
    print(f"  STEP 1: LOADING ML MODELS")
    print(f"{'='*80}")
    
    if bot.load_models():
        print(f"{Fore.GREEN}[OK] All models loaded and ready for testing{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.RED}[ERROR] Failed to load models{Style.RESET_ALL}")
        return
    
    # Step 2: Display model features
    print(f"{'='*80}")
    print(f"  STEP 2: MODEL ARCHITECTURE OVERVIEW")
    print(f"{'='*80}")
    
    bot.display_model_features()
    
    # Step 3: Generate test data
    print(f"{'='*80}")
    print(f"  STEP 3: GENERATING TEST DATASETS")
    print(f"{'='*80}")
    
    bot.generate_test_dataset(n_human=60, n_bot=60, n_fraud=40)
    
    # Step 4: Run batch tests
    print(f"{'='*80}")
    print(f"  STEP 4: RUNNING MODEL EVALUATIONS")
    print(f"{'='*80}")
    
    bot.run_batch_tests(num_samples=50)
    
    # Step 5: Display summary
    print(f"{'='*80}")
    print(f"  STEP 5: PERFORMANCE METRICS & ACCURACY")
    print(f"{'='*80}")
    
    bot.display_results_summary()
    
    # Step 6: Display detailed results
    print(f"{'='*80}")
    print(f"  STEP 6: DETAILED TEST CASE ANALYSIS")
    print(f"{'='*80}")
    
    bot.display_detailed_results(max_samples=15)
    
    # Step 7: Save results
    print(f"{'='*80}")
    print(f"  STEP 7: SAVING EVALUATION RESULTS")
    print(f"{'='*80}")
    
    bot.save_results_to_json("poc_presentation_results.json")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"  POC DEMONSTRATION COMPLETE")
    print(f"{'='*80}\n")
    
    print(f"{Fore.CYAN}[OK] All 4 ML models tested successfully{Style.RESET_ALL}")
    print(f"{Fore.CYAN}[OK] 50 behavioral samples analyzed{Style.RESET_ALL}")
    print(f"{Fore.CYAN}[OK] Results saved to: poc_presentation_results.json{Style.RESET_ALL}")
    print(f"{Fore.CYAN}[OK] Ready for presentation & evaluation{Style.RESET_ALL}\n")
    
    print(f"{Fore.YELLOW}[*] For interactive mode, run: python hacker_bot.py{Style.RESET_ALL}\n")

if __name__ == "__main__":
    try:
        run_poc_demo()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[!] POC interrupted by user{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Error during POC: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
