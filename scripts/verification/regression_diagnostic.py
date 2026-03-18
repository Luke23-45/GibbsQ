#!/usr/bin/env python3
"""
Regression Diagnostic Suite for REINFORCE Training.
Identifies 'Unlearning Dips' and 'Critic Stagnation'.
"""

import sys
import os
import subprocess
import json
from pathlib import Path
import numpy as np

def run_diagnostic(epochs=5):
    print("=" * 60)
    print(" RUNNING REINFORCE REGRESSION DIAGNOSTIC (STIMULATED)")
    print("=" * 60)
    
    # Run a very short training with specific overrides
    # Lower sim_time to 500 for speed
    cmd = [
        "python", "scripts/execution/experiment_runner.py", 
        "reinforce_train", 
        "--config-name", "small",
        f"train_epochs={epochs}",
        "simulation.ssa.sim_time=500"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Diagnostic Run Crashed!")
        print(result.stderr)
        return
    
    # Parse logs from stdout
    output = result.stdout
    rewards = []
    evs = []
    
    for line in output.splitlines():
        if "mean_reward:" in line:
            # Extract reward
            try:
                parts = line.split("mean_reward:")
                val = parts[1].split("%")[0].strip()
                rewards.append(float(val))
            except: pass
        if "EV:" in line:
            # Extract EV
            try:
                parts = line.split("EV:")
                val = parts[1].split("[")[0].strip()
                evs.append(float(val))
            except: pass
            
    if not rewards:
        print("❌ Could not extract metrics from logs.")
        return

    print("\n" + "-" * 60)
    print(" ANALYZING REGRESSION METRICS")
    print("-" * 60)
    
    # 1. Check for Unlearning Dip
    max_reward = rewards[0]
    min_reward = min(rewards)
    dip = max_reward - min_reward
    
    print(f"Expert Starting Reward: {rewards[0]:.1f}%")
    print(f"Minimum Observed Reward: {min_reward:.1f}%")
    print(f"Worst-Case Regression:   {dip:.1f}%")
    
    if dip > 5.0:
        print("⚠️  REGRESSION DETECTED: Policy drifted significantly (>5%) from Expert initialization.")
    else:
        print("✅ STABLE: Policy maintained Expert-level performance.")
        
    # 2. Check for Critic Stagnation
    avg_ev = np.mean(evs) if evs else 0
    print(f"Average Explained Variance (EV): {avg_ev:.4f}")
    
    if avg_ev < 0.05:
        print("⚠️  CRITIC STAGNATIOIN: Value Network failed to provide a useful baseline (EV < 0.05).")
    else:
        print("✅ CRITIC ACTIVE: Value Network is learning to explain rewards.")
        
    print("-" * 60)
    if dip > 5.0 or avg_ev < 0.05:
        print("RESULT: ❌ REGRESSION TEST FAILED. Apply Stability Patches.")
    else:
        print("RESULT: ✅ REGRESSION TEST PASSED.")
    print("-" * 60)

if __name__ == "__main__":
    run_diagnostic()
