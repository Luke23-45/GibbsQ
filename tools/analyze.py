#!/usr/bin/env python3
"""
Analysis CLI
---------------------
High-level tool for exploring and auditing GibbsQ experiment results.
"""

import argparse
import logging
from pathlib import Path
import sys
import os

# Ensure the src/ directory is in PYTHONPATH if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from gibbsq.utils.discovery import find_latest_run
from gibbsq.analysis.reporting import (
    report_stability_sweep, 
    report_policy_comparison, 
    report_drift
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="GibbsQ Run Analyzer Hub")
    parser.add_argument("--dir", type=str, default="outputs", help="Base outputs directory")
    parser.add_argument("--run_id", type=str, help="Analyze a specific run ID instead of the latest")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        log.error(f"Outputs directory not found: {base_dir}")
        sys.exit(1)

    # Core experiment types with specialized reporting
    exp_types = [
        "stability_sweep", 
        "policy_comparison", 
        "drift_verification", 
        "scientific_stress", 
        "dga_training"
    ]
    
    found_any = False
    for etype in exp_types:
        if args.run_id:
            # Look for specific run_id inside experiment type subfolder
            run_dir = base_dir / etype / args.run_id
        else:
            run_dir = find_latest_run(base_dir, etype)
            
        if run_dir and run_dir.exists():
            found_any = True
            if etype == "stability_sweep":
                report_stability_sweep(run_dir)
            elif etype == "policy_comparison":
                report_policy_comparison(run_dir)
            elif etype == "drift_verification":
                report_drift(run_dir)
            else:
                log.info(f"Run Capsule found for {etype}: {run_dir.name} (Automated audit pending)")
        
    if not found_any:
        log.warning("No run capsules discovered in the output directory.")
    else:
        print("\n[Audit] Holistic Analysis Complete.")

if __name__ == "__main__":
    main()
