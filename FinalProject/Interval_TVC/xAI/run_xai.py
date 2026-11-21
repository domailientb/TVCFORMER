"""
run_xai.py – Orchestrator for Interval-TVC XAI pipeline.

Usage (from repo root):
  python FinalProject/Interval_TVC/xAI/run_xai.py [--skip-audit] [--skip-calibration]
                                                    [--skip-plot] [--skip-ablation]
                                                    [--only-plot] [--only-calibration]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def run_step(name: str, script: Path):
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=str(THIS_DIR.parent.parent))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓ {name} completed in {elapsed:.1f}s")
    else:
        print(f"  ✗ {name} FAILED (returncode={result.returncode}) in {elapsed:.1f}s")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Interval-TVC XAI Pipeline")
    parser.add_argument("--skip-audit",       action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-plot",        action="store_true")
    parser.add_argument("--skip-ablation",    action="store_true")
    parser.add_argument("--only-plot",        action="store_true")
    parser.add_argument("--only-calibration", action="store_true")
    args = parser.parse_args()

    if args.only_plot:
        run_step("Plots", THIS_DIR / "plot_xai.py")
        return
    if args.only_calibration:
        run_step("Calibration Analysis", THIS_DIR / "calibration_analysis.py")
        return

    steps = []
    if not args.skip_audit:
        steps.append(("Artifact Audit",        THIS_DIR / "audit_artifacts.py"))
    if not args.skip_calibration:
        steps.append(("Calibration Analysis",  THIS_DIR / "calibration_analysis.py"))
    if not args.skip_plot:
        steps.append(("XAI Plots",             THIS_DIR / "plot_xai.py"))
    if not args.skip_ablation:
        steps.append(("Feature Ablation",      THIS_DIR / "feature_ablation.py"))

    failures = []
    for name, script in steps:
        rc = run_step(name, script)
        if rc != 0:
            failures.append(name)

    print(f"\n{'='*70}")
    print(f"Pipeline complete. Failures: {failures if failures else 'none'}")
    print(f"Results in: {THIS_DIR / 'results'}")
    print(f"Figures in: {THIS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
