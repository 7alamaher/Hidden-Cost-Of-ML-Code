

# Runs the baseline experiment 10 times using fresh Python processes

import subprocess
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, required=True)
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--runs", type=int, default=10)

args = parser.parse_args()

N_RUNS = args.runs
VARIANT = args.variant
CSV_FILE = args.csv


for run_id in range(1, N_RUNS + 1):
    print(f"Starting baseline run {run_id}/{N_RUNS}")

    cmd = [
        sys.executable,                # uses venv Python
        "baseline_cnn_cifar10.py",
        "--run_id", str(run_id),
        "--variant",  VARIANT,
        "--csv", CSV_FILE
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Run {run_id} failed")

print("\nAll baseline runs completed.")

