
# Runs the baseline experiment 10 times using fresh Python processes

import subprocess
import sys

N_RUNS = 10

for run_id in range(1, N_RUNS + 1):
    print(f"Starting baseline run {run_id}/{N_RUNS}")

    cmd = [
        sys.executable,                # uses venv Python
        "baseline_cnn_cifar10.py",
        "--run_id", str(run_id),
        "--variant", "baseline",
        "--csv", "hidden_cost_experiments.csv"
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Run {run_id} failed")

print("\nAll baseline runs completed.")
