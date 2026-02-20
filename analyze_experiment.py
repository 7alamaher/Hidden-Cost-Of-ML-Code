import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="CSV file to analyze")
parser.add_argument("--smell", required=True, help="Variant label (e.g., smell3, smell4)")
args = parser.parse_args()

df = pd.read_csv(args.csv)

baseline = df[df["variant"] == "baseline"]
smell = df[df["variant"] == args.smell]

def summarize(label, data):
    print(f"\n--- {label} ---")
    print(f"Mean train time (sec): {data['train_time_sec'].mean():.4f}")
    print(f"Mean eval time (sec): {data['eval_time_sec'].mean():.4f}")
    print(f"Mean train energy (kWh): {data['train_energy_kwh'].mean():.6f}")
    print(f"Mean train CO2 (kg): {data['train_co2_kg'].mean():.6f}")
    print(f"Mean accuracy: {data['test_accuracy'].mean():.4f}")

summarize("Baseline", baseline)
summarize(args.smell, smell)

co2_increase = ((smell["train_co2_kg"].mean() - baseline["train_co2_kg"].mean())
                / baseline["train_co2_kg"].mean()) * 100

time_increase = ((smell["train_time_sec"].mean() - baseline["train_time_sec"].mean())
                 / baseline["train_time_sec"].mean()) * 100

print("\n--- Impact ---")
print(f"Train time increase: {time_increase:.2f}%")
print(f"CO2 increase: {co2_increase:.2f}%")


