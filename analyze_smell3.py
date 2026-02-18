import pandas as pd

# Load CSV
df = pd.read_csv("smell3_experiment.csv")

# Separate baseline and smell3
baseline = df[df["variant"] == "baseline"]
smell3 = df[df["variant"] == "smell3"]

def summarize(label, data):
    print(f"\n--- {label} ---")
    print(f"Mean train time (sec): {data['train_time_sec'].mean():.4f}")
    print(f"Mean eval time (sec): {data['eval_time_sec'].mean():.4f}")
    print(f"Mean train energy (kWh): {data['train_energy_kwh'].mean():.6f}")
    print(f"Mean train CO2 (kg): {data['train_co2_kg'].mean():.6f}")
    print(f"Mean accuracy: {data['test_accuracy'].mean():.4f}")

summarize("Baseline", baseline)
summarize("Smell 3", smell3)

# Percentage increases
co2_increase = (
    (smell3["train_co2_kg"].mean() - baseline["train_co2_kg"].mean())
    / baseline["train_co2_kg"].mean()
) * 100

time_increase = (
    (smell3["train_time_sec"].mean() - baseline["train_time_sec"].mean())
    / baseline["train_time_sec"].mean()
) * 100

print("\n--- Impact of Smell 3 ---")
print(f"Train time increase: {time_increase:.2f}%")
print(f"CO2 increase: {co2_increase:.2f}%")

