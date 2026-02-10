import pandas as pd
import numpy as np

np.random.seed(42)

rows = []
num_sensors = 25
samples_per_class = 5000

# -------------------------------
# Noise parameters
# -------------------------------
GAUSSIAN_NOISE_STD = 10
SPIKE_PROB = 0.05
SPIKE_RANGE = (80, 150)
DROPOUT_PROB = 0.05
DROPOUT_VALUE = 0

# -------------------------------
# Spatial correlation (local smoothing)
# -------------------------------
def add_spatial_correlation(sensors, strength=0.15):
    sensors = sensors.astype(float)
    correlated = sensors.copy()
    for i in range(1, len(sensors) - 1):
        correlated[i] += strength * (sensors[i - 1] + sensors[i + 1]) / 2
    return correlated

# -------------------------------
# Noise model
# -------------------------------
def apply_noise(sensors, protected_idx=None):
    sensors = sensors.astype(float)

    # Gaussian noise
    sensors += np.random.normal(0, GAUSSIAN_NOISE_STD, size=sensors.shape)

    # Random spikes
    for i in range(len(sensors)):
        if np.random.rand() < SPIKE_PROB:
            sensors[i] += np.random.randint(*SPIKE_RANGE)

    # Random dropouts (except protected sensors)
    for i in range(len(sensors)):
        if protected_idx is not None and i in protected_idx:
            continue
        if np.random.rand() < DROPOUT_PROB:
            sensors[i] = DROPOUT_VALUE

    return np.clip(sensors, 0, None)

# ------------------------------------------------------------
# R1: Low overall congestion
# ------------------------------------------------------------
for _ in range(samples_per_class):
    sensors = np.random.randint(20, 70, size=num_sensors)
    sensors = add_spatial_correlation(sensors)
    sensors = apply_noise(sensors)
    rows.append(
        {f"s{i+1}": sensors[i] for i in range(num_sensors)} |
        {"best_route": "R1"}
    )

# ------------------------------------------------------------
# R2: Central bottleneck (sensor 13)
# ------------------------------------------------------------
for _ in range(samples_per_class):
    sensors = np.random.randint(40, 90, size=num_sensors)
    sensors[12] = np.random.randint(150, 220)  # defining bottleneck
    sensors = add_spatial_correlation(sensors)
    sensors = apply_noise(sensors, protected_idx=[12])
    rows.append(
        {f"s{i+1}": sensors[i] for i in range(num_sensors)} |
        {"best_route": "R2"}
    )

# ------------------------------------------------------------
# R3: Globally high congestion
# ------------------------------------------------------------
for _ in range(samples_per_class):
    sensors = np.random.randint(180, 260, size=num_sensors)
    sensors = add_spatial_correlation(sensors)
    sensors = apply_noise(sensors)
    rows.append(
        {f"s{i+1}": sensors[i] for i in range(num_sensors)} |
        {"best_route": "R3"}
    )

# ------------------------------------------------------------
# Create DataFrame & shuffle
# ------------------------------------------------------------
df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data.csv", index=False)

print("Dataset with structured noise created successfully!")
print(df.head())
print("\nClass distribution:")
print(df["best_route"].value_counts())

