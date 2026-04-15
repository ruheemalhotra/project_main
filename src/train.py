# ======================================
# 0. Setup (CI/CD friendly)
# ======================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ======================================
# 1. Load Dataset
# ======================================
data_path = "data/blood_cell_anomaly_detection.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print("Original Shape:", df.shape)


# ======================================
# 2. Preprocessing
# ======================================

# Separate columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Handle missing values
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df[cat_cols] = df[cat_cols].fillna("Unknown")

# One-hot encoding (keeps categorical info)
df_encoded = pd.get_dummies(df, columns=cat_cols)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_encoded)

print("Processed Shape:", data_scaled.shape)

# Train-test split
X_train, X_test = train_test_split(
    data_scaled, test_size=0.2, random_state=42
)


# ======================================
# 3. Build Autoencoder
# ======================================
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),  # bottleneck
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')
model.summary()


# ======================================
# 4. Train Model
# ======================================
history = model.fit(
    X_train, X_train,
    epochs=10,          # keep small for CI/CD
    batch_size=32,
    validation_data=(X_test, X_test),
    shuffle=True,
    verbose=1
)


# ======================================
# 5. Reconstruction Error
# ======================================
reconstructions = model.predict(data_scaled)
mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)


# ======================================
# 6. Threshold (robust)
# ======================================
threshold = np.percentile(mse, 95)
print("Threshold:", threshold)

anomalies = mse > threshold
print("Total anomalies detected:", np.sum(anomalies))


# ======================================
# 7. Save Results
# ======================================
df_results = df.copy()
df_results['reconstruction_error'] = mse
df_results['anomaly'] = anomalies

# Save results
df_results.to_csv("outputs/anomaly_results.csv", index=False)

# Top anomalies
top_anomalies = df_results.sort_values(
    by='reconstruction_error', ascending=False
).head(10)

top_anomalies.to_csv("outputs/top_anomalies.csv", index=False)


# ======================================
# 8. Visualizations (saved to outputs/)
# ======================================

# 1. Error distribution
plt.figure()
plt.hist(mse, bins=50)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.savefig("outputs/error_distribution.png")
plt.close()

# 2. Threshold line plot
plt.figure()
plt.plot(mse, label="Reconstruction Error")
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.title("Error vs Threshold")
plt.savefig("outputs/error_threshold_plot.png")
plt.close()

print("Visualizations saved in outputs/")


# ======================================
# 9. Save Model Artifacts
# ======================================
model.save("model/autoencoder.h5")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(threshold, "model/threshold.pkl")

print("Model, Scaler, Threshold saved successfully!")
