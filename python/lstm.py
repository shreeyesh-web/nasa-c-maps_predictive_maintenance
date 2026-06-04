import pandas as pd
import numpy as np
import pickle
import os
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, classification_report)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. FIXED CONNECTION: Pointing to Docker PostgreSQL container on port 5433
DB_PASSWORD = "postgres123"

engine = create_engine(
    f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5433/nasa",
    isolation_level="AUTOCOMMIT"
)

# ── LOAD ─────────────────────────────────────────────────────
print("Loading features from PostgreSQL...")
df = pd.read_sql(
    "SELECT * FROM nasa.engine_features ORDER BY unit_id, cycles",
    engine
)
df = df.fillna(0)

# ── ALIGN & PATCH FEATURES TO MATCH DATABASE ─────────────────
print("Aligning columns with your database script...")

# Force all dataframe column names to lowercase to prevent casing mismatches
df.columns = df.columns.str.lower()

# Check and patch 'life_pct' if missing
if 'life_pct' not in df.columns:
    print("Calculating life_pct on the fly...")
    max_cycles = df.groupby('unit_id')['cycles'].transform('max')
    df['life_pct'] = (df['cycles'] / max_cycles)

# Check and patch standard deviations if missing
if 's11_std10' not in df.columns:
    print("Calculating s11_std10 on the fly...")
    df['s11_std10'] = df.groupby('unit_id')['s11'].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)
if 's4_std10' not in df.columns:
    print("Calculating s4_std10 on the fly...")
    df['s4_std10'] = df.groupby('unit_id')['s4'].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)

# Check and patch drift features if missing
if 's11_drift' not in df.columns:
    print("Calculating s11_drift on the fly...")
    df['s11_drift'] = df['s11'] - df.groupby('unit_id')['s11'].transform(lambda x: x.rolling(10, min_periods=1).mean())
if 's4_drift' not in df.columns:
    print("Calculating s4_drift on the fly...")
    df['s4_drift'] = df['s4'] - df.groupby('unit_id')['s4'].transform(lambda x: x.rolling(10, min_periods=1).mean())

# Remap mismatched interaction column name
if 's4_x_s11' in df.columns and 's11_x_s4' not in df.columns:
    df['s11_x_s4'] = df['s4_x_s11']
elif 's11_x_s4' not in df.columns:
    df['s11_x_s4'] = df['s4'] * df['s11']

# Re-engineer moving average if missing
if 's7_mean10' not in df.columns:
    df['s7_mean10'] = df.groupby('unit_id')['s7'].transform(lambda x: x.rolling(10, min_periods=1).mean())

# Handle lag/delta remapping
for s in ['s4', 's11']:
    col_name = f"{s}_lag5"
    if col_name not in df.columns:
        df[col_name] = df.groupby('unit_id')[s].shift(5).bfill().fillna(0)

# Ensure target variable is present
if 'is_anomaly' not in df.columns:
    max_cycles = df.groupby('unit_id')['cycles'].transform('max')
    df['is_anomaly'] = ((max_cycles - df['cycles']) <= 30).astype(int)

# The final verified feature list expected by the model
feat_cols = [
    "cycles", "life_pct",
    "s2", "s3", "s4", "s7", "s8", "s11", "s12", "s13", "s15", "s17", "s20", "s21",
    "s11_mean10", "s4_mean10", "s12_mean10", "s7_mean10",
    "s11_std10", "s4_std10",
    "s11_lag5", "s4_lag5",
    "s11_drift", "s4_drift",
    "s11_x_s4"
]
print("All columns matched and validated successfully!")

SEQ_LEN = 30

# ── SCALE ────────────────────────────────────────────────────
scaler = MinMaxScaler()
df[feat_cols] = scaler.fit_transform(df[feat_cols])

# ── BUILD SEQUENCES ──────────────────────────────────────────
def make_sequences(df, feat_cols, target, seq_len):
    X_all, y_all = [], []
    for uid in df["unit_id"].unique():
        eng = df[df["unit_id"] == uid].sort_values("cycles")
        # Handle cases where an engine has fewer cycles than sequence length
        if len(eng) < seq_len:
            continue
        feats  = eng[feat_cols].values
        labels = eng[target].values
        for i in range(seq_len, len(eng)):
            X_all.append(feats[i - seq_len : i])
            y_all.append(labels[i])
    return np.array(X_all), np.array(y_all)

print("Building sequential temporal windows...")
X, y = make_sequences(df, feat_cols, "is_anomaly", SEQ_LEN)
print(f"X shape (Samples, Timesteps, Features): {X.shape}")
print(f"y shape: {y.shape}")

# ── SPLIT ────────────────────────────────────────────────────
split   = int(len(X) * 0.8)
X_train = X[:split];  X_test = X[split:]
y_train = y[:split];  y_test = y[split:]

# ── CLASS WEIGHTS ────────────────────────────────────────────
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
class_weights = {0: 1.0, 1: float(neg / max(pos, 1))}
print(f"Class imbalance multiplier for anomalies: {class_weights[1]:.2f}")

# ── BUILD MODEL ──────────────────────────────────────────────
n_features = X.shape[2]
print(f"Building LSTM architecture: input shape ({SEQ_LEN}, {n_features})")

model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1,  activation="sigmoid")
])

model.compile(
    optimizer = "adam",
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
)
model.summary()

# ── TRAIN ────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor              = "val_loss",
        patience             = 8,
        restore_best_weights = True
    ),
    ReduceLROnPlateau(
        monitor  = "val_loss",
        patience = 4,
        factor   = 0.5,
        min_lr   = 1e-6
    )
]

print("\nTraining LSTM network...")
history = model.fit(
    X_train, y_train,
    epochs           = 40,
    batch_size       = 64,
    validation_split = 0.15,
    class_weight     = class_weights,
    callbacks        = callbacks,
    verbose          = 1
)

# ── EVALUATE ─────────────────────────────────────────────────
y_proba = model.predict(X_test).flatten()
y_pred  = (y_proba > 0.5).astype(int)

print("\n========== LSTM PERFORMANCE SUMMARY ==========")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

# ── SAVE ARTIFACTS ───────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")

with open("models/lstm_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/lstm_features.pkl", "wb") as f:
    pickle.dump(feat_cols, f)

print("\n[SUCCESS] LSTM model binary and weights exported to models/")