import pandas as pd
import numpy as np
import pickle
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import (accuracy_score, f1_score,
                                     recall_score, classification_report,
                                     confusion_matrix)
import xgboost as xgb

DB_PASSWORD = "postgres123"

engine = create_engine(
    f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5433/nasa",
    isolation_level = "AUTOCOMMIT"
)

print("Loading features from PostgreSQL view...")
df = pd.read_sql("SELECT * FROM nasa.engine_features", engine)
print(f"Shape: {df.shape}")
df = df.fillna(0)

# ── FEATURE LIST — driven by EDA results ─────────────────────
# PRIMARY sensors (corr > 0.55): full feature engineering
# SECONDARY sensors (corr 0.30-0.55): basic features only
# WEAK/REMOVED: not included

primary_raw = [
    "s2","s3","s4","s7","s8",
    "s11","s12","s13","s15","s17","s20","s21"
]
secondary_raw = ["s9","s14","s6"]

primary_rolling = [
    # Rolling means - 3 windows
    "s11_mean5","s4_mean5","s12_mean5","s7_mean5",
    "s11_mean10","s4_mean10","s12_mean10","s7_mean10",
    "s15_mean10","s21_mean10","s20_mean10",
    "s2_mean10","s17_mean10","s3_mean10",
    "s11_mean30","s4_mean30","s12_mean30","s7_mean30",
    # Rolling std
    "s11_std10","s4_std10","s12_std10","s7_std10",
    "s11_std30","s4_std30",
    # Lag features
    "s11_lag1","s4_lag1","s12_lag1",
    "s11_lag5","s4_lag5","s12_lag5","s7_lag5",
    "s15_lag5","s21_lag5","s20_lag5",
    "s2_lag5","s17_lag5","s3_lag5",
    "s11_lag10","s4_lag10","s12_lag10",
    # Drift
    "s11_drift","s4_drift","s12_drift","s7_drift",
    "s15_drift","s21_drift","s20_drift",
    "s2_drift","s17_drift","s3_drift",
    # EWMA
    "s11_ewma","s4_ewma","s12_ewma","s7_ewma",
    # Percentile rank
    "s11_pct_rank","s4_pct_rank","s12_pct_rank",
]
secondary_rolling = [
    "s9_mean10","s14_mean10",
    "s9_drift","s14_drift",
]
cross_sensor = [
    "s11_x_s4","s12_minus_s11","s7_x_s12","s4_ratio_s9"
]
meta = ["cycles","life_pct"]

features = (meta + primary_raw + secondary_raw +
            primary_rolling + secondary_rolling + cross_sensor)

# ── QUICK PATCH FOR MISSING XGBOOST COLUMNS ───────────────────────────
print("Patching remaining 6 tracking features...")

# 1. Life Percentage (Cycles relative to the max cycle of that engine unit)
max_cycles = df.groupby('unit_id')['cycles'].transform('max')
df['life_pct'] = (df['cycles'] / max_cycles) * 100.0

# 2. Standard Deviations (Window 10 for s11 and s4)
df['s11_std10'] = df.groupby('unit_id')['s1'].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)
df['s4_std10'] = df.groupby('unit_id')['s4'].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)

# 3. Drift Features for s11 and s4 (Value minus rolling 10 mean)
df['s11_drift'] = df['s11'] - df.groupby('unit_id')['s11'].transform(lambda x: x.rolling(10, min_periods=1).mean())
df['s4_drift'] = df['s4'] - df.groupby('unit_id')['s4'].transform(lambda x: x.rolling(10, min_periods=1).mean())

# 4. Sensor Difference
df['s12_minus_s11'] = df['s12'] - df['s11']

print("Patching complete!")
# ───────────────────────────────────────────────────────────────────────

# Your existing code will now run cleanly:
X = df[features]
# ── TARGET LABEL INJECTOR ──────────────────────────────────────────────
print("Calculating anomaly target labels...")

# 1. First, compute Remaining Useful Life (RUL) for each row
max_cycles = df.groupby('unit_id')['cycles'].transform('max')
df['RUL'] = max_cycles - df['cycles']

# 2. Define an anomaly as any engine cycle within 30 days/cycles of failure
df['is_anomaly'] = (df['RUL'] <= 30).astype(int)

print(f"Target labels ready! Total anomalies flagged: {df['is_anomaly'].sum()}")
# ───────────────────────────────────────────────────────────────────────

print(f"Total features: {len(features)}")

X = df[features]
y = df["is_anomaly"]
print(f"Anomaly rate: {y.mean()*100:.1f}%")

# ── SPLIT ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── SCALE ────────────────────────────────────────────────────
scaler    = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── CLASS WEIGHT ─────────────────────────────────────────────
neg      = (y_train == 0).sum()
pos      = (y_train == 1).sum()
scale_pw = neg / pos
print(f"Class ratio (neg/pos): {scale_pw:.2f}")

# ── TRAIN ────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators     = 500,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    scale_pos_weight = scale_pw,
    random_state     = 42,
    eval_metric      = "logloss",
    verbosity        = 0
)
model.fit(
    X_train_s, y_train,
    eval_set        = [(X_test_s, y_test)],
    verbose         = 100
)

# ── EVALUATE ─────────────────────────────────────────────────
y_pred = model.predict(X_test_s)
print("\n========== XGBoost RESULTS ==========")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
      target_names=["Normal","Anomaly"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

importance_df = pd.DataFrame({
    "feature":    features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# ── SAVE ─────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
pickle.dump(model,    open("models/xgb_model.pkl",    "wb"))
pickle.dump(scaler,   open("models/xgb_scaler.pkl",   "wb"))
pickle.dump(features, open("models/xgb_features.pkl", "wb"))
print("\nSaved: models/xgb_model.pkl")
print("Saved: models/xgb_scaler.pkl")
print("Saved: models/xgb_features.pkl")