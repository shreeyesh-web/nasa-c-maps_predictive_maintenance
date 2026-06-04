import pandas as pd
import numpy as np
import pickle
import os

# 1. BYPASS MLFLOW MAINTENANCE BLOCK
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.xgboost
import mlflow.sklearn
from sqlalchemy          import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import (accuracy_score, f1_score,
                                     recall_score, precision_score,
                                     classification_report,
                                     confusion_matrix)
import xgboost as xgb

# ── CONNECTION ───────────────────────────────────────────────
DB_PASSWORD = "postgres123"
engine = create_engine(
    f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5433/nasa",
    isolation_level="AUTOCOMMIT"
)

# ── MLFLOW SETUP ─────────────────────────────────────────────
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("nasa_predictive_maintenance")

# ── LOAD DATA ────────────────────────────────────────────────
print("Loading features from PostgreSQL...")
df = pd.read_sql("SELECT * FROM nasa.engine_features", engine)
df = df.fillna(0)
print(f"Initial Shape from DB: {df.shape}")

# ── ALIGN & PATCH FEATURES TO MATCH DATABASE ─────────────────
print("Aligning columns and computing missing metrics on the fly...")

# Force all dataframe column names to lowercase to prevent casing mismatches
df.columns = df.columns.str.lower()

# 1. Check and patch 'life_pct'
if 'life_pct' not in df.columns:
    max_cycles = df.groupby('unit_id')['cycles'].transform('max')
    df['life_pct'] = (df['cycles'] / max_cycles)

# 2. Check and patch standard deviations
if 's11_std10' not in df.columns:
    df['s11_std10'] = df.groupby('unit_id')['s11'].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)
if 's4_std10' not in df.columns:
    df['s4_std10'] = df.groupby('unit_id')['s4'].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)

# 3. Check and patch drift features
if 's11_drift' not in df.columns:
    df['s11_drift'] = df['s11'] - df.groupby('unit_id')['s11'].transform(lambda x: x.rolling(10, min_periods=1).mean())
if 's4_drift' not in df.columns:
    df['s4_drift'] = df['s4'] - df.groupby('unit_id')['s4'].transform(lambda x: x.rolling(10, min_periods=1).mean())

# 4. Check and patch cross-sensor interaction features
if 's12_minus_s11' not in df.columns:
    df['s12_minus_s11'] = df['s12'] - df['s11']
if 's11_x_s4' not in df.columns:
    if 's4_x_s11' in df.columns:
        df['s11_x_s4'] = df['s4_x_s11']
    else:
        df['s11_x_s4'] = df['s4'] * df['s11']

# 5. Handle lag remappings if needed
for s in ['s4', 's11', 's12', 's7', 's15', 's21', 's20', 's2', 's17', 's3']:
    for lag in [1, 5, 10]:
        col_name = f"{s}_lag{lag}"
        # Skip creating lags that aren't requested in our core feature blueprint to save memory
        if lag == 1 and s not in ['s11', 's4', 's12']: continue
        if lag == 10 and s not in ['s11', 's4', 's12']: continue
        if col_name not in df.columns:
            df[col_name] = df.groupby('unit_id')[s].shift(lag).bfill().fillna(0)

# 6. Safety target engine flag check
if 'is_anomaly' not in df.columns:
    max_cycles = df.groupby('unit_id')['cycles'].transform('max')
    df['is_anomaly'] = ((max_cycles - df['cycles']) <= 30).astype(int)

print("All columns matched and calculated successfully!")

# ── FEATURES BLUEPRINT ───────────────────────────────────────
primary_raw      = ["s2","s3","s4","s7","s8","s11","s12","s13","s15","s17","s20","s21"]
secondary_raw    = ["s9","s14","s6"]
primary_rolling  = [
    "s11_mean5","s4_mean5","s12_mean5","s7_mean5",
    "s11_mean10","s4_mean10","s12_mean10","s7_mean10",
    "s15_mean10","s21_mean10","s20_mean10",
    "s2_mean10","s17_mean10","s3_mean10",
    "s11_mean30","s4_mean30","s12_mean30","s7_mean30",
    "s11_std10","s4_std10","s12_std10","s7_std10",
    "s11_std30","s4_std30",
    "s11_lag1","s4_lag1","s12_lag1",
    "s11_lag5","s4_lag5","s12_lag5","s7_lag5",
    "s15_lag5","s21_lag5","s20_lag5",
    "s2_lag5","s17_lag5","s3_lag5",
    "s11_lag10","s4_lag10","s12_lag10",
    "s11_drift","s4_drift","s12_drift","s7_drift",
    "s15_drift","s21_drift","s20_drift",
    "s2_drift","s17_drift","s3_drift",
    "s11_ewma","s4_ewma","s12_ewma","s7_ewma",
    "s11_pct_rank","s4_pct_rank","s12_pct_rank",
]
secondary_rolling = ["s9_mean10","s14_mean10","s9_drift","s14_drift"]
cross_sensor      = ["s11_x_s4","s12_minus_s11","s7_x_s12","s4_ratio_s9"]
meta              = ["cycles","life_pct"]

features = (meta + primary_raw + secondary_raw +
            primary_rolling + secondary_rolling + cross_sensor)

X = df[features]
y = df["is_anomaly"]

# ── SPLIT + SCALE ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler    = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

neg      = (y_train == 0).sum()
pos      = (y_train == 1).sum()
scale_pw = neg / pos

# ── EXPERIMENT CONFIGURATIONS TO TRY ─────────────────────────
experiments = [
    {
        "run_name":        "xgboost_n500_depth6",
        "n_estimators":    500,
        "max_depth":       6,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "colsample":       0.8,
        "min_child_weight":3,
    },
    {
        "run_name":        "xgboost_n300_depth4",
        "n_estimators":    300,
        "max_depth":       4,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "colsample":       0.8,
        "min_child_weight":1,
    },
    {
        "run_name":        "xgboost_n700_depth8",
        "n_estimators":    700,
        "max_depth":       8,
        "learning_rate":   0.03,
        "subsample":       0.9,
        "colsample":       0.9,
        "min_child_weight":5,
    },
]

best_f1    = 0
best_model = None
best_config = "None"

os.makedirs("plots", exist_ok=True)

for config in experiments:
    print(f"\nRunning Optimization Step: {config['run_name']}")

    with mlflow.start_run(run_name=config["run_name"]):

        # ── LOG PARAMETERS ───────────────────────────────────
        mlflow.log_param("n_estimators",     config["n_estimators"])
        mlflow.log_param("max_depth",        config["max_depth"])
        mlflow.log_param("learning_rate",    config["learning_rate"])
        mlflow.log_param("subsample",        config["subsample"])
        mlflow.log_param("colsample_bytree", config["colsample"])
        mlflow.log_param("min_child_weight", config["min_child_weight"])
        mlflow.log_param("scale_pos_weight", round(scale_pw, 3))
        mlflow.log_param("n_features",       len(features))
        mlflow.log_param("train_size",       len(X_train))
        mlflow.log_param("test_size",        len(X_test))
        mlflow.log_param("anomaly_rate_pct", round(y.mean()*100, 2))

        # ── TRAIN ─────────────────────────────────────────────
        model = xgb.XGBClassifier(
            n_estimators     = config["n_estimators"],
            max_depth        = config["max_depth"],
            learning_rate    = config["learning_rate"],
            subsample        = config["subsample"],
            colsample_bytree = config["colsample"],
            min_child_weight = config["min_child_weight"],
            scale_pos_weight = scale_pw,
            random_state     = 42,
            eval_metric      = "logloss",
            verbosity        = 0
        )
        model.fit(X_train_s, y_train, verbose=False)

        # ── EVALUATE ──────────────────────────────────────────
        y_pred = model.predict(X_test_s)

        acc  = accuracy_score(y_test,  y_pred)
        f1   = f1_score(y_test,        y_pred)
        rec  = recall_score(y_test,    y_pred)
        prec = precision_score(y_test, y_pred)

        # ── LOG METRICS ───────────────────────────────────────
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("precision", prec)

        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_metric("true_negatives",  int(cm[0][0]))
        mlflow.log_metric("false_positives", int(cm[0][1]))
        mlflow.log_metric("false_negatives", int(cm[1][0]))
        mlflow.log_metric("true_positives",  int(cm[1][1]))

        # ── LOG MODEL ─────────────────────────────────────────
        mlflow.xgboost.log_model(model, "xgboost_model")

        # ── LOG FEATURE IMPORTANCE ────────────────────────────
        import matplotlib.pyplot as plt
        importance_df = pd.DataFrame({
            "feature":    features,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df["feature"][::-1],
                importance_df["importance"][::-1])
        ax.set_title(f"Feature Importance - {config['run_name']}")
        plt.tight_layout()

        plot_path = f"plots/importance_{config['run_name']}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Recall:    {rec:.4f}")

        if f1 > best_f1:
            best_f1    = f1
            best_model = model
            best_config = config["run_name"]

# ── SAVE BEST MODEL ───────────────────────────────────────────
os.makedirs("models", exist_ok=True)
pickle.dump(best_model, open("models/xgb_model.pkl",    "wb"))
pickle.dump(scaler,     open("models/xgb_scaler.pkl",   "wb"))
pickle.dump(features,   open("models/xgb_features.pkl", "wb"))

print(f"\n[SUCCESS] Best run selected: {best_config} with F1={best_f1:.4f}")
print("Champion model configurations exported to models/")