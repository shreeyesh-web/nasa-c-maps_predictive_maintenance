import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve)

DB_PASSWORD = "postgres123"

engine = create_engine(
    f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5432/nasa",
    isolation_level = "AUTOCOMMIT"
)

# ── LOAD DATA AND MODELS ─────────────────────────────────────
print("Loading data and models...")
df = pd.read_sql(
    "SELECT * FROM nasa.engine_features ORDER BY unit_id, cycles",
    engine
)
df = df.fillna(0)

xgb_model  = pickle.load(open("models/xgb_model.pkl",    "rb"))
xgb_scaler = pickle.load(open("models/xgb_scaler.pkl",   "rb"))
xgb_feats  = pickle.load(open("models/xgb_features.pkl", "rb"))

lstm_model  = __import__('tensorflow').keras.models.load_model("models/lstm_model.keras")
lstm_scaler = pickle.load(open("models/lstm_scaler.pkl",   "rb"))
lstm_feats  = pickle.load(open("models/lstm_features.pkl", "rb"))

import os
os.makedirs("plots", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "primary":  "#1A3A5C",
    "accent":   "#2563A8",
    "green":    "#166534",
    "red":      "#991B1B",
    "orange":   "#92400E",
    "light":    "#D6E8FA",
}

print("Data loaded. Generating plots...")

# ═══════════════════════════════════════════════════════════
# PLOT 1 — Engine Lifespan Distribution
# ═══════════════════════════════════════════════════════════
print("Plot 1: Engine lifespan distribution...")

lifespans = df.groupby("unit_id")["cycles"].max()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Engine Lifespan Distribution — NASA CMAPSS FD001",
             fontsize=14, fontweight="bold", color=COLORS["primary"])

axes[0].hist(lifespans, bins=20, color=COLORS["accent"],
             edgecolor="white", linewidth=0.8)
axes[0].axvline(lifespans.mean(), color=COLORS["red"],
                linestyle="--", linewidth=2,
                label=f"Mean: {lifespans.mean():.0f} cycles")
axes[0].axvline(lifespans.median(), color=COLORS["green"],
                linestyle="--", linewidth=2,
                label=f"Median: {lifespans.median():.0f} cycles")
axes[0].set_xlabel("Lifespan (cycles)")
axes[0].set_ylabel("Number of Engines")
axes[0].set_title("Distribution of Engine Lifespans")
axes[0].legend()

stats_text = (f"Min:    {lifespans.min()} cycles\n"
              f"Max:    {lifespans.max()} cycles\n"
              f"Mean:   {lifespans.mean():.0f} cycles\n"
              f"Std:    {lifespans.std():.0f} cycles\n"
              f"Engines: {len(lifespans)}")
axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes,
             fontsize=13, verticalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor=COLORS["light"],
                       alpha=0.8))
axes[1].axis("off")
axes[1].set_title("Summary Statistics")

plt.tight_layout()
plt.savefig("plots/01_lifespan_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/01_lifespan_distribution.png")

# ═══════════════════════════════════════════════════════════
# PLOT 2 — Sensor Correlation With RUL (from your EDA)
# ═══════════════════════════════════════════════════════════
print("Plot 2: Sensor correlation with RUL...")

sensors = ["s2","s3","s4","s7","s8","s9","s11","s12",
           "s13","s14","s15","s17","s20","s21"]
correlations = {s: df[s].corr(df["rul"]) for s in sensors}
corr_df = (pd.Series(correlations)
           .reset_index()
           .rename(columns={"index":"sensor", 0:"correlation"}))
corr_df = corr_df.sort_values("correlation")

fig, ax = plt.subplots(figsize=(10, 7))
colors = [COLORS["red"] if c < -0.5
          else COLORS["green"] if c > 0.5
          else COLORS["orange"]
          for c in corr_df["correlation"]]
bars = ax.barh(corr_df["sensor"], corr_df["correlation"],
               color=colors, edgecolor="white", linewidth=0.5)
ax.axvline(0,  color="black", linewidth=0.8)
ax.axvline(0.5,  color=COLORS["green"],
           linestyle="--", linewidth=1.2, alpha=0.7, label="+0.5 threshold")
ax.axvline(-0.5, color=COLORS["red"],
           linestyle="--", linewidth=1.2, alpha=0.7, label="-0.5 threshold")

for bar, val in zip(bars, corr_df["correlation"]):
    ax.text(val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9)

ax.set_xlabel("Pearson Correlation with RUL")
ax.set_title("Sensor Correlation with Remaining Useful Life\n"
             "Red = strong negative (increases as engine degrades)  "
             "Green = strong positive",
             fontsize=12, fontweight="bold", color=COLORS["primary"])
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/02_sensor_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/02_sensor_correlation.png")

# ═══════════════════════════════════════════════════════════
# PLOT 3 — Single Engine Degradation Pattern
# ═══════════════════════════════════════════════════════════
print("Plot 3: Engine degradation pattern...")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Engine Degradation Pattern — Unit 1 Through Its Full Life",
             fontsize=13, fontweight="bold", color=COLORS["primary"])

eng = df[df["unit_id"] == 1].sort_values("cycles")

plot_pairs = [
    ("s11",      "s11_mean10", "S11 (bypass ratio)",        axes[0,0]),
    ("s4",       "s4_mean10",  "S4 (HPC outlet temp)",      axes[0,1]),
    ("s12",      "s12_mean10", "S12 (HPC outlet pressure)", axes[1,0]),
    ("s7",       "s7_mean10",  "S7 (bypass ratio total)",   axes[1,1]),
    ("s15",      "s15_mean10", "S15 (bleed enthalpy)",      axes[2,0]),
    ("s21",      "s21_mean10", "S21 (oil pressure ratio)",  axes[2,1]),
]

for raw_col, smooth_col, title, ax in plot_pairs:
    ax.plot(eng["cycles"], eng[raw_col],
            color=COLORS["accent"], alpha=0.3,
            linewidth=0.8, label="Raw sensor")
    ax.plot(eng["cycles"], eng[smooth_col],
            color=COLORS["primary"], linewidth=2,
            label="Rolling mean (10 cycles)")
    ax.axvspan(eng["cycles"].max() - 30, eng["cycles"].max(),
               alpha=0.15, color=COLORS["red"],
               label="Failure zone (RUL≤30)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plots/03_engine_degradation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/03_engine_degradation.png")

# ═══════════════════════════════════════════════════════════
# PLOT 4 — Class Distribution
# ═══════════════════════════════════════════════════════════
print("Plot 4: Class distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Anomaly Label Distribution",
             fontsize=13, fontweight="bold", color=COLORS["primary"])

status_counts = df["health_status"].value_counts()
colors_pie = [COLORS["red"], COLORS["orange"],
              COLORS["accent"], COLORS["green"]]
axes[0].pie(status_counts.values,
            labels=status_counts.index,
            colors=colors_pie[:len(status_counts)],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 11})
axes[0].set_title("Health Status Breakdown")

binary_counts = df["is_anomaly"].value_counts()
bars = axes[1].bar(["Normal (0)", "Anomaly (1)"],
                   binary_counts.values,
                   color=[COLORS["green"], COLORS["red"]],
                   edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, binary_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 100,
                 f"{val:,}\n({val/len(df)*100:.1f}%)",
                 ha="center", fontweight="bold")
axes[1].set_title("Binary Class Distribution")
axes[1].set_ylabel("Number of Rows")

plt.tight_layout()
plt.savefig("plots/04_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/04_class_distribution.png")

# ═══════════════════════════════════════════════════════════
# PLOT 5 — Feature Importance (XGBoost)
# ═══════════════════════════════════════════════════════════
print("Plot 5: XGBoost feature importance...")

importance_df = pd.DataFrame({
    "feature":    xgb_feats,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(importance_df["feature"][::-1],
               importance_df["importance"][::-1],
               color=COLORS["accent"], edgecolor="white")
ax.set_xlabel("Feature Importance Score")
ax.set_title("Top 20 Most Important Features — XGBoost\n"
             "Engineered SQL features dominate over raw sensors",
             fontsize=12, fontweight="bold", color=COLORS["primary"])

for bar, val in zip(bars, importance_df["importance"][::-1]):
    ax.text(bar.get_width() + 0.001,
            bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("plots/05_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/05_feature_importance.png")

# ═══════════════════════════════════════════════════════════
# PLOT 6 — Confusion Matrices Side by Side
# ═══════════════════════════════════════════════════════════
print("Plot 6: Confusion matrices...")

from sklearn.model_selection import train_test_split

# XGBoost predictions
X = df[xgb_feats].fillna(0)
y = df["is_anomaly"]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_s = xgb_scaler.transform(X_test)
y_pred_xgb = xgb_model.predict(X_test_s)

# LSTM predictions
SEQ_LEN   = 30
df_lstm   = df.copy()
df_lstm[lstm_feats] = lstm_scaler.transform(df_lstm[lstm_feats])

def make_sequences(df, feat_cols, target, seq_len):
    X_all, y_all = [], []
    for uid in df["unit_id"].unique():
        eng    = df[df["unit_id"] == uid].sort_values("cycles")
        feats  = eng[feat_cols].values
        labels = eng[target].values
        for i in range(seq_len, len(eng)):
            X_all.append(feats[i - seq_len : i])
            y_all.append(labels[i])
    return np.array(X_all), np.array(y_all)

X_seq, y_seq = make_sequences(df_lstm, lstm_feats, "is_anomaly", SEQ_LEN)
split        = int(len(X_seq) * 0.8)
X_test_seq   = X_seq[split:]
y_test_lstm  = y_seq[split:]
y_proba_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
y_pred_lstm  = (y_proba_lstm > 0.5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices — XGBoost vs LSTM",
             fontsize=13, fontweight="bold", color=COLORS["primary"])

for ax, y_true, y_pred, title in [
    (axes[0], y_test,      y_pred_xgb,  "XGBoost"),
    (axes[1], y_test_lstm, y_pred_lstm, "LSTM"),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap="Blues",
                xticklabels=["Normal","Anomaly"],
                yticklabels=["Normal","Anomaly"])
    ax.set_title(f"{title}\nAccuracy: {(cm.diagonal().sum()/cm.sum()):.4f}",
                 fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("plots/06_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/06_confusion_matrices.png")

# ═══════════════════════════════════════════════════════════
# PLOT 7 — ROC Curves: XGBoost vs LSTM
# ═══════════════════════════════════════════════════════════
print("Plot 7: ROC curves...")

xgb_proba  = xgb_model.predict_proba(X_test_s)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test,      xgb_proba)
fpr_lstm,tpr_lstm, _ = roc_curve(y_test_lstm, y_proba_lstm)
auc_xgb  = auc(fpr_xgb,  tpr_xgb)
auc_lstm = auc(fpr_lstm, tpr_lstm)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_xgb,  tpr_xgb,
        color=COLORS["primary"], linewidth=2.5,
        label=f"XGBoost  (AUC = {auc_xgb:.4f})")
ax.plot(fpr_lstm, tpr_lstm,
        color=COLORS["accent"], linewidth=2.5, linestyle="--",
        label=f"LSTM     (AUC = {auc_lstm:.4f})")
ax.plot([0,1],[0,1], color="gray", linestyle=":", linewidth=1.5,
        label="Random classifier")
ax.fill_between(fpr_xgb, tpr_xgb, alpha=0.08, color=COLORS["primary"])
ax.fill_between(fpr_lstm,tpr_lstm, alpha=0.08, color=COLORS["accent"])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — XGBoost vs LSTM\nNASA CMAPSS FD001 Anomaly Detection",
             fontsize=13, fontweight="bold", color=COLORS["primary"])
ax.legend(fontsize=11, loc="lower right")
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig("plots/07_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/07_roc_curves.png")

# ═══════════════════════════════════════════════════════════
# PLOT 8 — Rolling Feature vs Raw: Why Features Matter
# ═══════════════════════════════════════════════════════════
print("Plot 8: Feature engineering value...")

eng3 = df[df["unit_id"] == 3].sort_values("cycles")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Why SQL Feature Engineering Matters\n"
             "Raw sensors are noisy — engineered features reveal the trend",
             fontsize=13, fontweight="bold", color=COLORS["primary"])

pairs = [
    ("s11",   "s11_mean10",  "s11_drift",  "S11: Raw vs Rolling Mean vs Drift"),
    ("s4",    "s4_mean10",   "s4_drift",   "S4: Raw vs Rolling Mean vs Drift"),
    ("s12",   "s12_mean10",  "s12_drift",  "S12: Raw vs Rolling Mean vs Drift"),
    ("s7",    "s7_mean10",   "s7_drift",   "S7: Raw vs Rolling Mean vs Drift"),
]

for ax, (raw, smooth, drift, title) in zip(axes.flat, pairs):
    ax2 = ax.twinx()
    ax.plot(eng3["cycles"], eng3[raw],
            color=COLORS["accent"], alpha=0.35,
            linewidth=0.8, label="Raw")
    ax.plot(eng3["cycles"], eng3[smooth],
            color=COLORS["primary"], linewidth=2,
            label="Mean10")
    ax2.plot(eng3["cycles"], eng3[drift],
             color=COLORS["red"], linewidth=1.5,
             linestyle="--", label="Drift")
    ax.axvspan(eng3["cycles"].max()-30, eng3["cycles"].max(),
               alpha=0.1, color=COLORS["red"])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Cycle")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("plots/08_feature_engineering_value.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/08_feature_engineering_value.png")

# ═══════════════════════════════════════════════════════════
# PLOT 9 — Model Comparison Summary
# ═══════════════════════════════════════════════════════════
print("Plot 9: Model comparison summary...")

metrics = {
    "Accuracy": [0.9966, 0.9779],
    "F1 Score": [0.9887, 0.9302],
    "Recall":   [0.9919, 0.8874],
}
x     = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, [v[0] for v in metrics.values()],
               width, label="XGBoost",
               color=COLORS["primary"], edgecolor="white")
bars2 = ax.bar(x + width/2, [v[1] for v in metrics.values()],
               width, label="LSTM",
               color=COLORS["accent"], edgecolor="white")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.4f}",
            ha="center", fontsize=9, fontweight="bold")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("XGBoost vs LSTM — Performance Comparison\nNASA CMAPSS FD001",
             fontsize=13, fontweight="bold", color=COLORS["primary"])
ax.set_xticks(x)
ax.set_xticklabels(metrics.keys(), fontsize=12)
ax.set_ylim([0.85, 1.02])
ax.legend(fontsize=11)
ax.axhline(0.95, color="gray", linestyle=":", linewidth=1,
           alpha=0.5, label="0.95 baseline")

plt.tight_layout()
plt.savefig("plots/09_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: plots/09_model_comparison.png")

# ═══════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════
print("\n========== ALL PLOTS DONE ==========")
print("Check your plots/ folder — 9 files saved")
print("\nPlots generated:")
print("  01_lifespan_distribution.png")
print("  02_sensor_correlation.png")
print("  03_engine_degradation.png")
print("  04_class_distribution.png")
print("  05_feature_importance.png")
print("  06_confusion_matrices.png")
print("  07_roc_curves.png")
print("  08_feature_engineering_value.png")
print("  09_model_comparison.png")