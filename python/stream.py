import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import os  # Added to read Docker environment variables

# --- DATABASE CONFIGURATION ---
# These variables will automatically pull from your docker-compose.yml
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "nasa")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123")

st.set_page_config(
    page_title="NASA Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    # Ensure these paths match your Docker volume mappings
    model = pickle.load(open("models/xgb_model.pkl", "rb"))
    scaler = pickle.load(open("models/xgb_scaler.pkl", "rb"))
    feats = pickle.load(open("models/xgb_features.pkl", "rb"))
    return model, scaler, feats

@st.cache_data
def load_data():
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    try:
        # 🟢 CHANGED: Querying the public schema directly where your data lives
        query = "SELECT * FROM nasa.engine_features"
        df = pd.read_sql(query, engine)

        return df.fillna(0)

    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return pd.DataFrame()

# ── SIDEBAR ──────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigate",
    ["Fleet Overview", "Engine Deep Dive", "Live Prediction"]
)

# ── PAGE 1: FLEET OVERVIEW ───────────────────────────────────
if page == "Fleet Overview":
    st.header("Fleet Overview")

    df = load_data()

    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Engines", df["unit_id"].nunique())
        col2.metric("Total Readings", f"{len(df):,}")
        col3.metric("Anomaly Rate", f"{df['is_anomaly'].mean()*100:.1f}%")
        col4.metric("Avg Lifespan",
                    f"{df.groupby('unit_id')['cycles'].max().mean():.0f} cycles")

        st.subheader("Health Status Distribution")
        status_counts = df["health_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        st.bar_chart(status_counts.set_index("Status"))

        st.subheader("Critical and Warning Engines")
        danger = df[df["health_status"].isin(["CRITICAL","WARNING"])]
        if not danger.empty:
            latest = danger.sort_values("cycles").groupby("unit_id").last().reset_index()
            latest = latest[["unit_id","cycles","rul","health_status"]].sort_values("rul")
            st.dataframe(latest, use_container_width=True)
        else:
            st.success("No engines in Critical or Warning status!")
    else:
        st.warning("No data found in table 'nasa.engine_features'. Check if your feature engineering script has run.")

# ── PAGE 2: ENGINE DEEP DIVE ─────────────────────────────────
elif page == "Engine Deep Dive":
    st.header("Engine Deep Dive")

    df = load_data()
    if not df.empty:
        engine_id = st.selectbox(
            "Select Engine",
            sorted(df["unit_id"].unique())
        )

        eng = df[df["unit_id"] == engine_id].sort_values("cycles")
        latest = eng.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Cycle", int(latest["cycles"]))
        col2.metric("RUL", int(latest["rul"]))
        col3.metric("Status", latest["health_status"])

        st.subheader("Sensor Trends Over Lifecycle")
        tab1, tab2, tab3 = st.tabs(["S11", "S4", "S12"])

        with tab1:
            chart_data = eng[["cycles","s11","s11_mean10"]].set_index("cycles")
            st.line_chart(chart_data)
        with tab2:
            chart_data = eng[["cycles","s4","s4_mean10"]].set_index("cycles")
            st.line_chart(chart_data)
        with tab3:
            chart_data = eng[["cycles","s12","s12_mean10"]].set_index("cycles")
            st.line_chart(chart_data)

        st.subheader("RUL Over Lifecycle")
        st.line_chart(eng[["cycles","rul"]].set_index("cycles"))


# ── PAGE 3: LIVE PREDICTION ──────────────────────────────────
# ── PAGE 3: LIVE PREDICTION ──────────────────────────────────
elif page == "Live Prediction":
    st.header("Live Prediction")
    st.markdown("Enter sensor readings to get an instant prediction")

    model, scaler, feats = load_model()

    col1, col2, col3 = st.columns(3)

    with col1:
        unit_id = st.number_input("Unit ID", 1, 100, 1)
        cycles  = st.number_input("Cycles", 1, 500, 100)
        s2      = st.number_input("Sensor S2", 0.0, 1000.0, 642.0)
        s3      = st.number_input("Sensor S3", 0.0, 2000.0, 1589.0)

    with col2:
        s4      = st.number_input("Sensor S4", 0.0, 2000.0, 1400.0)
        s6      = st.number_input("Sensor S6", 0.0, 100.0, 9.0)
        s7      = st.number_input("Sensor S7", 0.0, 600.0, 554.0)
        s11     = st.number_input("Sensor S11", 0.0, 100.0, 47.0)

    with col3:
        s12     = st.number_input("Sensor S12", 0.0, 600.0, 521.0)
        s14     = st.number_input("Sensor S14", 0.0, 9000.0, 309.0)
        s17     = st.number_input("Sensor S17", 0.0, 600.0, 394.0)
        s20     = st.number_input("Sensor S20", 0.0, 100.0, 38.0)
        s21     = st.number_input("Sensor S21", 0.0, 100.0, 23.0)

    if st.button("Predict Health Status", type="primary"):
        life_pct = min(cycles / 200.0, 1.0)
        
        # Static mock fallbacks for sensors not explicitly listed in the input boxes
        s8_val = 0.0
        s9_val = 0.0
        s13_val = 0.0
        s15_val = 0.0

        # Building a complete dictionary mapping all 82 expected keys
        row = {
            # Base variables
            "cycles": cycles, "life_pct": life_pct,
            "s2": s2, "s3": s3, "s4": s4, "s6": s6, "s7": s7, "s8": s8_val, 
            "s9": s9_val, "s11": s11, "s12": s12, "s13": s13_val, "s14": s14, 
            "s15": s15_val, "s17": s17, "s20": s20, "s21": s21,

            # Rolling 5-cycle means
            "s11_mean5": s11, "s4_mean5": s4, "s12_mean5": s12, "s7_mean5": s7,

            # Rolling 10-cycle means
            "s11_mean10": s11, "s4_mean10": s4, "s12_mean10": s12, "s7_mean10": s7,
            "s15_mean10": s15_val, "s21_mean10": s21, "s20_mean10": s20, "s22_mean10": s22 if 's22' in locals() else 0.0, # safety bound
            "s2_mean10": s2, "s17_mean10": s17, "s3_mean10": s3, "s9_mean10": s9_val, "s14_mean10": s14,

            # Rolling 30-cycle means
            "s11_mean30": s11, "s4_mean30": s4, "s12_mean30": s12, "s7_mean30": s7,

            # Standard Deviations
            "s11_std10": 0.0, "s4_std10": 0.0, "s12_std10": 0.0, "s7_std10": 0.0,
            "s11_std30": 0.0, "s4_std30": 0.0,

            # Lag variables
            "s11_lag1": s11, "s4_lag1": s4, "s12_lag1": s12,
            "s11_lag5": s11, "s4_lag5": s4, "s12_lag5": s12, "s7_lag5": s7,
            "s15_lag5": s15_val, "s21_lag5": s21, "s20_lag5": s20, "s2_lag5": s2,
            "s17_lag5": s17, "s3_lag5": s3,
            "s11_lag10": s11, "s4_lag10": s4, "s12_lag10": s12,

            # Drift variables
            "s11_drift": 0.0, "s4_drift": 0.0, "s12_drift": 0.0, "s7_drift": 0.0,
            "s15_drift": 0.0, "s21_drift": 0.0, "s20_drift": 0.0, "s2_drift": 0.0,
            "s17_drift": 0.0, "s3_drift": 0.0, "s9_drift": 0.0, "s14_drift": 0.0,

            # Exponentially Weighted Moving Averages (EWMA)
            "s11_ewma": s11, "s4_ewma": s4, "s12_ewma": s12, "s7_ewma": s7,

            # Percent Ranks
            "s11_pct_rank": 0.5, "s4_pct_rank": 0.5, "s12_pct_rank": 0.5,

            # Interaction Terms / Customized Math Features
            "s11_x_s4": s11 * s4,
            "s12_minus_s11": s12 - s11,
            "s7_x_s12": s7 * s12,
            "s4_ratio_s9": s4 / (s9_val + 0.0001)  # Added small value to avoid division by zero
        }

        # Safe feature mapping and final predictions
        try:
            X = np.array([[row[f] for f in feats]])
            X_s = scaler.transform(X)

            pred = int(model.predict(X_s)[0])
            proba = float(model.predict_proba(X_s)[0][1])

            st.divider()
            if pred == 1:
                st.error(f"ANOMALY DETECTED — Confidence: {proba*100:.1f}%")
                st.warning("Schedule maintenance immediately")
            else:
                st.success(f"NORMAL — Confidence: {(1-proba)*100:.1f}%")
                st.info("Engine operating within normal parameters")

            col1, col2 = st.columns(2)
            col1.metric("Prediction", "ANOMALY" if pred==1 else "NORMAL")
            col2.metric("Anomaly Probability", f"{proba*100:.1f}%")
            
        except KeyError as e:
            st.error(f"❌ Another missing feature found: {e}")
            st.write("Expected features:", feats)