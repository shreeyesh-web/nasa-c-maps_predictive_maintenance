Turbofan Engine Predictive Maintenance

Project Overview
Objective: Implements an end-to-end predictive maintenance system using the NASA CMAPSS dataset.
Core Problem: Moves away from simple binary failure classification to estimate Remaining Useful Life (RUL) and categorical health statuses.
Production Focus: Built using a full production stack rather than isolated development notebooks.

System Architecture
1. PostgreSQL Feature Store
Database Layer Processing: Shfited data transformation logic to a PostgreSQL view (engine_features) to serve as a lightweight, query-time feature store.
Feature Scope: Derived 82 features from 14 high-variance raw sensors, filtering out 5 constant and 2 low-correlation sensors during initial SQL EDA.

SQL transformations:
Rolling averages and standard deviations across multiple window sizes (5, 10, and 30 cycles).
Lag deltas and cumulative drift from engine baseline readings.
Exponentially weighted moving averages (EWMA) and cross-sensor interaction terms.

2. Dual-Model Architecture
XGBoost Classifier:
Optimised for low-latency inference via the REST API.
Evaluates single-cycle data snapshots to categorize real-time engine risk zones.

LSTM Network:
Handles sequence-based time-series inference using a 30-cycle lookback window.
Captures long-term degradation trajectories within the monitoring UI.

Imbalance Handling: Utilizes custom class weighting ($5.66$ for XGBoost, $5.25$ for LSTM) calculated directly from the training split's $15\%$ anomaly threshold.

3. MLOps & Infrastructure
MLflow: Tracks hyperparameter tuning history, validation metrics, feature importance data, and model artifacts.
FastAPI: Exposes the XGBoost model via a REST API endpoint featuring Pydantic data validation and auto-generated Swagger UI docs.
Streamlit: Serves a three-page user interface covering a Fleet Overview, Engine Deep Dive charts, and a Live Prediction form.

Docker & CI/CD:
Multi-container deployment orchestrated via docker-compose.
Automated testing pipeline using GitHub Actions and a pytest suite to check RUL calculations, model loading, and API status on every push.

MODEL EVALUATION METRICS

[XGBoost Classifier]
Accuracy: 99.99%
F1 Score: 0.9997
Recall: 1.0000 -- Caught 100% of anomalies in test split
Precision: 0.9994

[LSTM Network]
Accuracy: 98.85%
F1 Score: 0.9615
Recall: 0.9530
Precision: 0.9701

Primary Metric Selection: Treated Recall as the critical benchmark to minimize undetected critical field failures; 
the XGBoost classifier achieved a recall score of 1.00 on test data.

Feature Validation: The top three most important features inside the XGBoost model were all engineered SQL features (s11_pct_rank, s4_mean10, and life_pct), validating the custom window function approach over raw data.

Tech Stack
Languages: Python, SQL
Data & Core ML: PostgreSQL, XGBoost, TensorFlow/Keras, Pandas, NumPy, Scikit-learn
API & Frontend: FastAPI, Pydantic, Streamlit
DevOps & MLOps: MLflow, Docker, Docker Compose, GitHub Actions, pytest

Project Structure
nasa-predictive-maintenance/
├── python/
│   ├── 01_load_data.py          # Extracts and loads NASA files into PostgreSQL
│   ├── 02_train_xgboost.py      # Trains the baseline XGBoost classifier
│   ├── 03_train_lstm.py         # Trains the LSTM sequence model
│   ├── 04_api.py                # FastAPI deployment script
│   ├── 05_dashboard.py          # Streamlit UI implementation
│   ├── 06_plots.py              # Visualization modules
│   ├── 07_check_results.py      # Script to view evaluations without retraining
│   └── 08_train_with_mlflow.py  # MLflow tracking wrapper
├── sql/
│   ├── 01_create_tables.sql     # Database schema setup
│   ├── 02_eda.sql               # Exploratory data queries
│   └── 03_features.sql          # Core engine_features VIEW logic
├── tests/
│   ├── test_models.py           # Model regression and load tests
│   └── test_features.py         # Verification for SQL feature calculation logic
├── init_db/
│   └── 01_init.sql              # Database initialization script for Docker
├── .github/workflows/
│   └── test.yml                 # GitHub Actions pipeline
├── Dockerfile                   # Build steps for the API container
├── Dockerfile.streamlit         # Build steps for the Dashboard container
├── docker-compose.yml           # Multi-container orchestration
└── requirements.txt


Data Pre-requisites
Sourced from the NASA CMAPSS FD001 dataset.

Raw text files must be saved in the local directory as follows:

data/train_FD001.txt
data/test_FD001.txt
data/RUL_FD001.txt

Deployment Option A: Multi-Container Docker (Recommended)
Clone the repository and navigate into the project directory:

Bash
git clone https://github.com/YOURUSERNAME/nasa-predictive-maintenance
cd nasa-predictive-maintenance
* Launch the container stack via docker-compose:
  ```bash
  docker-compose up --build
Open a separate terminal window, initialize a virtual environment, and populate the database:

Bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python python/01_load_data.py

### Deployment Option B: Local Environment Installation
* Ensure a local PostgreSQL instance is active and update individual script `DB_PASSWORD` variables.
* Initialize schema architecture using files inside the `/sql` directory.
* Build the local environment, seed the database, and run model training:
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  python python/01_load_data.py
  python python/02_train_xgboost.py
  python python/03_train_lstm.py
Launch individual localized application services:

Bash
uvicorn python.04_api:app --reload --port 8000
streamlit run python/05_dashboard.py

### Active Local Endpoints
* **FastAPI Swagger Interaction:** `http://localhost:8000/docs`
* **Streamlit Monitoring Interface:** `http://localhost:8501`
* **MLflow UI Server:** `http://localhost:5000` *(Start manually via `mlflow ui` command)*

---

## Key Takeaways
* **SQL Engineering Benefits:** Computing sliding transformations inside PostgreSQL window functions reduces runtime memory overhead and creates central, auditable data assets.
* **Environment Isolation:** Standardizing application dependencies inside Docker Compose
