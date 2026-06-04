import pandas as pd
from sqlalchemy import create_engine, text

DB_PASSWORD = "postgres123"

# Port 5433 to talk to Docker from Windows
# Database name 'nasa'
engine = create_engine(
    f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5433/nasa",
    isolation_level="AUTOCOMMIT"
)

# --- NEW: CREATE THE SCHEMA FIRST ---
print("Checking/Creating Schema 'nasa'...")
with engine.connect() as conn: 
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS nasa;"))
print("Schema ready.")

sensor_cols = [f"s{i}" for i in range(1, 22)]
all_cols = ["unit_id", "cycles", "setting_1", "setting_2", "setting_3"] + sensor_cols

# LOAD TRAIN
print("\nLoading train_FD001.txt ...")
df_train = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=all_cols
)
print(f"  Shape: {df_train.shape}")
print(f"  Engines: {df_train['unit_id'].nunique()}")

df_train.to_sql("raw_train", engine,
                schema="nasa", if_exists="append", index=False)
print("  Saved to PostgreSQL")

# LOAD TEST
print("\nLoading test_FD001.txt ...")
df_test = pd.read_csv(
    "data/test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=all_cols
)
print(f"  Shape: {df_test.shape}")

df_test.to_sql("raw_test", engine,
               schema="nasa", if_exists="replace", index=False)
print("  Saved to PostgreSQL")

# LOAD RUL
print("\nLoading RUL_FD001.txt ...")
df_rul = pd.read_csv(
    "data/RUL_FD001.txt",
    sep=r"\s+",
    header=None,
    names=["true_rul"]
)
df_rul["unit_id"] = range(1, len(df_rul) + 1)
df_rul = df_rul[["unit_id", "true_rul"]]
print(f"  Shape: {df_rul.shape}")

df_rul.to_sql("test_rul", engine,
              schema="nasa", if_exists="replace", index=False)
print("  Saved to PostgreSQL")

print("\nALL 3 FILES LOADED SUCCESSFULLY")