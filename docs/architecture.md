# Atlas Architecture & System Design

## 1. System Overview
Atlas automates the process of evaluating baseline machine learning models on a single tabular CSV dataset.

```text
Client (React Frontend) 
  │
  ├──> POST /upload ─────────> Save CSV, Create Experiment Record (status: uploaded)
  ├──> GET /dataset/{exp_id} ──> Inspect Preview & Statistics
  ├──> POST /train ──────────> Validate Target, Detect Problem Type, Queue Async Training
  └──> GET /results/{exp_id} ─> Poll Execution Status, Display Metrics & Best Model
```

## 2. Process Isolation & Async Training
- CPU-bound model training runs inside Python's `ProcessPoolExecutor`.
- The FastAPI event loop remains unblocked and responsive to incoming requests.
- Workers initialize their own database sessions and emit structured log files to `logs/EXP-YYYYMMDD-###.log`.

## 3. Leak-Free Preprocessing Pipeline
- Split 80/20 train/test split occurs prior to fitting any transformers.
- Imputers (`median` for numeric, `most_frequent` for categorical), encoders (`OneHotEncoder`), and scalers (`StandardScaler`) are fit strictly on `X_train`.
- Test data `X_test` is transformed using fitted transformers.
