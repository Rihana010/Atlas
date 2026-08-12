# Atlas — Machine Learning Baseline Engine

Atlas is an end-to-end Machine Learning baseline platform designed to automate dataset preprocessing, deterministic problem detection, model training (Logistic Regression / Ridge, Random Forest, XGBoost), and performance evaluation without writing scikit-learn boilerplate.

## Architecture

- **Backend**: FastAPI (Python 3.14+), SQLite database, Scikit-Learn, XGBoost, Pandas.
- **Async Execution**: `ProcessPoolExecutor` handles CPU-bound model training outside the FastAPI event loop.
- **Frontend**: React application styled with a dark W&B / Grafana experiment-tracking aesthetic.
- **Zero Magic Policy**: Leak-free preprocessing (imputing, encoding, scaling fit strictly on train split), deterministic problem detection with explicit reasoning, zero placeholder scores or buttons.

## Setup & Running

### 1. Backend
```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- `POST /upload`: Upload dataset CSV file.
- `GET /dataset/{experiment_id}`: View preview, data types, null counts, and column statistics.
- `POST /train`: Initiate async training pipeline for specified target column.
- `GET /results/{experiment_id}`: Get experiment status, metrics comparison, and best model details.
