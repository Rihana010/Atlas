import os
import re
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.orm import Session

from backend.database.models import Experiment, Metric
from backend.schemas.experiment import ColumnSummary
from backend.ml.detection import detect_problem_type
from backend.ml.trainer import run_async_training_worker

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ProcessPoolExecutor for background CPU-bound model training
executor = ProcessPoolExecutor(max_workers=2)

def generate_experiment_id(db: Session) -> str:
    """Generates sequential ID format: EXP-YYYYMMDD-001"""
    today_str = datetime.utcnow().strftime("%Y%m%d")
    prefix = f"EXP-{today_str}-"
    
    # Query highest index for today
    today_exps = db.query(Experiment.experiment_id).filter(
        Experiment.experiment_id.like(f"{prefix}%")
    ).all()
    
    max_num = 0
    for (exp_id,) in today_exps:
        match = re.search(r"EXP-\d{8}-(\d{3})$", exp_id)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    new_num = max_num + 1
    return f"{prefix}{new_num:03d}"

def save_uploaded_csv(file_bytes: bytes, filename: str) -> str:
    """Saves uploaded CSV file to datasets/ directory."""
    clean_filename = re.sub(r'[^a-zA-Z0-9_\.-]', '_', filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"{timestamp}_{clean_filename}"
    file_path = os.path.join(DATASETS_DIR, saved_filename)
    
    with open(file_path, "wb") as f:
        f.write(file_bytes)
        
    return file_path

def analyze_dataset_file(file_path: str) -> Dict[str, Any]:
    """Reads dataset CSV and computes column metadata and row preview."""
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset file missing on disk.")
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")

    if df.empty or len(df.columns) == 0:
        raise ValueError("Uploaded CSV dataset is empty.")

    total_rows = len(df)
    total_columns = len(df.columns)
    
    columns_summary: List[ColumnSummary] = []
    for col in df.columns:
        series = df[col]
        null_cnt = int(series.isna().sum())
        null_pct = round((null_cnt / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        uniq_cnt = int(series.nunique(dropna=True))
        
        # Sample non-null values
        sample_vals = series.dropna().unique()[:5].tolist()
        # Convert non-serializable objects to string representation
        sample_vals_serializable = [
            val.item() if isinstance(val, (pd.Timestamp, float, int, str)) and hasattr(val, 'item') else str(val)
            for val in sample_vals
        ]
        
        columns_summary.append(ColumnSummary(
            name=str(col),
            dtype=str(series.dtype),
            null_count=null_cnt,
            null_percentage=null_pct,
            unique_count=uniq_cnt,
            sample_values=sample_vals_serializable
        ))
        
    # Preview (first 10 rows converted to json-safe dict)
    preview_df = df.head(10).fillna("")
    preview_records = preview_df.to_dict(orient="records")

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "columns": columns_summary,
        "preview": preview_records
    }

def start_training_job(db: Session, experiment_id: str, target_column: str) -> Experiment:
    exp = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found.")

    if not os.path.exists(exp.dataset_path):
        raise FileNotFoundError(f"Dataset file at {exp.dataset_path} does not exist.")

    df = pd.read_csv(exp.dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in dataset.")

    # Deterministic Problem Detection
    problem_type, reasoning = detect_problem_type(df[target_column])

    # Update state to queued
    exp.target_column = target_column
    exp.problem_type = problem_type
    exp.reasoning = reasoning
    exp.status = "queued"
    db.commit()
    db.refresh(exp)

    # Submit worker to ProcessPoolExecutor
    executor.submit(
        run_async_training_worker,
        experiment_id=exp.experiment_id,
        dataset_path=exp.dataset_path,
        target_column=target_column,
        problem_type=problem_type,
        reasoning=reasoning,
        log_dir=LOGS_DIR
    )

    return exp
