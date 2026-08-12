import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.session import init_db, SessionLocal
from backend.database.models import Experiment, Metric
from backend.services.experiment_service import (
    generate_experiment_id, analyze_dataset_file
)
from backend.ml.detection import detect_problem_type
from backend.ml.trainer import run_async_training_worker
import pandas as pd

def run_test():
    print("Initializing Database...")
    init_db()
    db = SessionLocal()
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

    # TEST 1: CLASSIFICATION PIPELINE
    print("\n==========================================")
    print("TEST 1: CLASSIFICATION DATASET (sample_classification.csv)")
    print("==========================================")
    class_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "sample_classification.csv"))
    
    exp_id_1 = generate_experiment_id(db)
    print(f"Generated Experiment ID: {exp_id_1}")
    
    analysis_1 = analyze_dataset_file(class_csv)
    print(f"Dataset summary: {analysis_1['total_rows']} rows, {analysis_1['total_columns']} columns")
    
    exp1 = Experiment(
        experiment_id=exp_id_1,
        dataset_name="sample_classification.csv",
        dataset_path=class_csv,
        status="uploaded"
    )
    db.add(exp1)
    db.commit()

    df_1 = pd.read_csv(class_csv)
    target_1 = "Survived"
    problem_type_1, reasoning_1 = detect_problem_type(df_1[target_1])
    print(f"Detected Problem Type: {problem_type_1}")
    print(f"Reasoning: {reasoning_1}")
    assert problem_type_1 == "classification", f"Expected classification, got {problem_type_1}"

    print("\nRunning worker training synchronously...")
    run_async_training_worker(
        experiment_id=exp_id_1,
        dataset_path=class_csv,
        target_column=target_1,
        problem_type=problem_type_1,
        reasoning=reasoning_1,
        log_dir=log_dir
    )

    db.refresh(exp1)
    print(f"\nExperiment 1 Status: {exp1.status}")
    print(f"Best Model: {exp1.best_model}")
    assert exp1.status == "completed", f"Experiment status should be completed, got {exp1.status}"

    metrics_1 = db.query(Metric).filter(Metric.experiment_id == exp_id_1).all()
    print("\nClassification Metrics Summary:")
    for m in metrics_1:
        print(f" - {m.model_name}: Acc={m.accuracy:.4f}, Prec={m.precision:.4f}, Rec={m.recall:.4f}, F1={m.f1:.4f}, Time={m.training_time:.3f}s, Status={m.model_status}")

    # TEST 2: REGRESSION PIPELINE
    print("\n==========================================")
    print("TEST 2: REGRESSION DATASET (sample_regression.csv)")
    print("==========================================")
    reg_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "sample_regression.csv"))
    
    exp_id_2 = generate_experiment_id(db)
    print(f"Generated Experiment ID: {exp_id_2}")
    
    analysis_2 = analyze_dataset_file(reg_csv)
    print(f"Dataset summary: {analysis_2['total_rows']} rows, {analysis_2['total_columns']} columns")
    
    exp2 = Experiment(
        experiment_id=exp_id_2,
        dataset_name="sample_regression.csv",
        dataset_path=reg_csv,
        status="uploaded"
    )
    db.add(exp2)
    db.commit()

    df_2 = pd.read_csv(reg_csv)
    target_2 = "Price"
    problem_type_2, reasoning_2 = detect_problem_type(df_2[target_2])
    print(f"Detected Problem Type: {problem_type_2}")
    print(f"Reasoning: {reasoning_2}")
    assert problem_type_2 == "regression", f"Expected regression, got {problem_type_2}"

    print("\nRunning worker training synchronously...")
    run_async_training_worker(
        experiment_id=exp_id_2,
        dataset_path=reg_csv,
        target_column=target_2,
        problem_type=problem_type_2,
        reasoning=reasoning_2,
        log_dir=log_dir
    )

    db.refresh(exp2)
    print(f"\nExperiment 2 Status: {exp2.status}")
    print(f"Best Model: {exp2.best_model}")
    assert exp2.status == "completed", f"Experiment status should be completed, got {exp2.status}"

    metrics_2 = db.query(Metric).filter(Metric.experiment_id == exp_id_2).all()
    print("\nRegression Metrics Summary:")
    for m in metrics_2:
        print(f" - {m.model_name}: RMSE={m.rmse:.4f}, MAE={m.mae:.4f}, R2={m.r2:.4f}, Time={m.training_time:.3f}s, Status={m.model_status}")

    db.close()
    print("\n>>> ALL BACKEND PIPELINE TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_test()
