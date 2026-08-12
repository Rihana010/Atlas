import os
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

from backend.ml.preprocessing import DataPreprocessor
from backend.database.session import SessionLocal
from backend.database.models import Experiment, Metric

def configure_experiment_logger(experiment_id: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{experiment_id}.log")
    
    logger = logging.getLogger(experiment_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def run_async_training_worker(
    experiment_id: str,
    dataset_path: str,
    target_column: str,
    problem_type: str,
    reasoning: str,
    log_dir: str
):
    """
    Executed inside ProcessPoolExecutor.
    Creates an independent database session and executes preprocessing & model training.
    """
    logger = configure_experiment_logger(experiment_id, log_dir)
    logger.info(f"Worker started training pipeline for Experiment ID: {experiment_id}")
    logger.info(f"Dataset path: {dataset_path} | Target: {target_column} | Problem: {problem_type}")
    logger.info(f"Reasoning: {reasoning}")
    
    db = SessionLocal()
    try:
        # Update experiment state to 'running'
        exp = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
        if not exp:
            logger.error(f"Experiment record {experiment_id} not found in database.")
            return
            
        exp.status = "running"
        exp.started_at = datetime.utcnow()
        exp.target_column = target_column
        exp.problem_type = problem_type
        exp.reasoning = reasoning
        db.commit()

        # Load Dataset
        logger.info("Loading CSV dataset into memory...")
        df = pd.read_csv(dataset_path)

        # Preprocess Dataset (Leak-free)
        logger.info("Executing leak-free preprocessing (imputation, encoding, scaling fit strictly on train split)...")
        preprocessor = DataPreprocessor(problem_type=problem_type, target_column=target_column)
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, meta = preprocessor.prepare_data(df)

        logger.info(
            f"Preprocessing completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}, "
            f"Numeric features: {meta['num_features_count']}, Categorical features: {meta['cat_features_count']}"
        )

        metrics_records = []
        models_to_train = []

        if problem_type == "classification":
            unique_classes = len(np.unique(y_train))
            is_multiclass = unique_classes > 2
            avg_method = 'weighted' if is_multiclass else 'binary'
            
            logger.info(f"Target has {unique_classes} unique classes. Multi-class mode: {is_multiclass}")

            models_to_train = [
                ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), X_train_scaled, X_test_scaled),
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
                ("XGBoost", xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'), X_train, X_test)
            ]

            best_f1 = -1.0
            best_model_name = None

            for model_name, model_obj, X_tr, X_te in models_to_train:
                logger.info(f"--- Training {model_name} ---")
                start_time = time.time()
                try:
                    model_obj.fit(X_tr, y_train)
                    y_pred = model_obj.predict(X_te)
                    exec_time = round(time.time() - start_time, 4)

                    acc = float(accuracy_score(y_test, y_pred))
                    prec = float(precision_score(y_test, y_pred, average=avg_method, zero_division=0))
                    rec = float(recall_score(y_test, y_pred, average=avg_method, zero_division=0))
                    f1 = float(f1_score(y_test, y_pred, average=avg_method, zero_division=0))

                    logger.info(f"Completed {model_name} in {exec_time}s | Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

                    metric_rec = Metric(
                        experiment_id=experiment_id,
                        model_name=model_name,
                        accuracy=acc,
                        precision=prec,
                        recall=rec,
                        f1=f1,
                        training_time=exec_time,
                        model_status="completed"
                    )
                    metrics_records.append(metric_rec)

                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_name = model_name

                except Exception as ex:
                    exec_time = round(time.time() - start_time, 4)
                    logger.error(f"Error training {model_name}: {str(ex)}")
                    metric_rec = Metric(
                        experiment_id=experiment_id,
                        model_name=model_name,
                        training_time=exec_time,
                        model_status="failed",
                        error_message=str(ex)
                    )
                    metrics_records.append(metric_rec)

        else:
            # Regression Models
            logger.info("Setting up regression models...")
            models_to_train = [
                ("Ridge Regression", Ridge(random_state=42), X_train_scaled, X_test_scaled),
                ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test),
                ("XGBoost", xgb.XGBRegressor(n_estimators=100, random_state=42), X_train, X_test)
            ]

            best_r2 = -float('inf')
            best_model_name = None

            for model_name, model_obj, X_tr, X_te in models_to_train:
                logger.info(f"--- Training {model_name} ---")
                start_time = time.time()
                try:
                    model_obj.fit(X_tr, y_train)
                    y_pred = model_obj.predict(X_te)
                    exec_time = round(time.time() - start_time, 4)

                    mse_val = mean_squared_error(y_test, y_pred)
                    rmse = float(np.sqrt(mse_val))
                    mae = float(mean_absolute_error(y_test, y_pred))
                    r2 = float(r2_score(y_test, y_pred))

                    logger.info(f"Completed {model_name} in {exec_time}s | RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

                    metric_rec = Metric(
                        experiment_id=experiment_id,
                        model_name=model_name,
                        rmse=rmse,
                        mae=mae,
                        r2=r2,
                        training_time=exec_time,
                        model_status="completed"
                    )
                    metrics_records.append(metric_rec)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_name = model_name

                except Exception as ex:
                    exec_time = round(time.time() - start_time, 4)
                    logger.error(f"Error training {model_name}: {str(ex)}")
                    metric_rec = Metric(
                        experiment_id=experiment_id,
                        model_name=model_name,
                        training_time=exec_time,
                        model_status="failed",
                        error_message=str(ex)
                    )
                    metrics_records.append(metric_rec)

        # Save metrics to DB
        for m in metrics_records:
            db.add(m)
            
        exp.status = "completed"
        exp.best_model = best_model_name
        exp.completed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Training pipeline finished successfully! Best model: {best_model_name}")

    except Exception as e:
        logger.exception(f"Pipeline execution failed for experiment {experiment_id}: {str(e)}")
        exp = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
        if exp:
            exp.status = "failed"
            exp.error_message = str(e)
            exp.completed_at = datetime.utcnow()
            db.commit()

    finally:
        db.close()
