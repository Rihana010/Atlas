from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    experiment_id: str
    status: str
    dataset_name: str
    total_rows: int
    total_columns: int

class ColumnSummary(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any]

class DatasetResponse(BaseModel):
    experiment_id: str
    dataset_name: str
    total_rows: int
    total_columns: int
    columns: List[ColumnSummary]
    preview: List[Dict[str, Any]]

class TrainRequest(BaseModel):
    experiment_id: str
    target_column: str

class TrainResponse(BaseModel):
    experiment_id: str
    status: str
    target_column: str
    problem_type: str
    reasoning: str

class MetricSchema(BaseModel):
    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    training_time: Optional[float] = None
    model_status: str
    error_message: Optional[str] = None

class ResultsResponse(BaseModel):
    experiment_id: str
    status: str
    dataset_name: str
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    reasoning: Optional[str] = None
    best_model: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: List[MetricSchema] = []
