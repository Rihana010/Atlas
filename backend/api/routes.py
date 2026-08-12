import io
import zipfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.session import get_db
from backend.database.models import Experiment, Metric
from backend.schemas.experiment import (
    UploadResponse, DatasetResponse, TrainRequest, TrainResponse, ResultsResponse, MetricSchema
)
from backend.services.experiment_service import (
    generate_experiment_id, save_uploaded_csv, analyze_dataset_file, start_training_job
)

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing."
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        filename = file.filename
        
        # Check if file is a ZIP archive
        if content.startswith(b"PK\x03\x04") or filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv") and not m.startswith("__MACOSX")]
                    if not csv_members:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Uploaded ZIP file does not contain any .csv file."
                        )
                    # Extract first CSV file inside zip
                    extracted_name = csv_members[0]
                    content = zf.read(extracted_name)
                    filename = extracted_name
            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or corrupted ZIP archive."
                )

        saved_path = save_uploaded_csv(content, filename)
        
        # Analyze dataset to ensure it can be parsed
        analysis = analyze_dataset_file(saved_path)

        exp_id = generate_experiment_id(db)
        experiment = Experiment(
            experiment_id=exp_id,
            dataset_name=filename,
            dataset_path=saved_path,
            status="uploaded"
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        return UploadResponse(
            experiment_id=experiment.experiment_id,
            status=experiment.status,
            dataset_name=experiment.dataset_name,
            total_rows=analysis["total_rows"],
            total_columns=analysis["total_columns"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process CSV file: {str(e)}"
        )

@router.get("/dataset/{experiment_id}", response_model=DatasetResponse)
def get_dataset_info(experiment_id: str, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found."
        )

    try:
        analysis = analyze_dataset_file(exp.dataset_path)
        return DatasetResponse(
            experiment_id=exp.experiment_id,
            dataset_name=exp.dataset_name,
            total_rows=analysis["total_rows"],
            total_columns=analysis["total_columns"],
            columns=analysis["columns"],
            preview=analysis["preview"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing dataset: {str(e)}"
        )

@router.post("/train", response_model=TrainResponse)
def train_experiment(req: TrainRequest, db: Session = Depends(get_db)):
    try:
        exp = start_training_job(db, req.experiment_id, req.target_column)
        return TrainResponse(
            experiment_id=exp.experiment_id,
            status="running",
            target_column=exp.target_column,
            problem_type=exp.problem_type,
            reasoning=exp.reasoning
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/results/{experiment_id}", response_model=ResultsResponse)
def get_experiment_results(experiment_id: str, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found."
        )

    metrics_list = []
    for m in exp.metrics:
        metrics_list.append(MetricSchema(
            model_name=m.model_name,
            accuracy=m.accuracy,
            precision=m.precision,
            recall=m.recall,
            f1=m.f1,
            rmse=m.rmse,
            mae=m.mae,
            r2=m.r2,
            training_time=m.training_time,
            model_status=m.model_status,
            error_message=m.error_message
        ))

    return ResultsResponse(
        experiment_id=exp.experiment_id,
        status=exp.status,
        dataset_name=exp.dataset_name,
        target_column=exp.target_column,
        problem_type=exp.problem_type,
        reasoning=exp.reasoning,
        best_model=exp.best_model,
        error_message=exp.error_message,
        created_at=exp.created_at.isoformat() if exp.created_at else "",
        started_at=exp.started_at.isoformat() if exp.started_at else None,
        completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
        metrics=metrics_list
    )
