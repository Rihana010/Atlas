from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id = Column(String, primary_key=True, index=True)
    dataset_name = Column(String, nullable=False)
    dataset_path = Column(String, nullable=False)
    target_column = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    problem_type = Column(String, nullable=True)
    reasoning = Column(Text, nullable=True)
    pipeline_version = Column(String, default="1.0.0", nullable=False)
    best_model = Column(String, nullable=True)
    status = Column(String, default="uploaded", nullable=False)  # uploaded, queued, running, completed, failed
    error_message = Column(Text, nullable=True)

    metrics = relationship("Metric", back_populates="experiment", cascade="all, delete-orphan")

class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.experiment_id"), nullable=False, index=True)
    model_name = Column(String, nullable=False)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    training_time = Column(Float, nullable=True)
    model_status = Column(String, default="completed", nullable=False)  # completed, failed
    error_message = Column(Text, nullable=True)

    experiment = relationship("Experiment", back_populates="metrics")
