from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


FailureType = Literal["TWF", "HDF", "PWF", "OSF", "RNF", "NONE"]


class AnomalyResult(BaseModel):
    machine_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalised anomaly score 0→1")
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of failure")
    is_anomaly: bool
    failure_type_prediction: Optional[FailureType] = None
    sensor_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Per-sensor deviation from normal baseline (z-score)",
    )
    ml_model_used: str = Field(
        default="lstm_autoencoder",
        description="Which model(s) flagged this: lstm_autoencoder | isolation_forest | ensemble",
    )
    reconstruction_error: Optional[float] = Field(
        None, description="LSTM reconstruction error (MSE) — raw value before thresholding"
    )
    isolation_score: Optional[float] = Field(
        None, description="Isolation Forest raw decision score"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "M001",
            "anomaly_score": 0.87,
            "failure_probability": 0.74,
            "is_anomaly": True,
            "failure_type_prediction": "HDF",
            "sensor_deltas": {
                "air_temperature": 2.1,
                "process_temperature": 3.8,
                "rotational_speed": -0.4,
                "torque": 1.2,
                "tool_wear": 0.9,
            },
            "ml_model_used": "ensemble",
        }
    }}
