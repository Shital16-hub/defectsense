from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


FailureType = Literal["EVF", "RSF", "PBF", "BWF", "NONE"]
# EVF = Electrical Overload Failure  (high volt)
# RSF = Rotor Speed Failure          (low rotate)
# PBF = Pressure Blockage Failure    (high pressure)
# BWF = Bearing Wear Failure         (high vibration)


class AnomalyResult(BaseModel):
    machine_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalised anomaly score 0->1")
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of failure")
    is_anomaly: bool
    failure_type_prediction: Optional[FailureType] = None
    sensor_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Per-sensor deviation from normal baseline (z-score)",
    )
    ml_model_used: str = Field(
        default="lstm_autoencoder",
        description="Which model(s) flagged this: lstm_autoencoder | ensemble | none",
    )
    reconstruction_error: Optional[float] = Field(
        None, description="LSTM reconstruction error (MSE) -- raw value before thresholding"
    )
    isolation_score: Optional[float] = Field(
        None, description="Raw anomaly decision score (reserved, currently unused)"
    )
    most_anomalous_sensor: Optional[str] = Field(
        None, description="Sensor name with the highest deviation from baseline"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "M001",
            "anomaly_score": 0.87,
            "failure_probability": 0.74,
            "is_anomaly": True,
            "failure_type_prediction": "PBF",
            "sensor_deltas": {
                "volt": 0.1,
                "rotate": -0.3,
                "pressure": 2.1,
                "vibration": 0.4,
            },
            "ml_model_used": "ensemble",
        }
    }}
