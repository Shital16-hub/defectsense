from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    machine_id: str = Field(..., description="Unique machine identifier, e.g. M001")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    volt: float = Field(..., description="Voltage reading, normal range ~97–251 V")
    rotate: float = Field(..., description="Rotation speed in RPM, normal range ~160–684")
    pressure: float = Field(..., description="Pressure reading, normal range ~51–182")
    vibration: float = Field(..., description="Vibration reading, normal range ~15–72")
    source: str = Field(default="azure_pdm", description="Data source identifier")

    @field_validator("volt")
    @classmethod
    def volt_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Voltage must be positive")
        return v

    @field_validator("rotate")
    @classmethod
    def rotate_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Rotation speed cannot be negative")
        return v

    @field_validator("pressure")
    @classmethod
    def pressure_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Pressure must be positive")
        return v

    @field_validator("vibration")
    @classmethod
    def vibration_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Vibration cannot be negative")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "M001",
            "volt": 176.22,
            "rotate": 418.50,
            "pressure": 113.08,
            "vibration": 45.09,
            "source": "azure_pdm",
        }
    }}


class SensorBatch(BaseModel):
    readings: list[SensorReading]
    batch_id: Optional[str] = None
