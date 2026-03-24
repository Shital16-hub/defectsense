from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    machine_id: str = Field(..., description="Unique machine identifier, e.g. M001")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    air_temperature: float = Field(..., description="Air temperature in Kelvin")
    process_temperature: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed: float = Field(..., description="Rotational speed in RPM")
    torque: float = Field(..., description="Torque in Nm")
    tool_wear: float = Field(..., description="Tool wear in minutes")
    source: str = Field(default="ai4i", description="Data source identifier")

    @field_validator("air_temperature", "process_temperature")
    @classmethod
    def temperature_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Temperature must be positive (Kelvin)")
        return v

    @field_validator("rotational_speed")
    @classmethod
    def speed_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Rotational speed cannot be negative")
        return v

    @field_validator("tool_wear")
    @classmethod
    def tool_wear_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Tool wear cannot be negative")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "M001",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551.0,
            "torque": 42.8,
            "tool_wear": 0.0,
            "source": "ai4i",
        }
    }}


class SensorBatch(BaseModel):
    readings: list[SensorReading]
    batch_id: Optional[str] = None
