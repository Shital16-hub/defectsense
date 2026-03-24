from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class MaintenanceLog(BaseModel):
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str
    date: datetime
    failure_type: str = Field(
        ...,
        description="TWF | HDF | PWF | OSF | RNF",
    )
    symptoms: str = Field(..., description="Observable symptoms before/during failure")
    root_cause: str = Field(..., description="Identified root cause")
    action_taken: str = Field(..., description="Maintenance action performed")
    resolution_time_hours: float = Field(..., ge=0.0)
    technician: str
    machine_type: Optional[str] = Field(None, description="Machine variant: L | M | H")
    notes: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "M042",
            "date": "2024-03-15T08:30:00",
            "failure_type": "HDF",
            "symptoms": "Process temperature exceeded 315K, rotational speed dropped below 1400 RPM",
            "root_cause": "Cooling fan blade fracture causing heat dissipation failure",
            "action_taken": "Replaced cooling fan assembly, cleaned heat exchange fins",
            "resolution_time_hours": 4.5,
            "technician": "J. Smith",
        }
    }}
