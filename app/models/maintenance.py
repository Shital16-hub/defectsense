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
        description="EVF | RSF | PBF | BWF",
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
            "date": "2015-03-15T08:30:00",
            "failure_type": "BWF",
            "symptoms": "Vibration level 10 units above normal baseline, bearing wear suspected",
            "root_cause": "Rolling element bearing fatigue causing increased vibration signature",
            "action_taken": "Replaced worn bearing assembly and verified alignment",
            "resolution_time_hours": 4.5,
            "technician": "J. Smith",
        }
    }}
