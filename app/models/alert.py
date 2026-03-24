from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field
import uuid

from app.models.anomaly import AnomalyResult
from app.models.maintenance import MaintenanceLog


Severity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class RootCauseReport(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str
    anomaly_result: AnomalyResult
    similar_incidents: list[MaintenanceLog] = Field(default_factory=list)
    root_cause: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence statements")
    recommended_actions: list[str] = Field(
        default_factory=list, description="Ranked maintenance actions"
    )
    severity: Severity
    agent_memory_used: list[str] = Field(
        default_factory=list, description="A-MEM memory notes recalled during reasoning"
    )
    reasoning_steps: list[str] = Field(
        default_factory=list, description="ReAct reasoning trace (THINK / ACT / OBSERVE / CONCLUDE)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "M001",
            "root_cause": "Heat dissipation failure caused by degraded cooling system",
            "confidence": 0.88,
            "severity": "HIGH",
            "evidence": [
                "Process temperature 3.8 std deviations above normal",
                "Similar incident on M001 in Jan 2024 — cooling fan failure",
                "Tool wear at 180 min (threshold: 200 min)",
            ],
            "recommended_actions": [
                "Inspect and replace cooling fan assembly",
                "Schedule preventive maintenance within 4 hours",
            ],
        }
    }}


class MaintenanceAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    machine_id: str
    root_cause_report: RootCauseReport
    plain_language_explanation: str = Field(
        ..., description="Non-technical explanation for factory floor workers"
    )
    approved: Optional[bool] = Field(None, description="None=pending, True=approved, False=rejected")
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = None
    auto_approved: bool = Field(
        False, description="True if confidence exceeded auto-approve threshold"
    )

    @property
    def is_pending(self) -> bool:
        return self.approved is None

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "M001",
            "plain_language_explanation": (
                "Machine M001 is showing signs of overheating. "
                "The cooling system may be failing. "
                "Please stop the machine and call the maintenance team now."
            ),
            "approved": None,
        }
    }}
