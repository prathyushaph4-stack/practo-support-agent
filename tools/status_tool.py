"""
status_tool.py
Practo Capstone - Task 7: Wraps check_appointment_status() as a CrewAI tool.
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from lookup_tool import check_appointment_status


class StatusLookupInput(BaseModel):
    record_id: str = Field(..., description="The appointment record ID, e.g. APT0011")


class StatusLookupTool(BaseTool):
    name: str = "check_appointment_status"
    description: str = (
        "Looks up a specific appointment by its record_id and returns its "
        "status, consultation fee, and a designed escalation_score."
    )
    args_schema: type[BaseModel] = StatusLookupInput

    def _run(self, record_id: str) -> str:
        result = check_appointment_status(record_id)
        return str(result)