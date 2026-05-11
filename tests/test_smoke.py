from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from starlette.requests import Request

from mcp_server import run_med_agent, run_med_agent_http


PLAIN_TEXT_LABS = """Age: 52
Gender: Male
Hemoglobin: 10.2 g/dL
Glucose: 180 mg/dL (High)
"""


JSON_PATIENT = {
    "patient": {
        "id": "smoke-json",
        "demographics": {"age": 52, "gender": "Male"},
        "observations": [
            {
                "code": "glucose",
                "display_name": "Glucose",
                "value": 180,
                "unit": "mg/dL",
                "reference_range": [70, 99],
            }
        ],
    }
}


@pytest.fixture(autouse=True)
def disable_remote_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDEXPLAINER_ALLOW_REMOTE_REASONING", "false")
    monkeypatch.setenv("MEDEXPLAINER_ALLOW_LLM_PHI", "false")


def _request_from_body(body: bytes, content_type: str = "application/json") -> Request:
    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/run",
            "headers": [(b"content-type", content_type.encode("utf-8"))],
        },
        receive,
    )


def _json_request(payload: Any) -> Request:
    body = json.dumps(payload).encode("utf-8")
    return _request_from_body(body)


def _response_json(response: Any) -> dict[str, Any]:
    return json.loads(response.body.decode("utf-8"))


def test_api_run_accepts_plain_text_labs() -> None:
    response = asyncio.run(run_med_agent_http(_json_request({"input_text": PLAIN_TEXT_LABS})))
    payload = _response_json(response)

    assert response.status_code == 200
    assert payload.get("status") != "error"
    assert "health_risk_score" in payload
    assert "clinical_insight" in payload


def test_api_run_accepts_json_patient_payload() -> None:
    response = asyncio.run(run_med_agent_http(_json_request(JSON_PATIENT)))
    payload = _response_json(response)

    assert response.status_code == 200
    assert "health_risk_score" in payload
    assert payload["risk_level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_api_run_tolerates_raw_newlines_inside_json_string() -> None:
    body = b"""{
  "input_text": "Age: 54
Gender: Female
Hemoglobin: 9.8 g/dL (Low)
Glucose: 192 mg/dL (High)
HbA1c: 7.4%
Creatinine: 1.6 mg/dL (High)"
}
"""
    response = asyncio.run(run_med_agent_http(_request_from_body(body)))
    payload = _response_json(response)

    assert response.status_code == 200
    assert payload["health_risk_score"] > 0
    assert payload["risk_level"] in {"MODERATE", "HIGH", "CRITICAL"}


def test_mcp_run_med_agent_tool_accepts_plain_text_labs() -> None:
    payload = asyncio.run(run_med_agent(PLAIN_TEXT_LABS))

    assert "health_risk_score" in payload
    assert "recommended_actions" in payload
