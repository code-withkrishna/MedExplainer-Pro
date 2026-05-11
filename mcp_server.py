#!/usr/bin/env python3
from __future__ import annotations

import contextvars
import inspect
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.staticfiles import StaticFiles

from medexplainer_pro.agent import MedAgent
from medexplainer_pro.models import debug_print
from medexplainer_pro.tools import (
    analyze_abnormal_values,
    compute_health_risk,
    extract_lab_values,
    generate_explanation,
    generate_trend_analysis,
)

_logger = logging.getLogger(__name__)

_fhir_server_url: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "fhir_server_url",
    default=None,
)
_fhir_access_token: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "fhir_access_token",
    default=None,
)
_patient_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "patient_id",
    default=None,
)


def _ensure_mcp_accept_header(request: Request) -> None:
    if request.method != "POST" or request.url.path.rstrip("/") != "/mcp":
        return

    headers = list(request.scope.get("headers", []))
    accept = next(
        (value for key, value in headers if key.lower() == b"accept"),
        b"",
    )
    if b"application/json" in accept and b"text/event-stream" in accept:
        return

    request.scope["headers"] = [
        (key, value) for key, value in headers if key.lower() != b"accept"
    ]
    request.scope["headers"].append(
        (b"accept", b"application/json, text/event-stream")
    )


class SharpContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        _ensure_mcp_accept_header(request)
        t1 = _fhir_server_url.set(request.headers.get("X-FHIR-Server-URL"))
        t2 = _fhir_access_token.set(request.headers.get("X-FHIR-Access-Token"))
        t3 = _patient_id.set(request.headers.get("X-Patient-ID"))
        try:
            return await call_next(request)
        finally:
            _fhir_server_url.reset(t1)
            _fhir_access_token.reset(t2)
            _patient_id.reset(t3)


def _create_mcp() -> FastMCP:
    if "experimental_capabilities" in inspect.signature(FastMCP).parameters:
        return FastMCP("MedExplainer MCP Server", experimental_capabilities={})
    return FastMCP("MedExplainer MCP Server")


# SHARP context capability advertisement.
# FastMCP does not natively expose a SHARP flag, so we patch the initialization
# options builder to advertise fhir_context_required in the experimental block.
# The patch is hardened with try/except so FastMCP version changes degrade
# gracefully instead of crashing startup.
def _advertise_sharp_capability(server: FastMCP) -> None:
    try:
        low_level_server = getattr(server, "_mcp_server")
        if low_level_server is None or getattr(low_level_server, "_sharp_capability_patched", False):
            return

        if not callable(getattr(low_level_server, "create_initialization_options", None)):
            warnings.warn("SHARP patch skipped: create_initialization_options not callable.")
            return

        original_create_initialization_options = getattr(
            low_level_server,
            "create_initialization_options",
            None,
        )

        def create_initialization_options_with_sharp(*args: Any, **kwargs: Any) -> Any:
            options = original_create_initialization_options(*args, **kwargs)
            capabilities = getattr(options, "capabilities", None)
            if capabilities is None:
                return options

            experimental = getattr(capabilities, "experimental", None)
            if not isinstance(experimental, dict):
                experimental = {}
                setattr(capabilities, "experimental", experimental)

            # The Python MCP SDK validates experimental capabilities as dict values.
            experimental["fhir_context_required"] = {"supported": True}
            return options

        low_level_server.create_initialization_options = create_initialization_options_with_sharp
        low_level_server._sharp_capability_patched = True
    except Exception as exc:
        warnings.warn(
            f"SHARP capability advertisement skipped: {exc}. MCP tools will still work without it."
        )
        return


_ROOT = Path(__file__).resolve().parent
mcp = _create_mcp()
_FRONTEND_DIR = _ROOT / "frontend ui"
_advertise_sharp_capability(mcp)
mcp.tool(extract_lab_values)
mcp.tool(analyze_abnormal_values)
mcp.tool(compute_health_risk)
mcp.tool(generate_trend_analysis)
mcp.tool(generate_explanation)

_SAMPLE_HISTORY = _ROOT / "samples" / "synthetic_patient_history.json"
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": (
        "Content-Type, X-FHIR-Server-URL, X-FHIR-Access-Token, X-Patient-ID"
    ),
}
_SHARP_ASGI_MIDDLEWARE = [Middleware(SharpContextMiddleware)]


def _json_response(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, headers=_CORS_HEADERS)


async def _read_api_payload(request: Request) -> Any:
    """Read API JSON with tolerance for PowerShell here-strings containing raw newlines."""
    body = await request.body()
    if not body:
        return {}

    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            return {"input_text": text}


def get_sharp_context() -> dict:
    return {
        "fhir_server_url": _fhir_server_url.get(),
        "fhir_access_token": _fhir_access_token.get(),
        "patient_id": _patient_id.get(),
    }


# FHIR patient fetch uses SHARP context headers injected by SharpContextMiddleware.
# All failure paths are logged at WARNING level so demo and production runs
# surface integration errors without crashing the MCP tool pipeline.
async def fetch_fhir_patient(patient_id: str) -> dict | None:
    ctx = get_sharp_context()
    if not ctx["fhir_server_url"] or not patient_id:
        return None

    url = f"{ctx['fhir_server_url'].rstrip('/')}/Patient/{patient_id}"
    headers = {"Accept": "application/fhir+json"}
    token = ctx["fhir_access_token"]
    if token and token != "anonymous":
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            else:
                _logger.warning(
                    "FHIR server returned unexpected status %s for patient %s",
                    resp.status_code,
                    patient_id,
                )
    except httpx.TimeoutException as exc:
        _logger.warning("FHIR fetch timed out for patient %s: %s", patient_id, exc)
    except httpx.HTTPStatusError as exc:
        _logger.warning(
            "FHIR server returned HTTP %s for patient %s: %s",
            exc.response.status_code,
            patient_id,
            exc,
        )
    except Exception as exc:
        _logger.warning("FHIR fetch failed for patient %s: %s", patient_id, exc)
    return None


@mcp.tool
async def run_med_agent(input_text: str) -> dict:
    debug_print("Tool called:", "run_med_agent", flush=True)
    ctx = get_sharp_context()
    pipeline_input: dict | list | str = input_text
    patient_id = ctx["patient_id"]
    input_text_value = "" if input_text is None else str(input_text).strip()

    if patient_id and (not input_text_value or input_text_value == patient_id):
        patient = await fetch_fhir_patient(patient_id)
        if patient is not None:
            pipeline_input = patient

    agent = MedAgent()
    result = agent.run(pipeline_input)
    result = result if isinstance(result, dict) else {"result": result}
    if patient_id:
        result["sharp_context"] = {
            "patient_id": patient_id,
            "fhir_server_url": ctx["fhir_server_url"],
        }
    return result


@mcp.custom_route("/api/run", methods=["POST", "OPTIONS"])
async def run_med_agent_http(request: Request) -> Response:
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=_CORS_HEADERS)

    payload = await _read_api_payload(request)
    input_text = payload.get("input_text", payload) if isinstance(payload, dict) else payload
    result = MedAgent().run(input_text)
    status_code = 400 if isinstance(result, dict) and result.get("status") == "error" else 200
    return _json_response(result if isinstance(result, dict) else {"result": result}, status_code=status_code)


@mcp.custom_route("/api/demo", methods=["GET", "OPTIONS"])
async def run_demo_http(request: Request) -> Response:
    if request.method == "OPTIONS":
        return Response(status_code=204, headers=_CORS_HEADERS)

    try:
        payload = json.loads(_SAMPLE_HISTORY.read_text(encoding="utf-8"))
    except OSError as exc:
        result = MedAgent().run(f"Demo sample unavailable: {exc}")
    else:
        result = MedAgent().run(payload)
    return _json_response(result)


# Mount the frontend UI so judges can visit http://localhost:5000 directly.
# StaticFiles with html=True serves index.html for the root path automatically.
if _FRONTEND_DIR.is_dir():
    _frontend_static = StaticFiles(directory=str(_FRONTEND_DIR), html=True)

    # TODO: This FastMCP version's mount() attaches FastMCP subservers rather
    # than Starlette ASGI apps, so serve StaticFiles through custom routes until
    # a Starlette-compatible mcp.mount("/", ..., name="frontend") is available.
    @mcp.custom_route("/", methods=["GET"], name="frontend")
    async def frontend_index(request: Request) -> Response:
        return await _frontend_static.get_response("", request.scope)

    @mcp.custom_route("/{path:path}", methods=["GET"], name="frontend_static")
    async def frontend_static(request: Request) -> Response:
        return await _frontend_static.get_response(request.path_params.get("path", ""), request.scope)


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", "5000"))
    print(f"MedExplainer Pro running at http://localhost:{port}", flush=True)
    print(f"  UI  → http://localhost:{port}/", flush=True)
    print(f"  API → http://localhost:{port}/api/run", flush=True)
    print(f"  MCP → http://localhost:{port}/mcp", flush=True)
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=port,
        middleware=_SHARP_ASGI_MIDDLEWARE,
        stateless_http=True,
    )
