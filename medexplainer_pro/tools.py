from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Literal

from .fhir import patient_from_any, patient_from_dict, reports_from_any
from .models import (
    AgentConfig,
    FHIRPatient,
    MCPToolDefinition,
    ObservationFinding,
    ReferenceRange,
    ReasoningLogger,
    RiskAssessment,
    debug_print,
)
from .reasoning import HybridReasoner
from .trends import ClinicalTrendAnalyzer


Status = Literal["LOW", "NORMAL", "HIGH", "UNKNOWN"]
RiskLevel = Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]

_CRITICAL_PARAMETERS = (
    "hemoglobin",
    "potassium",
    "sodium",
    "creatinine",
    "blood sugar",
    "glucose",
    "troponin",
    "wbc",
)
_MODERATE_PARAMETERS = (
    "cholesterol",
    "ldl",
    "triglycerides",
    "hba1c",
    "platelet count",
    "egfr",
)
_TRAILING_STATUS_PATTERN = re.compile(r"[\s:,-]*(?:\(|\[)?\b(high|low|normal|unknown)\b(?:\)|\])?\s*$", re.IGNORECASE)


def _status_text(status: object) -> str:
    return str(status or "").strip().lower()


def _has_status(status: object, expected: str) -> bool:
    return expected.lower() in _status_text(status)


def _is_low_status(status: object) -> bool:
    return _has_status(status, "low")


def _is_high_status(status: object) -> bool:
    return _has_status(status, "high")


def _is_unknown_status(status: object) -> bool:
    return _has_status(status, "unknown")


def _is_normal_status(status: object) -> bool:
    return _has_status(status, "normal") and not _is_low_status(status) and not _is_high_status(status)


def _is_abnormal_status(status: object) -> bool:
    return _is_low_status(status) or _is_high_status(status)


def _status_label(status: object) -> str:
    if _is_high_status(status):
        return "HIGH"
    if _is_low_status(status):
        return "LOW"
    if _is_normal_status(status):
        return "NORMAL"
    if _is_unknown_status(status):
        return "UNKNOWN"
    return str(status or "UNKNOWN").strip().upper() or "UNKNOWN"


def _status_from_label(label: object) -> str:
    match = _TRAILING_STATUS_PATTERN.search(str(label or "").strip())
    return match.group(1).upper() if match else "UNKNOWN"


def _strip_status_from_label(label: object) -> str:
    text = str(label or "").strip()
    return _TRAILING_STATUS_PATTERN.sub("", text).strip() or text or "Observation"


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_parameter(value: object, low: object | None, high: object | None) -> Status:
    numeric_value = _to_float(value)
    numeric_low = _to_float(low)
    numeric_high = _to_float(high)
    if numeric_value is None:
        return "UNKNOWN"
    if numeric_low is None and numeric_high is None:
        return "UNKNOWN"
    if numeric_low is None:
        return "HIGH" if numeric_high is not None and numeric_value > numeric_high else "NORMAL"
    if numeric_high is None:
        return "LOW" if numeric_value < numeric_low else "NORMAL"
    if numeric_value < numeric_low:
        return "LOW"
    if numeric_value > numeric_high:
        return "HIGH"
    return "NORMAL"


def analyze_parameters(parameters: list[dict]) -> list[dict]:
    findings = []
    for param in parameters:
        findings.append(
            {
                **param,
                "status": _classify_parameter(
                    value=param.get("value"),
                    low=param.get("low"),
                    high=param.get("high"),
                ),
            }
        )
    return findings


def _normalise_name(name: object) -> str:
    normalized = str(name or "").strip().lower()
    for char in ("-", "_", "/", "(", ")", ","):
        normalized = normalized.replace(char, " ")
    return " ".join(normalized.split())


def _finding_name(finding: dict) -> str:
    return str(finding.get("code") or finding.get("display_name") or finding.get("name") or "")


def _matches_parameter(name: object, aliases: tuple[str, ...]) -> bool:
    normalized = _normalise_name(name)
    tokens = set(normalized.split())
    for alias in aliases:
        normalized_alias = _normalise_name(alias)
        if len(normalized_alias) <= 3 and normalized_alias.isalpha():
            if normalized_alias in tokens:
                return True
            continue
        if normalized_alias in normalized:
            return True
    return False


# ---------------------------------------------------------------------------
# Severity weight design rationale
# ---------------------------------------------------------------------------
# Weights are a prototype heuristic inspired by clinical triage prioritisation,
# not a validated scoring instrument. They intentionally mimic the direction
# (not the magnitude) of established urgency tiers used in systems such as
# APACHE II and SOFA, where electrolyte and haemoglobin derangements carry
# higher mortality correlation than isolated lipid abnormalities.
#
# Critical parameters (weight 3): haemoglobin, potassium, sodium, creatinine,
#   glucose, troponin, WBC — derangements in these markers are associated with
#   acute life-threatening conditions in standard emergency triage literature.
# Moderate parameters (weight 2): cholesterol, LDL, triglycerides, HbA1c,
#   platelets, eGFR — chronic disease markers where trends matter more than
#   single-point values.
# Other parameters (weight 1): everything else; flagged but lower urgency.
#
# Deviation bonuses (+1 at ≥10 %, +2 at ≥25 %) are based on the common
# clinical rule of thumb that a value >25 % outside its reference boundary
# warrants prompt attention regardless of marker type (Lundberg 1972,
# "panic value" concept, CLSI EP23-A).
# ---------------------------------------------------------------------------
def severity_weight_for(code_or_name: object, status: object, deviation_percent: object = 0) -> int:
    if not _is_abnormal_status(status):
        return 0

    if _matches_parameter(code_or_name, _CRITICAL_PARAMETERS):
        base_weight = 3
    elif _matches_parameter(code_or_name, _MODERATE_PARAMETERS):
        base_weight = 2
    else:
        base_weight = 1

    try:
        deviation = float(deviation_percent)
    except (TypeError, ValueError):
        deviation = 0.0
    if deviation >= 25:
        return base_weight + 2
    if deviation >= 10:
        return base_weight + 1
    return base_weight


def _severity_weight(finding: dict) -> int:
    return severity_weight_for(
        _finding_name(finding),
        finding.get("status"),
        finding.get("deviation_percent", 0),
    )


def _level_from_health_score(score: int) -> RiskLevel:
    if score >= 70:
        return "CRITICAL"
    if score >= 45:
        return "HIGH"
    if score >= 20:
        return "MODERATE"
    return "LOW"


def _finding_label(finding: dict) -> str:
    return str(finding.get("display_name") or finding.get("name") or finding.get("code") or "Observation")


def _finding_factor(finding: dict, severity_weight: int) -> dict:
    value = finding.get("value")
    unit = finding.get("unit", "")
    status = _status_label(finding.get("status", "UNKNOWN"))
    return {
        "factor": f"{_finding_label(finding)}: {status} ({value} {unit})".strip(),
        "impact": severity_weight * 8,
    }


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_final_risk(findings: list[dict], patient: dict | None = None) -> dict:
    unknown_findings = [
        finding for finding in findings
        if _is_unknown_status(finding.get("status", ""))
    ]
    abnormal_findings = [
        finding for finding in findings
        if _is_abnormal_status(finding.get("status", "NORMAL"))
    ]
    severity_weights = [_severity_weight(finding) for finding in abnormal_findings]
    severity_score = sum(severity_weights)
    abnormal_count = len(abnormal_findings)
    total_count = len(findings)
    abnormal_ratio = abnormal_count / max(total_count, 1)

    risk_breakdown = [
        _finding_factor(finding, severity_weight)
        for finding, severity_weight in zip(abnormal_findings, severity_weights)
        if severity_weight > 0
    ]

    abnormal_ratio_impact = round(abnormal_ratio * 35, 2)
    if abnormal_ratio_impact:
        risk_breakdown.append(
            {
                "factor": f"Abnormal marker burden: {abnormal_count} of {total_count}",
                "impact": abnormal_ratio_impact,
            }
        )

    extreme_deviation_bonus = sum(
        6
        for finding in abnormal_findings
        if _safe_float(finding.get("deviation_percent")) >= 25
    )
    if extreme_deviation_bonus:
        risk_breakdown.append({"factor": "Large deviation from reference range", "impact": extreme_deviation_bonus})

    pattern_bonus = 8 if abnormal_count >= 3 else 0
    if pattern_bonus:
        risk_breakdown.append({"factor": "Multiple abnormal markers present together", "impact": pattern_bonus})

    for finding in unknown_findings:
        risk_breakdown.append(
            {
                "factor": f"Reference range unavailable for {_finding_label(finding)}; marker was not scored",
                "impact": 0,
            }
        )

    # Score formula (prototype weights — not a validated clinical instrument):
    #   severity_score * 8   : per-marker urgency (see severity_weight_for comments)
    #   abnormal_ratio * 35  : burden adjustment — many concurrent abnormalities
    #                          increase clinical concern beyond individual weights
    #                          (reflects the "multi-organ dysfunction" concept)
    #   extreme_deviation_bonus : extra signal for values far outside range
    #   pattern_bonus (8)    : flat bonus when ≥3 abnormal markers co-occur,
    #                          consistent with multi-marker risk amplification
    #                          seen in composite risk indices (e.g. Framingham,
    #                          CHADS2-VASc use additive multi-factor scoring)
    # Cap at 100 to keep the output on a familiar 0–100 percentage scale.
    score = round(
        min(
            100,
            severity_score * 8
            + abnormal_ratio_impact
            + extreme_deviation_bonus
            + pattern_bonus,
        )
    )
    risk_level = _level_from_health_score(score)
    contributing_factors = [
        item["factor"]
        for item in sorted(risk_breakdown, key=lambda item: item["impact"], reverse=True)
        if ":" in item["factor"]
    ][:5]

    if not risk_breakdown:
        risk_breakdown.append({"factor": "No scored abnormalities detected in the supplied observations", "impact": 0})

    warnings = []
    if unknown_findings:
        warnings.append("Some observations lack reference ranges; interpretation may be limited")

    return {
        "risk_level": risk_level,
        "health_risk_score": score,
        "abnormal_count": abnormal_count,
        "total_count": total_count,
        "severity_score": severity_score,
        "contributing_factors": contributing_factors,
        "risk_breakdown": risk_breakdown,
        "legacy_risk_level": None,
        "warnings": warnings,
    }


PATIENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "demographics": {"type": "object"},
        "source": {"type": "string"},
        "timestamp": {"type": ["string", "null"]},
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "display_name": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "reference_range": {
                        "type": "array",
                        "items": {"type": ["number", "null"]},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "reference_range_detail": {"type": "object"},
                    "referenceRange": {"type": ["array", "object"]},
                    "status": {"type": "string"},
                },
                "required": ["code", "value"],
            },
        },
    },
    "required": ["id", "observations"],
}

FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "display_name": {"type": "string"},
        "value": {"type": "number"},
        "unit": {"type": "string"},
        "reference_range": {"type": "array", "items": {"type": ["number", "null"]}},
        "reference_range_detail": {"type": "object"},
        "referenceRange": {"type": ["array", "object"]},
        "status": {"type": "string"},
        "deviation_percent": {"type": "number"},
        "severity_weight": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["code", "display_name", "value", "status"],
}

RISK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "string"},
        "health_risk_score": {"type": "integer"},
        "abnormal_count": {"type": "integer"},
        "total_count": {"type": "integer"},
        "severity_score": {"type": "integer"},
        "contributing_factors": {"type": "array", "items": {"type": "string"}},
        "risk_breakdown": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "factor": {"type": "string"},
                    "impact": {"type": "number"},
                },
                "required": ["factor", "impact"],
            },
        },
        "legacy_risk_level": {"type": ["string", "null"]},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "risk_level",
        "health_risk_score",
        "abnormal_count",
        "total_count",
        "severity_score",
        "risk_breakdown",
    ],
}

EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patient_summary": {"type": "string"},
        "clinical_insight": {"type": "string"},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "explanation_confidence": {"type": "number"},
        "pattern_signals": {"type": "array", "items": {"type": "string"}},
        "safety_disclaimer": {"type": "string"},
        "reasoning_mode": {"type": "string"},
    },
    "required": [
        "patient_summary",
        "clinical_insight",
        "recommended_actions",
        "explanation_confidence",
        "pattern_signals",
        "safety_disclaimer",
        "reasoning_mode",
    ],
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "timestamp": {"type": ["string", "null"]},
        "demographics": {"type": "object"},
        "source": {"type": "string"},
        "observations": PATIENT_SCHEMA["properties"]["observations"],
    },
    "required": ["observations"],
}

MARKER_TREND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "display_name": {"type": "string"},
        "timestamps": {"type": "array", "items": {"type": "string"}},
        "values": {"type": "array", "items": {"type": "number"}},
        "unit": {"type": "string"},
        "direction": {"type": "string"},
        "severity_change": {"type": "string"},
        "status_path": {"type": "array", "items": {"type": "string"}},
        "relative_change_percent": {"type": "number"},
        "clinical_interpretation": {"type": "string"},
        "risk_delta": {"type": "integer"},
        "step_deltas": {"type": "array", "items": {"type": "number"}},
        "avg_delta": {"type": "number"},
        "abs_avg_delta": {"type": "number"},
        "volatility": {"type": "number"},
        "stability": {"type": "string"},
    },
    "required": [
        "code",
        "display_name",
        "timestamps",
        "values",
        "direction",
        "severity_change",
        "clinical_interpretation",
        "risk_delta",
        "avg_delta",
        "volatility",
        "stability",
    ],
}

TREND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "trend_summary": {"type": "string"},
        "marker_trends": {"type": "array", "items": MARKER_TREND_SCHEMA},
        "overall_direction": {"type": "string"},
        "risk_delta": {"type": "integer"},
        "trajectory_score": {"type": "integer"},
        "pattern_signals": {"type": "array", "items": {"type": "string"}},
        "safety_disclaimer": {"type": "string"},
        "trend_confidence": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": [
        "trend_summary",
        "marker_trends",
        "overall_direction",
        "risk_delta",
        "trajectory_score",
        "pattern_signals",
        "safety_disclaimer",
    ],
}


@dataclass(slots=True)
class MCPStyleTool:
    definition: MCPToolDefinition
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.handler(payload)


class ToolRegistry:
    """Registry of agent-callable MCP-style tools."""

    def __init__(self, config: AgentConfig, logger: ReasoningLogger) -> None:
        self.config = config
        self.logger = logger
        self.reasoner = HybridReasoner(config)
        self.trend_analyzer = ClinicalTrendAnalyzer()
        self._tools: dict[str, MCPStyleTool] = {}
        self._register_defaults()

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition.to_dict() for tool in self._tools.values()]

    def call(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        self.logger.log("tool.call", f"Invoking {name}.", {"tool": name})
        return self._tools[name].invoke(payload)

    def _register(self, tool: MCPStyleTool) -> None:
        self._tools[tool.definition.name] = tool

    def _register_defaults(self) -> None:
        self._register(
            MCPStyleTool(
                definition=MCPToolDefinition(
                    name="extract_lab_values_tool",
                    title="Extract Lab Values",
                    description="Normalize raw report text, a FHIR resource, or a patient JSON payload into the internal FHIR-style patient structure.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string"},
                            "report_text": {"type": "string"},
                            "fhir_resource": {"type": ["object", "array"]},
                            "patient": PATIENT_SCHEMA,
                        },
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "patient": PATIENT_SCHEMA,
                            "observation_count": {"type": "integer"},
                            "normalization_notes": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["patient", "observation_count", "normalization_notes"],
                    },
                    annotations={"readOnlyHint": True, "idempotentHint": True},
                ),
                handler=self._extract_lab_values,
            )
        )
        self._register(
            MCPStyleTool(
                definition=MCPToolDefinition(
                    name="analyze_values_tool",
                    title="Analyze Values",
                    description="Classify every observation as LOW, NORMAL, or HIGH and identify abnormal markers.",
                    input_schema={
                        "type": "object",
                        "properties": {"patient": PATIENT_SCHEMA},
                        "required": ["patient"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string"},
                            "findings": {"type": "array", "items": FINDING_SCHEMA},
                            "abnormal_values": {"type": "array", "items": FINDING_SCHEMA},
                            "abnormal_count": {"type": "integer"},
                            "total_count": {"type": "integer"},
                            "warnings": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["patient_id", "findings", "abnormal_values", "abnormal_count", "total_count", "warnings"],
                    },
                    annotations={"readOnlyHint": True, "idempotentHint": True},
                ),
                handler=self._analyze_values,
            )
        )
        self._register(
            MCPStyleTool(
                definition=MCPToolDefinition(
                    name="compute_risk_tool",
                    title="Compute Risk",
                    description="Combine abnormal findings into a risk tier and a 0-100 Health Risk Score.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "patient": PATIENT_SCHEMA,
                            "findings": {"type": "array", "items": FINDING_SCHEMA},
                        },
                        "required": ["patient", "findings"],
                    },
                    output_schema=RISK_SCHEMA,
                    annotations={"readOnlyHint": True, "idempotentHint": True},
                ),
                handler=self._compute_risk,
            )
        )
        self._register(
            MCPStyleTool(
                definition=MCPToolDefinition(
                    name="generate_explanation_tool",
                    title="Generate Explanation",
                    description="Use the reasoning layer to summarize abnormalities, pattern signals, risk interpretation, and safe next-step suggestions.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "patient": PATIENT_SCHEMA,
                            "findings": {"type": "array", "items": FINDING_SCHEMA},
                            "risk": RISK_SCHEMA,
                            "user_question": {"type": "string"},
                        },
                        "required": ["patient", "findings", "risk"],
                    },
                    output_schema=EXPLANATION_SCHEMA,
                    annotations={"readOnlyHint": True, "idempotentHint": True},
                ),
                handler=self._generate_explanation,
            )
        )
        self._register(
            MCPStyleTool(
                definition=MCPToolDefinition(
                    name="analyze_trends_tool",
                    title="Analyze Trends",
                    description="Analyze longitudinal marker trajectories across two or more reports, including direction, severity change, risk delta, and cautious clinical trend interpretation.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "reports": {"type": "array", "items": REPORT_SCHEMA},
                        },
                        "required": ["reports"],
                    },
                    output_schema=TREND_SCHEMA,
                    annotations={"readOnlyHint": True, "idempotentHint": True},
                ),
                handler=self._analyze_trends,
            )
        )

    def _extract_lab_values(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("patient") is not None:
            patient = patient_from_dict({"patient": payload["patient"]})
            note = "Structured patient payload normalized to internal FHIR-style patient model."
        elif payload.get("fhir_resource") is not None:
            patient = patient_from_any(payload["fhir_resource"], patient_id=payload.get("patient_id"))
            note = "FHIR resource converted into internal FHIR-style patient model."
        elif payload.get("report_text"):
            patient = patient_from_any(str(payload["report_text"]), patient_id=payload.get("patient_id"))
            note = "Raw report text extracted into internal FHIR-style patient model."
        else:
            raise ValueError("Supply patient, fhir_resource, or report_text.")

        notes = [note]
        if not patient.observations:
            notes.append("No observations were detected.")
        if any(
            observation.reference_range.low is None and observation.reference_range.high is None
            for observation in patient.observations
        ):
            notes.append("Some observations are missing numeric reference ranges.")

        return {
            "patient": patient.to_dict(),
            "observation_count": len(patient.observations),
            "normalization_notes": notes,
        }

    def _analyze_values(self, payload: dict[str, Any]) -> dict[str, Any]:
        patient = patient_from_dict({"patient": payload["patient"]})
        parameter_rows = [
            {
                "name": observation.display_name or observation.code,
                "value": observation.value,
                "unit": observation.unit,
                "low": observation.reference_range.low,
                "high": observation.reference_range.high,
            }
            for observation in patient.observations
        ]
        analyzed_rows = analyze_parameters(parameter_rows)
        findings: list[ObservationFinding] = []

        for observation, analyzed_row in zip(patient.observations, analyzed_rows):
            status = _status_label(observation.status or analyzed_row["status"])
            deviation_percent = round(self._deviation_percent(observation.value, observation.reference_range, status), 2)
            severity_weight = self._severity_weight(observation.code, status, deviation_percent)
            findings.append(
                ObservationFinding(
                    code=observation.code,
                    display_name=observation.display_name or observation.code,
                    value=observation.value,
                    unit=observation.unit,
                    reference_range=observation.reference_range,
                    status=status,
                    deviation_percent=deviation_percent,
                    severity_weight=severity_weight,
                    rationale=self._build_rationale(observation.reference_range, status),
                )
            )

        abnormal_findings = [finding.to_dict() for finding in findings if _is_abnormal_status(finding.status)]
        warnings = []
        if any(_is_unknown_status(finding.status) for finding in findings):
            warnings.append("Some observations lack reference ranges; interpretation may be limited")
        return {
            "patient_id": patient.id,
            "findings": [finding.to_dict() for finding in findings],
            "abnormal_values": abnormal_findings,
            "abnormal_count": len(abnormal_findings),
            "total_count": len(findings),
            "warnings": warnings,
        }

    def _compute_risk(self, payload: dict[str, Any]) -> dict[str, Any]:
        patient = patient_from_dict({"patient": payload["patient"]})
        findings = [self._finding_from_dict(finding) for finding in payload["findings"]]
        return compute_final_risk(
            [finding.to_dict() for finding in findings],
            patient=patient.to_dict(),
        )

    def _generate_explanation(self, payload: dict[str, Any]) -> dict[str, Any]:
        patient = patient_from_dict({"patient": payload["patient"]})
        findings = [self._finding_from_dict(finding) for finding in payload["findings"]]
        risk = RiskAssessment(
            risk_level=payload["risk"]["risk_level"],
            health_risk_score=int(payload["risk"]["health_risk_score"]),
            abnormal_count=int(payload["risk"]["abnormal_count"]),
            total_count=int(payload["risk"]["total_count"]),
            severity_score=int(payload["risk"]["severity_score"]),
            contributing_factors=list(payload["risk"].get("contributing_factors", [])),
            risk_breakdown=list(payload["risk"].get("risk_breakdown", [])),
            legacy_risk_level=payload["risk"].get("legacy_risk_level"),
        )
        reasoning = self.reasoner.generate(
            patient=patient,
            findings=findings,
            risk=risk,
            logger=self.logger,
            user_question=payload.get("user_question"),
        )
        return reasoning.to_dict()

    def _analyze_trends(self, payload: dict[str, Any]) -> dict[str, Any]:
        reports = reports_from_any(payload.get("reports", []))
        if len(reports) < 2:
            raise ValueError("Trend analysis requires at least two reports.")
        self.logger.log("trend.start", "Analyzing longitudinal report trends.", {"report_count": len(reports)})
        result = self.trend_analyzer.analyze(reports)
        self.logger.log(
            "trend.complete",
            "Trend analysis completed.",
            {"overall_direction": result.overall_direction, "risk_delta": result.risk_delta},
        )
        return result.to_dict()

    @staticmethod
    def _build_rationale(reference_range: ReferenceRange, status: str) -> str:
        if _is_unknown_status(status):
            return "Reference range was not supplied, so this marker was not classified as low, normal, or high."
        if _is_normal_status(status):
            return f"Value is within the supplied reference range {reference_range.describe()}."
        if _is_low_status(status):
            return f"Value is below the supplied lower reference boundary {reference_range.describe()}."
        if _is_high_status(status):
            return f"Value is above the supplied upper reference boundary {reference_range.describe()}."
        return "Reference range interpretation was not possible."

    @staticmethod
    def _deviation_percent(value: float, reference_range: ReferenceRange, status: str) -> float:
        if _is_low_status(status) and reference_range.low not in (None, 0):
            return ((reference_range.low - value) / abs(reference_range.low)) * 100
        if _is_high_status(status) and reference_range.high not in (None, 0):
            return ((value - reference_range.high) / abs(reference_range.high)) * 100
        return 0.0

    @staticmethod
    def _severity_weight(code: str, status: str, deviation_percent: float) -> int:
        return severity_weight_for(code, status, deviation_percent)

    @staticmethod
    def _finding_from_dict(payload: dict[str, Any]) -> ObservationFinding:
        range_payload = payload.get("reference_range_detail", payload.get("reference_range"))
        if isinstance(range_payload, dict):
            reference_range = ReferenceRange(
                low=range_payload.get("low"),
                high=range_payload.get("high"),
                text=range_payload.get("text"),
            )
        else:
            values = range_payload if isinstance(range_payload, list) else [None, None]
            reference_range = ReferenceRange(
                low=values[0] if len(values) > 0 else None,
                high=values[1] if len(values) > 1 else None,
            )
        return ObservationFinding(
            code=str(payload["code"]),
            display_name=str(payload["display_name"]),
            value=float(payload["value"]),
            unit=str(payload.get("unit", "")),
            reference_range=reference_range,
            status=str(payload["status"]),
            deviation_percent=float(payload.get("deviation_percent", 0)),
            severity_weight=int(payload.get("severity_weight", 0)),
            rationale=str(payload.get("rationale", "")),
        )


def _default_tool_registry() -> ToolRegistry:
    return ToolRegistry(AgentConfig.from_env(), ReasoningLogger())


def _patient_payload_from_lab_data(lab_data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lab_data, dict):
        raise ValueError("lab_data must be a dictionary.")
    if isinstance(lab_data.get("patient"), dict):
        return lab_data["patient"]
    if isinstance(lab_data.get("lab_data"), dict):
        return _patient_payload_from_lab_data(lab_data["lab_data"])
    if "observations" in lab_data:
        return lab_data
    if lab_data.get("report_text"):
        extracted = extract_lab_values(str(lab_data["report_text"]))
        return extracted["patient"]
    raise ValueError("Supply extracted lab data with a patient or observations payload.")


def _as_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _candidate_finding_items(payload: object) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    queue = [payload]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        for key in ("findings", "abnormal_values", "flagged_labs", "flagged"):
            items = _as_list(current.get(key))
            if items:
                return items

        for key in ("analysis", "result", "results", "lab_data"):
            nested = current.get(key)
            if isinstance(nested, dict):
                queue.append(nested)

    return []


def _finding_from_any(item: object) -> dict[str, Any]:
    if isinstance(item, dict):
        finding = dict(item)
        label = str(
            finding.get("display_name")
            or finding.get("name")
            or finding.get("code")
            or finding.get("factor")
            or finding.get("label")
            or "Observation"
        )
        raw_status = (
            finding.get("status")
            or finding.get("value_status")
            or finding.get("flag")
            or finding.get("interpretation")
            or _status_from_label(label)
        )
        if not (_is_abnormal_status(raw_status) or _is_normal_status(raw_status) or _is_unknown_status(raw_status)):
            raw_status = _status_from_label(label)
        clean_label = _strip_status_from_label(label)
        finding.setdefault("code", clean_label)
        finding.setdefault("display_name", clean_label)
        finding.setdefault("name", clean_label)
        finding["status"] = _status_label(raw_status)
        if "value" not in finding and "result" in finding:
            finding["value"] = finding["result"]
        return finding

    label = str(item or "Observation").strip()
    return {
        "code": _strip_status_from_label(label),
        "display_name": _strip_status_from_label(label),
        "name": _strip_status_from_label(label),
        "status": _status_label(_status_from_label(label)),
    }


def _findings_from_analysis(analysis: object) -> list[dict[str, Any]]:
    return [_finding_from_any(item) for item in _candidate_finding_items(analysis)]


def _explanation_from_risk(risk: dict[str, Any], findings: list[dict[str, Any]]) -> str:
    abnormal_labels = [
        f"{_finding_label(finding)} ({_status_label(finding.get('status'))})"
        for finding in findings
        if _is_abnormal_status(finding.get("status"))
    ]
    marker_summary = ", ".join(abnormal_labels[:5]) if abnormal_labels else "no scored abnormal markers"
    return (
        f"Overall risk is {str(risk.get('risk_level', 'LOW')).lower()} with a health risk score of "
        f"{risk.get('health_risk_score', 0)}/100. Key findings include {marker_summary}. "
        "This output is educational support only, not a diagnosis. A licensed clinician should interpret these findings in clinical context."
    )


def extract_lab_values(report_text: str) -> dict:
    """Normalize raw lab report text into internal FHIR-style lab values."""
    debug_print("Tool called:", "extract_lab_values", flush=True)
    return _default_tool_registry().call("extract_lab_values_tool", {"report_text": report_text})


def analyze_abnormal_values(lab_data: dict) -> dict:
    """Classify lab values and return abnormal markers."""
    debug_print("Tool called:", "analyze_abnormal_values", flush=True)
    patient_payload = _patient_payload_from_lab_data(lab_data)
    return _default_tool_registry().call("analyze_values_tool", {"patient": patient_payload})


def compute_health_risk(analysis: dict) -> dict:
    """Compute a 0-100 health risk score from analyzed or flagged lab values."""
    debug_print("Tool called:", "compute_health_risk", flush=True)
    patient = analysis.get("patient") if isinstance(analysis, dict) else None
    findings = _findings_from_analysis(analysis)
    return compute_final_risk(findings, patient=patient if isinstance(patient, dict) else None)


def generate_trend_analysis(history: list) -> dict:
    """Analyze longitudinal lab report trends across multiple reports."""
    debug_print("Tool called:", "generate_trend_analysis", flush=True)
    return _default_tool_registry().call("analyze_trends_tool", {"reports": history})


def generate_explanation(analysis: dict) -> str:
    """Generate an educational, non-diagnostic explanation for analyzed lab values."""
    debug_print("Tool called:", "generate_explanation", flush=True)
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be a dictionary.")

    patient = analysis.get("patient")
    findings = _findings_from_analysis(analysis)
    risk = analysis.get("risk") or analysis.get("risk_assessment") or compute_health_risk(analysis)
    if isinstance(patient, dict) and findings and isinstance(risk, dict):
        try:
            explanation = _default_tool_registry().call(
                "generate_explanation_tool",
                {
                    "patient": patient,
                    "findings": findings,
                    "risk": risk,
                    "user_question": analysis.get("user_question"),
                },
            )
        except (KeyError, TypeError, ValueError):
            return _explanation_from_risk(risk, findings)
        return str(explanation.get("clinical_insight") or explanation.get("patient_summary") or "")

    return _explanation_from_risk(risk if isinstance(risk, dict) else {}, findings)


def test_compute_health_risk_flags_status_substrings() -> None:
    result = compute_health_risk(
        {
            "flagged": [
                "Glucose (High)",
                "HbA1c (High)",
                "Hemoglobin (Low)",
                "MCV (Low)",
            ]
        }
    )
    assert result["health_risk_score"] > 30
    assert result["risk_level"] != "LOW"
