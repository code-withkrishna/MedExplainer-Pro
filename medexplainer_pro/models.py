from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any, Literal


ObservationStatus = Literal["LOW", "NORMAL", "HIGH", "UNKNOWN"]
RiskLevel = Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]

TRUE_VALUES = {"1", "true", "yes", "on"}
REASONING_MODE_ENV_VAR = "MEDEXPLAINER_ALLOW_REMOTE_REASONING"
DEFAULT_SAFETY_DISCLAIMER = (
    "This output is educational support only, not a diagnosis. "
    "A licensed clinician should interpret these findings in clinical context."
)


def load_env_file(path: str | os.PathLike[str] | None = None) -> None:
    """Load simple KEY=value pairs from .env without overriding real environment variables."""
    env_path = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file()


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def debug_enabled() -> bool:
    return env_truthy("DEBUG") or env_truthy("MEDEXPLAINER_DEBUG")


def debug_print(*args: object, **kwargs: Any) -> None:
    if debug_enabled():
        print(*args, **kwargs)


@dataclass(slots=True)
class LLMConfig:
    provider: str = "disabled"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout_seconds: int = 20
    temperature: float = 0.2
    allow_remote_reasoning: bool = False

    @property
    def enabled(self) -> bool:
        return self.allow_remote_reasoning and bool(self.api_key and self.base_url and self.model)


@dataclass(slots=True)
class AgentConfig:
    app_name: str = "MedExplainer Pro"
    safety_disclaimer: str = DEFAULT_SAFETY_DISCLAIMER
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        provider = os.environ.get("MEDEXPLAINER_LLM_PROVIDER", "").strip().lower()
        allow_remote_flag = os.environ.get(REASONING_MODE_ENV_VAR)
        legacy_allow_flag = os.environ.get("MEDEXPLAINER_ALLOW_LLM_PHI")
        allow_remote_reasoning = (
            (allow_remote_flag is not None and env_truthy(REASONING_MODE_ENV_VAR))
            or (legacy_allow_flag is not None and env_truthy("MEDEXPLAINER_ALLOW_LLM_PHI"))
        )
        if allow_remote_flag is None and legacy_allow_flag is None:
            # Auto-enable remote reasoning when a provider key is present and no explicit opt-out
            # is set. This makes the AI path the default for demo and production use.
            allow_remote_reasoning = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))

        if provider == "groq" or (not provider and os.environ.get("GROQ_API_KEY")):
            llm = LLMConfig(
                provider="groq",
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.environ.get("GROQ_API_KEY", ""),
                base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                allow_remote_reasoning=allow_remote_reasoning,
            )
        elif provider == "openai" or (not provider and os.environ.get("OPENAI_API_KEY")):
            llm = LLMConfig(
                provider="openai",
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                allow_remote_reasoning=allow_remote_reasoning,
            )
        else:
            llm = LLMConfig(allow_remote_reasoning=False)

        return cls(
            safety_disclaimer=os.environ.get("MEDEXPLAINER_SAFETY_DISCLAIMER", DEFAULT_SAFETY_DISCLAIMER),
            llm=llm,
        )


class ReasoningLogger:
    """Captures and emits human-readable reasoning step logs."""

    def __init__(self, name: str = "medexplainer.agent") -> None:
        self._entries: list[dict[str, Any]] = []
        self._logger = logging.getLogger(name)
        self._logger.propagate = False
        if not self._logger.handlers and env_truthy("MEDEXPLAINER_VERBOSE_LOGS"):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s"))
            self._logger.addHandler(handler)
        elif not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())
        self._logger.setLevel(logging.INFO)

    def log(self, step: str, message: str, payload: dict[str, Any] | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "message": message,
            "payload": payload or {},
        }
        self._entries.append(entry)
        self._logger.info("%s | %s", step, message)

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)


@dataclass(slots=True)
class ReferenceRange:
    """FHIR-friendly representation of a lab reference range."""

    low: float | None = None
    high: float | None = None
    text: str | None = None

    def as_list(self) -> list[float | None]:
        return [self.low, self.high]

    def describe(self) -> str:
        if self.low is not None and self.high is not None:
            return f"{self.low}-{self.high}"
        if self.high is not None:
            return f"<{self.high}"
        if self.low is not None:
            return f">{self.low}"
        return self.text or "Not supplied"

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "text": self.text,
            "as_list": self.as_list(),
        }

    def to_fhir(self) -> list[dict[str, Any]]:
        if self.low is None and self.high is None and not self.text:
            return []
        payload: dict[str, Any] = {}
        if self.low is not None:
            payload["low"] = {"value": self.low}
        if self.high is not None:
            payload["high"] = {"value": self.high}
        if self.text:
            payload["text"] = self.text
        return [payload]


@dataclass(slots=True)
class FHIRObservation:
    """Simplified Observation model aligned to the hackathon payload shape."""

    code: str
    value: float
    unit: str = ""
    reference_range: ReferenceRange = field(default_factory=ReferenceRange)
    display_name: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "display_name": self.display_name or self.code,
            "value": self.value,
            "unit": self.unit,
            "reference_range": self.reference_range.as_list(),
            "reference_range_detail": self.reference_range.to_dict(),
            "referenceRange": self.reference_range.to_fhir(),
        }
        if self.status:
            payload["status"] = self.status
        return payload


@dataclass(slots=True)
class FHIRPatient:
    """Simplified patient record for internal reasoning and MCP transport."""

    id: str
    observations: list[FHIRObservation]
    demographics: dict[str, Any] = field(default_factory=dict)
    source: str = "synthetic"
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "demographics": dict(self.demographics),
            "source": self.source,
            "timestamp": self.timestamp,
            "observations": [observation.to_dict() for observation in self.observations],
        }


@dataclass(slots=True)
class ObservationFinding:
    """Classified interpretation of a patient observation."""

    code: str
    display_name: str
    value: float
    unit: str
    reference_range: ReferenceRange
    status: ObservationStatus
    deviation_percent: float
    severity_weight: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "display_name": self.display_name,
            "value": self.value,
            "unit": self.unit,
            "reference_range": self.reference_range.as_list(),
            "reference_range_detail": self.reference_range.to_dict(),
            "status": self.status,
            "deviation_percent": self.deviation_percent,
            "severity_weight": self.severity_weight,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class RiskAssessment:
    """Scored risk assessment built from abnormal findings."""

    risk_level: RiskLevel
    health_risk_score: int
    abnormal_count: int
    total_count: int
    severity_score: int
    contributing_factors: list[str] = field(default_factory=list)
    risk_breakdown: list[dict[str, Any]] = field(default_factory=list)
    legacy_risk_level: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "health_risk_score": self.health_risk_score,
            "abnormal_count": self.abnormal_count,
            "total_count": self.total_count,
            "severity_score": self.severity_score,
            "contributing_factors": list(self.contributing_factors),
            "risk_breakdown": [dict(item) for item in self.risk_breakdown],
            "legacy_risk_level": self.legacy_risk_level,
        }


@dataclass(slots=True)
class ReasoningOutput:
    """Narrative output from the reasoning layer."""

    patient_summary: str
    clinical_insight: str
    recommended_actions: list[str]
    explanation_confidence: float
    pattern_signals: list[str]
    safety_disclaimer: str
    reasoning_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_summary": self.patient_summary,
            "clinical_insight": self.clinical_insight,
            "recommended_actions": list(self.recommended_actions),
            "explanation_confidence": self.explanation_confidence,
            "pattern_signals": list(self.pattern_signals),
            "safety_disclaimer": self.safety_disclaimer,
            "reasoning_mode": self.reasoning_mode,
        }


@dataclass(slots=True)
class AgentOutput:
    """Final hackathon-ready response contract."""

    health_risk_score: int | float
    risk_level: RiskLevel
    trend_analysis: str
    clinical_insight: str
    recommended_actions: list[str]
    reasoning_mode: str | None = None
    ai_reasoning_enabled: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "health_risk_score": self.health_risk_score,
            "risk_level": self.risk_level,
            "trend_analysis": self.trend_analysis,
            "clinical_insight": self.clinical_insight,
            "recommended_actions": list(self.recommended_actions),
        }
        if self.reasoning_mode:
            payload["reasoning_mode"] = self.reasoning_mode
        if self.ai_reasoning_enabled is not None:
            payload["ai_reasoning_enabled"] = self.ai_reasoning_enabled
        return payload


@dataclass(frozen=True, slots=True)
class MCPToolDefinition:
    """Minimal MCP-style tool declaration with explicit JSON schemas."""

    name: str
    title: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "annotations": dict(self.annotations),
        }
