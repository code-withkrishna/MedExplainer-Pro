from __future__ import annotations

from typing import Any

from .fhir import patient_from_any, reports_from_any
from .models import AgentConfig, AgentOutput, ReasoningLogger, debug_print
from .tools import ToolRegistry
from .trends import sort_reports_chronologically


VALID_RISK_LEVELS = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
DEFAULT_TREND_ANALYSIS = "No longitudinal trend detected from the supplied report data."
DEFAULT_CLINICAL_INSIGHT = (
    "The supplied health data could not be fully interpreted. "
    "Please review the report with a licensed clinician."
)
DEFAULT_RECOMMENDED_ACTIONS = [
    "Review the supplied report with a licensed clinician.",
]


class MedAgent:
    """Central healthcare agent that reasons over FHIR-style patient data."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        logger: ReasoningLogger | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.logger = logger or ReasoningLogger()
        self.tool_registry = tool_registry or ToolRegistry(self.config, self.logger)

    @classmethod
    def from_env(cls) -> "MedAgent":
        return cls(config=AgentConfig.from_env())

    def tool_catalog(self) -> list[dict[str, Any]]:
        return self.tool_registry.definitions()

    def run(self, input_text: dict | list | str, user_question: str | None = None) -> dict[str, Any]:
        # Pipeline payloads may contain PHI, so full input/output traces are
        # emitted only when explicit DEBUG logging is enabled.
        debug_print("Pipeline input:", input_text, flush=True)
        try:
            self.logger.log("agent.start", "Starting MedAgent workflow.", {"user_question": user_question or ""})
            reports = reports_from_any(input_text)
            ordered_reports = sort_reports_chronologically(reports)
            patient = ordered_reports[-1] if ordered_reports else patient_from_any(input_text)

            extracted = self.tool_registry.call("extract_lab_values_tool", {"patient": patient.to_dict()})
            analyzed = self.tool_registry.call("analyze_values_tool", {"patient": extracted["patient"]})
            risk = self.tool_registry.call(
                "compute_risk_tool",
                {"patient": extracted["patient"], "findings": analyzed["findings"]},
            )
            explanation = self.tool_registry.call(
                "generate_explanation_tool",
                {
                    "patient": extracted["patient"],
                    "findings": analyzed["findings"],
                    "risk": risk,
                    "user_question": user_question,
                },
            )

            trend_summary = DEFAULT_TREND_ANALYSIS
            if len(ordered_reports) >= 2:
                trend_result = self.tool_registry.call(
                    "analyze_trends_tool",
                    {"reports": [report.to_dict() for report in ordered_reports]},
                )
                trend_summary = str(trend_result.get("trend_summary") or DEFAULT_TREND_ANALYSIS)

            result = self._final_output(
                health_risk_score=risk.get("health_risk_score", 0),
                risk_level=risk.get("risk_level", "LOW"),
                trend_analysis=trend_summary,
                clinical_insight=explanation.get("clinical_insight", ""),
                recommended_actions=explanation.get("recommended_actions", []),
                reasoning_mode=explanation.get("reasoning_mode"),
                ai_reasoning_enabled=bool(str(explanation.get("reasoning_mode", "")).lower().startswith("llm")),
            )

            self.logger.log(
                "agent.complete",
                "MedAgent workflow completed.",
                {
                    "patient_id": patient.id,
                    "risk_level": result["risk_level"],
                    "trend_analysis": bool(result["trend_analysis"]),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.log("agent.error", "MedAgent workflow failed.", {"error": str(exc)})
            result = self._error_output("Extraction failed", exc)

        debug_print("Pipeline output:", result, flush=True)
        return result

    @classmethod
    def _final_output(
        cls,
        *,
        health_risk_score: object,
        risk_level: object,
        trend_analysis: object,
        clinical_insight: object,
        recommended_actions: object,
        reasoning_mode: object = None,
        ai_reasoning_enabled: object = None,
    ) -> dict[str, Any]:
        try:
            score = float(health_risk_score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(100.0, score))
        if score.is_integer():
            score_value: int | float = int(score)
        else:
            score_value = round(score, 2)

        level = cls._normalize_risk_level(risk_level)
        actions = [
            action.strip()
            for action in recommended_actions
            if isinstance(action, str) and action.strip()
        ] if isinstance(recommended_actions, list) else []

        return AgentOutput(
            health_risk_score=score_value,
            risk_level=level,
            trend_analysis=str(trend_analysis or DEFAULT_TREND_ANALYSIS),
            clinical_insight=str(clinical_insight or DEFAULT_CLINICAL_INSIGHT),
            recommended_actions=actions or list(DEFAULT_RECOMMENDED_ACTIONS),
            reasoning_mode=str(reasoning_mode) if reasoning_mode else None,
            ai_reasoning_enabled=bool(ai_reasoning_enabled) if ai_reasoning_enabled is not None else None,
        ).to_dict()

    @staticmethod
    def _normalize_risk_level(value: object) -> str:
        normalized = str(value or "LOW").strip().upper()
        if normalized == "MEDIUM":
            normalized = "MODERATE"
        return normalized if normalized in VALID_RISK_LEVELS else "LOW"

    @classmethod
    def _error_output(cls, message: str, exc: Exception | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": "error", "message": message}
        if exc is not None:
            payload["detail"] = str(exc)
        return payload
