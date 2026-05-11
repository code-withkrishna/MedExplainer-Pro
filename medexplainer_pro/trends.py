from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, List

from .models import DEFAULT_SAFETY_DISCLAIMER, FHIRPatient, FHIRObservation, ReferenceRange


CRITICAL_TREND_CODES = {
    "hemoglobin",
    "potassium",
    "sodium",
    "creatinine",
    "glucose",
    "blood_sugar",
    "troponin",
    "wbc",
}

MODERATE_TREND_CODES = {
    "cholesterol",
    "ldl",
    "triglycerides",
    "hba1c",
    "egfr",
    "mcv",
    "crp",
}

STABILITY_VOLATILE_THRESHOLD = 15.0
STABILITY_UNSTABLE_THRESHOLD = 5.0


@dataclass(slots=True)
class MarkerSnapshot:
    timestamp: str
    display_name: str
    value: float
    unit: str
    status: str
    deviation_percent: float


@dataclass(slots=True)
class MarkerTrend:
    code: str
    display_name: str
    timestamps: list[str]
    values: list[float]
    unit: str
    direction: str
    severity_change: str
    status_path: list[str]
    relative_change_percent: float
    clinical_interpretation: str
    risk_delta: int
    step_deltas: list[float]
    avg_delta: float
    abs_avg_delta: float = field(default=0.0, kw_only=True)
    volatility: float
    stability: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "display_name": self.display_name,
            "timestamps": list(self.timestamps),
            "values": list(self.values),
            "unit": self.unit,
            "direction": self.direction,
            "severity_change": self.severity_change,
            "status_path": list(self.status_path),
            "relative_change_percent": self.relative_change_percent,
            "clinical_interpretation": self.clinical_interpretation,
            "risk_delta": self.risk_delta,
            "step_deltas": list(self.step_deltas),
            "avg_delta": self.avg_delta,
            "abs_avg_delta": self.abs_avg_delta,
            "volatility": self.volatility,
            "stability": self.stability,
        }


@dataclass(slots=True)
class TrendAnalysis:
    trend_summary: str
    marker_trends: list[MarkerTrend]
    overall_direction: str
    risk_delta: int
    trajectory_score: int
    pattern_signals: list[str]
    safety_disclaimer: str = DEFAULT_SAFETY_DISCLAIMER
    trend_confidence: str = "medium"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend_summary": self.trend_summary,
            "marker_trends": [marker.to_dict() for marker in self.marker_trends],
            "overall_direction": self.overall_direction,
            "risk_delta": self.risk_delta,
            "trajectory_score": self.trajectory_score,
            "pattern_signals": list(self.pattern_signals),
            "safety_disclaimer": self.safety_disclaimer,
            "trend_confidence": self.trend_confidence,
            "note": self.note,
        }


def _parse_timestamp(value: str | None) -> tuple[int, datetime | None]:
    if not value:
        return (1, None)
    normalized = str(value).strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return (1, None)
    return (0, parsed)


def sort_reports_chronologically(reports: Iterable[FHIRPatient]) -> list[FHIRPatient]:
    indexed_reports = list(enumerate(reports))
    indexed_reports.sort(
        key=lambda item: (
            *_parse_timestamp(item[1].timestamp),
            item[0],
        )
    )
    return [report for _, report in indexed_reports]


def _timestamps_are_valid(timestamps: list[str | None]) -> bool:
    if len(timestamps) < 2 or any(not timestamp for timestamp in timestamps):
        return False
    parsed = [_parse_timestamp(timestamp)[1] for timestamp in timestamps]
    return all(timestamp is not None for timestamp in parsed)


def _status_text(status: object) -> str:
    return str(status or "").strip().lower()


def _status_matches(status: object, expected: str) -> bool:
    return expected.lower() in _status_text(status)


def _is_normal_status(status: object) -> bool:
    return _status_matches(status, "normal") and not _status_matches(status, "low") and not _status_matches(status, "high")


def _unknown_trend_analysis(report_count: int) -> TrendAnalysis:
    note = "Insufficient chronological data"
    return TrendAnalysis(
        trend_summary=(
            f"Trend analysis across {report_count} report(s) could not be interpreted because "
            "valid chronological timestamps were not supplied. "
            f"{DEFAULT_SAFETY_DISCLAIMER}"
        ),
        marker_trends=[],
        overall_direction="unknown",
        risk_delta=0,
        trajectory_score=50,
        pattern_signals=[],
        safety_disclaimer=DEFAULT_SAFETY_DISCLAIMER,
        trend_confidence="low",
        note=note,
    )


def _classify(value: float, reference_range: ReferenceRange) -> str:
    low = reference_range.low
    high = reference_range.high
    if low is None and high is None:
        return "UNKNOWN"
    if low is None:
        return "HIGH" if high is not None and value > high else "NORMAL"
    if high is None:
        return "LOW" if value < low else "NORMAL"
    if value < low:
        return "LOW"
    if value > high:
        return "HIGH"
    return "NORMAL"


def _deviation_percent(value: float, reference_range: ReferenceRange, status: str) -> float:
    if _status_matches(status, "low") and reference_range.low not in (None, 0):
        return ((reference_range.low - value) / abs(reference_range.low)) * 100
    if _status_matches(status, "high") and reference_range.high not in (None, 0):
        return ((value - reference_range.high) / abs(reference_range.high)) * 100
    return 0.0


def compute_trend(values: List[float], timestamps: list[str | None] | None = None) -> dict:
    """Compute direction from every ordered point, not just first and last."""
    if timestamps is not None and not _timestamps_are_valid(timestamps):
        return {
            "overall_direction": "unknown",
            "trend_confidence": "low",
            "note": "Insufficient chronological data",
            "direction": "unknown",
            "step_deltas": [],
            "avg_delta": 0.0,
            "abs_avg_delta": 0.0,
            "volatility": 0.0,
            "stability": "unknown",
        }

    if len(values) < 2:
        return {
            "direction": "stable",
            "step_deltas": [],
            "avg_delta": 0.0,
            "abs_avg_delta": 0.0,
            "volatility": 0.0,
            "stability": "stable",
        }

    step_deltas = [
        round(values[index] - values[index - 1], 4)
        for index in range(1, len(values))
    ]
    avg_delta = sum(step_deltas) / len(step_deltas)
    abs_avg_delta = sum(abs(delta) for delta in step_deltas) / len(step_deltas)
    # Volatility uses population standard deviation of step deltas, not the mean.
    # This correctly flags a +100/-100 oscillation as volatile (pstdev=100)
    # rather than stable (mean=0). avg_delta is kept separately for direction signals.
    volatility = statistics.pstdev(step_deltas) if len(step_deltas) >= 2 else 0.0
    mean_abs_value = sum(abs(value) for value in values) / len(values)
    direction_tolerance = max(mean_abs_value * 0.03, 0.1)
    signs = [
        1 if delta > direction_tolerance else -1 if delta < -direction_tolerance else 0
        for delta in step_deltas
    ]
    non_zero_signs = [sign for sign in signs if sign != 0]
    sign_changes = sum(
        1
        for index in range(1, len(non_zero_signs))
        if non_zero_signs[index] != non_zero_signs[index - 1]
    )
    if volatility >= STABILITY_VOLATILE_THRESHOLD:
        stability = "volatile"
    elif volatility >= STABILITY_UNSTABLE_THRESHOLD:
        stability = "unstable"
    elif abs(avg_delta) < 2.0:
        stability = "stable"
    else:
        stability = "trending"

    if stability in {"volatile", "unstable"} and sign_changes:
        direction = "unstable"
    elif abs(avg_delta) <= direction_tolerance:
        direction = "stable"
    elif avg_delta > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    return {
        "direction": direction,
        "step_deltas": step_deltas,
        "avg_delta": round(avg_delta, 4),
        "abs_avg_delta": round(abs_avg_delta, 4),
        "volatility": round(volatility, 4),
        "stability": stability,
    }


def _relative_change_percent(values: list[float]) -> float:
    if len(values) < 2 or values[0] == 0:
        return 0.0
    return round(((values[-1] - values[0]) / abs(values[0])) * 100, 2)


def _severity_change(first: MarkerSnapshot, latest: MarkerSnapshot) -> str:
    if _is_normal_status(first.status) and _is_normal_status(latest.status):
        return "no_change"
    if not _is_normal_status(first.status) and _is_normal_status(latest.status):
        return "improving"
    if _is_normal_status(first.status) and not _is_normal_status(latest.status):
        return "worsening"

    delta = latest.deviation_percent - first.deviation_percent
    if delta >= 5:
        return "worsening"
    if delta <= -5:
        return "improving"
    return "no_change"


def _marker_risk_delta(code: str, severity_change: str, first: MarkerSnapshot, latest: MarkerSnapshot) -> int:
    if severity_change == "no_change":
        return 0

    is_critical = code in CRITICAL_TREND_CODES
    is_moderate = code in MODERATE_TREND_CODES

    if severity_change == "worsening":
        delta = 10
        if is_critical:
            delta += 6
        elif is_moderate:
            delta += 3
        if latest.deviation_percent - first.deviation_percent >= 15:
            delta += 4
        return min(delta, 20)

    delta = -5
    if _is_normal_status(latest.status):
        delta -= 3
    if is_critical:
        delta -= 2
    elif is_moderate:
        delta -= 1
    return max(delta, -10)


def _note_for_marker(code: str, direction: str, severity_change: str) -> str:
    if direction == "unstable":
        return "This marker shows volatile movement across reports and should be interpreted with clinical context."
    if code in {"glucose", "blood_sugar", "hba1c"}:
        if severity_change == "worsening":
            return "Rising glucose-related markers may reflect worsening glycemic control and should be reviewed clinically."
        if severity_change == "improving":
            return "Glucose-related markers are moving closer to the reference range, which may indicate improving glycemic control."
    if code in {"hemoglobin", "mcv"}:
        if severity_change == "worsening" or direction == "decreasing":
            return "Declining red-cell markers may reflect a worsening anemia-related pattern and merit clinician review."
        if severity_change == "improving":
            return "Red-cell markers are trending toward the reference range, which may suggest an improving red-cell pattern."
    if code in {"creatinine", "egfr"}:
        if severity_change == "worsening":
            return "Renal markers are moving in a less favorable direction and should be interpreted in clinical context."
        if severity_change == "improving":
            return "Renal markers are moving in a more favorable direction, though follow-up still depends on the broader clinical picture."
    if code in {"cholesterol", "ldl", "triglycerides"}:
        if severity_change == "worsening":
            return "Lipid markers are trending upward in a way that may increase cardiovascular risk over time."
        if severity_change == "improving":
            return "Lipid markers are moving in a more favorable direction compared with prior reports."

    if severity_change == "worsening":
        return "This marker is moving farther from the reference range and may merit follow-up."
    if severity_change == "improving":
        return "This marker is moving closer to the reference range."
    return "This marker is relatively stable across the available reports."


def _detect_pattern_signals(marker_trends: list[MarkerTrend]) -> list[str]:
    by_code = {marker.code: marker for marker in marker_trends}
    pattern_signals: list[str] = []

    glucose_markers = [
        by_code.get("glucose"),
        by_code.get("blood_sugar"),
        by_code.get("hba1c"),
    ]
    glucose_worsening = [marker for marker in glucose_markers if marker and marker.severity_change == "worsening"]
    if len(glucose_worsening) >= 2:
        pattern_signals.append(
            "Consistent rise in glucose-related markers suggests worsening glycemic control that should be reviewed with a clinician."
        )

    hemoglobin = by_code.get("hemoglobin")
    mcv = by_code.get("mcv")
    if hemoglobin and mcv and hemoglobin.direction == "decreasing" and mcv.severity_change in {"worsening", "no_change"}:
        pattern_signals.append(
            "Declining hemoglobin with persistently low or worsening MCV forms a red-cell trend pattern that may reflect an anemia-like progression."
        )

    creatinine = by_code.get("creatinine")
    egfr = by_code.get("egfr")
    if creatinine and egfr and creatinine.direction == "increasing" and egfr.direction == "decreasing":
        pattern_signals.append(
            "Creatinine rising while eGFR declines can signal a worsening kidney filtration trend and merits clinical review."
        )

    lipid_markers = [by_code.get("cholesterol"), by_code.get("ldl"), by_code.get("triglycerides")]
    if sum(1 for marker in lipid_markers if marker and marker.severity_change == "worsening") >= 2:
        pattern_signals.append(
            "Multiple lipid markers are trending upward together, suggesting a less favorable cardiometabolic trajectory."
        )

    return pattern_signals


class ClinicalTrendAnalyzer:
    """Analyze longitudinal trajectories across multiple synthetic patient reports."""

    def analyze(self, reports: list[FHIRPatient]) -> TrendAnalysis:
        ordered_reports = sort_reports_chronologically(reports)
        if len(ordered_reports) < 2:
            return _unknown_trend_analysis(len(ordered_reports))
        if not _timestamps_are_valid([report.timestamp for report in ordered_reports]):
            return _unknown_trend_analysis(len(ordered_reports))

        observations_by_code: dict[str, list[MarkerSnapshot]] = {}
        for index, report in enumerate(ordered_reports):
            timestamp = report.timestamp or f"report_{index + 1}"
            for observation in report.observations:
                if observation.value is None:
                    continue
                status = _classify(observation.value, observation.reference_range)
                snapshot = MarkerSnapshot(
                    timestamp=timestamp,
                    display_name=observation.display_name or observation.code,
                    value=observation.value,
                    unit=observation.unit,
                    status=status,
                    deviation_percent=round(_deviation_percent(observation.value, observation.reference_range, status), 2),
                )
                observations_by_code.setdefault(observation.code, []).append(snapshot)

        marker_trends: list[MarkerTrend] = []
        for code, snapshots in observations_by_code.items():
            if len(snapshots) < 2:
                continue

            values = [snapshot.value for snapshot in snapshots]
            trend_metrics = compute_trend(values, [snapshot.timestamp for snapshot in snapshots])
            direction = trend_metrics["direction"]
            severity_change = _severity_change(snapshots[0], snapshots[-1])
            marker_trends.append(
                MarkerTrend(
                    code=code,
                    display_name=snapshots[-1].display_name,
                    timestamps=[snapshot.timestamp for snapshot in snapshots],
                    values=values,
                    unit=snapshots[-1].unit,
                    direction=direction,
                    severity_change=severity_change,
                    status_path=[snapshot.status for snapshot in snapshots],
                    relative_change_percent=_relative_change_percent(values),
                    clinical_interpretation=_note_for_marker(code, direction, severity_change),
                    risk_delta=_marker_risk_delta(code, severity_change, snapshots[0], snapshots[-1]),
                    step_deltas=trend_metrics["step_deltas"],
                    avg_delta=trend_metrics["avg_delta"],
                    abs_avg_delta=trend_metrics["abs_avg_delta"],
                    volatility=trend_metrics["volatility"],
                    stability=trend_metrics["stability"],
                )
            )

        marker_trends.sort(key=lambda marker: (marker.risk_delta, marker.display_name), reverse=True)

        worsening_count = sum(1 for marker in marker_trends if marker.severity_change == "worsening")
        improving_count = sum(1 for marker in marker_trends if marker.severity_change == "improving")
        increasing_count = sum(1 for marker in marker_trends if marker.direction == "increasing")
        decreasing_count = sum(1 for marker in marker_trends if marker.direction == "decreasing")
        stable_count = sum(1 for marker in marker_trends if marker.direction == "stable")
        unstable_count = sum(1 for marker in marker_trends if marker.direction == "unstable")
        risk_delta = sum(marker.risk_delta for marker in marker_trends)

        if unstable_count:
            overall_direction = "unstable"
        elif increasing_count > decreasing_count and increasing_count >= stable_count:
            overall_direction = "increasing"
        elif decreasing_count > increasing_count and decreasing_count >= stable_count:
            overall_direction = "decreasing"
        else:
            overall_direction = "stable"

        pattern_signals = _detect_pattern_signals(marker_trends)
        report_span = f"{ordered_reports[0].timestamp or 'earliest available report'} to {ordered_reports[-1].timestamp or 'latest available report'}"
        summary_parts = [
            f"Trend analysis across {len(ordered_reports)} reports from {report_span} shows an overall {overall_direction} direction."
        ]
        if worsening_count:
            summary_parts.append(f"{worsening_count} markers are moving farther from their reference ranges.")
        if improving_count:
            summary_parts.append(f"{improving_count} markers are moving closer to their reference ranges.")
        if unstable_count:
            summary_parts.append(f"{unstable_count} markers show volatile movement across intermediate reports.")
        if pattern_signals:
            summary_parts.append("Pattern signals: " + " ".join(pattern_signals[:2]))
        summary_parts.append(DEFAULT_SAFETY_DISCLAIMER)

        average_delta = risk_delta / max(len(marker_trends), 1) if marker_trends else 0.0
        instability_penalty = unstable_count * 4
        trajectory_score = round(max(0, min(100, 50 + (average_delta * 2) + instability_penalty)))

        return TrendAnalysis(
            trend_summary=" ".join(summary_parts),
            marker_trends=marker_trends,
            overall_direction=overall_direction,
            risk_delta=risk_delta,
            trajectory_score=trajectory_score,
            pattern_signals=pattern_signals,
            safety_disclaimer=DEFAULT_SAFETY_DISCLAIMER,
        )


def report_from_patient_and_timestamp(patient: FHIRPatient, timestamp: str | None) -> FHIRPatient:
    patient.timestamp = timestamp
    return patient
