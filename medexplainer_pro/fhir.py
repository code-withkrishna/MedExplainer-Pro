from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from .models import FHIRObservation, FHIRPatient, ReferenceRange


# Sex-specific and age-adjusted reference range overrides.
# Ranges sourced from standard clinical reference tables
# (WHO, NHANES, and common laboratory reference intervals).
# Keys are normalized observation codes (lowercase, underscored).
# Each entry maps a (sex, age_min, age_max) tuple to (low, high).
# sex is "male", "female", or None (applies to all).
# age_min / age_max are inclusive years; None means unbounded.
DEMOGRAPHIC_REFERENCE_RANGES: dict[str, list[tuple]] = {
    "hemoglobin": [
        # (sex, age_min, age_max, low, high)
        ("male",   18, None, 13.5, 17.5),
        ("female", 18, None, 12.0, 15.5),
        ("male",    0,   17, 11.0, 16.0),
        ("female",  0,   17, 11.0, 16.0),
    ],
    "hematocrit": [
        ("male",   18, None, 41.0, 53.0),
        ("female", 18, None, 36.0, 46.0),
    ],
    "hemoglobin_a1c": [
        (None, None, None, None, 5.7),   # prediabetes threshold, all demographics
    ],
    "hba1c": [
        (None, None, None, None, 5.7),
    ],
    "creatinine": [
        ("male",   18, None, 0.74, 1.35),
        ("female", 18, None, 0.59, 1.04),
    ],
    "alkaline_phosphatase": [
        ("male",   18, None, 44.0, 147.0),
        ("female", 18, None, 44.0, 147.0),
        (None,      0,  17, 54.0, 369.0),   # higher in children/adolescents
    ],
}


def apply_demographic_reference_range(
    code: str,
    current_range: "ReferenceRange",
    demographics: dict,
) -> "ReferenceRange":
    """Return a demographically adjusted ReferenceRange for well-known codes.

    Only overrides the range when ALL of the following are true:
    - The observation code is present in DEMOGRAPHIC_REFERENCE_RANGES.
    - The patient demographics contain a usable 'gender'/'sex' and/or 'age'.
    - The supplied current_range has BOTH low and high as None (i.e. the caller
      provided no range), OR the code is in a known set where caller-supplied
      ranges are commonly wrong (currently: hemoglobin, hematocrit, creatinine).

    In all other cases the original range is returned unchanged so that
    caller-supplied ranges from FHIR servers take precedence.
    """
    ALWAYS_OVERRIDE_CODES = {"hemoglobin", "hematocrit", "creatinine"}
    normalized = normalize_code(code)
    entries = DEMOGRAPHIC_REFERENCE_RANGES.get(normalized)
    if not entries:
        return current_range

    has_no_range = current_range.low is None and current_range.high is None
    if not has_no_range and normalized not in ALWAYS_OVERRIDE_CODES:
        return current_range

    raw_sex = str(demographics.get("gender") or demographics.get("sex") or "").strip().lower()
    sex = "male" if raw_sex in {"male", "m"} else "female" if raw_sex in {"female", "f"} else None
    age: int | None = None
    try:
        age = int(demographics["age"])
    except (KeyError, TypeError, ValueError):
        pass

    for entry in entries:
        entry_sex, age_min, age_max, low, high = entry
        if entry_sex is not None and sex is not None and entry_sex != sex:
            continue
        if age is not None:
            if age_min is not None and age < age_min:
                continue
            if age_max is not None and age > age_max:
                continue
        # Match found — return adjusted range, preserving existing text label.
        return ReferenceRange(
            low=low if low is not None else current_range.low,
            high=high if high is not None else current_range.high,
            text=current_range.text,
        )

    return current_range


_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_RANGE_LABEL = r"(?:Normal(?:\s+Range)?|Reference(?:\s+Range)?|Ref(?:erence)?(?:\s+Range)?)"
_LINE_PATTERN = re.compile(
    rf"(?P<name>[A-Za-z][A-Za-z0-9 ./%+\-(),]*?)\s*:\s*"
    rf"(?P<value>{_NUM})\s*"
    r"(?P<unit>.*?)\s*"
    rf"(?:\(|\[)?\s*{_RANGE_LABEL}\s*:\s*"
    r"(?P<range>[^)\]\n]+)"
    r"(?:\)|\])?",
    re.IGNORECASE,
)
_VALUE_ONLY_PATTERN = re.compile(
    rf"^\s*(?P<name>[A-Za-z][A-Za-z0-9 ./%+\-(),]*?)\s*:\s*"
    rf"(?P<value>{_NUM})\s*"
    r"(?P<unit>[^\n()]*)"
    r"(?:\(\s*(?P<status>high|low|normal|unknown)\s*\))?\s*$",
    re.IGNORECASE,
)
_RANGE_UPPER_ONLY = re.compile(rf"^(?:<=|<|\u2264)\s*({_NUM})(?:\s*[^\d].*)?$")
_RANGE_LOWER_ONLY = re.compile(rf"^(?:>=|>|\u2265)\s*({_NUM})(?:\s*[^\d].*)?$")
_RANGE_BOTH = re.compile(rf"^({_NUM})\s*(?:-|\u2013|to)\s*({_NUM})(?:\s*[^\d].*)?$", re.IGNORECASE)
_AGE_PATTERN = re.compile(r"\bAge\s*:\s*(\d{1,3})\b", re.IGNORECASE)
_GENDER_PATTERN = re.compile(r"\b(?:Gender|Sex)\s*:\s*([A-Za-z]+)\b", re.IGNORECASE)
_CONTEXT_LABELS = {"age", "gender", "sex"}


def safe_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(re.sub(r"[^\d.\-]", "", str(value).strip()))
    except (TypeError, ValueError):
        return None


def _normalise_report_name(raw_name: str) -> str:
    name = " ".join(raw_name.strip().split())
    compact = name.replace(" ", "")
    if any(ch.isdigit() for ch in name) or any(ch in "-/" for ch in name):
        return name
    if any(ch.islower() for ch in name) and any(ch.isupper() for ch in name):
        return name
    if name.isupper() and len(compact) <= 6:
        return name
    return name.title()


def _parse_range(range_str: str) -> tuple[float | None, float | None]:
    value = range_str.strip()
    match = _RANGE_UPPER_ONLY.match(value)
    if match:
        return None, safe_float(match.group(1))
    match = _RANGE_LOWER_ONLY.match(value)
    if match:
        return safe_float(match.group(1)), None
    match = _RANGE_BOTH.match(value)
    if match:
        return safe_float(match.group(1)), safe_float(match.group(2))
    return None, None


def extract_parameters(report_text: str) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for line in str(report_text or "").splitlines():
        match = _LINE_PATTERN.search(line)
        if match:
            low, high = _parse_range(match.group("range"))
        else:
            match = _VALUE_ONLY_PATTERN.search(line)
            if not match:
                continue
            raw_name = match.group("name").strip().lower()
            if raw_name in _CONTEXT_LABELS:
                continue
            low, high = None, None

        value = safe_float(match.group("value"))
        if value is None:
            continue
        # Optional status groups return None when the plain-text line has only
        # "Name: value unit". Coerce through an empty string so value-only lab
        # reports do not crash while preserving the existing HIGH/LOW logic.
        status = (match.groupdict().get("status") or "").upper() or None
        parameters.append(
            {
                "name": _normalise_report_name(match.group("name")),
                "value": value,
                "unit": match.group("unit").strip(),
                "low": low,
                "high": high,
                "status": status,
            }
        )
    return parameters


def extract_context(report_text: str) -> dict[str, Any]:
    context: dict[str, Any] = {}
    age_match = _AGE_PATTERN.search(str(report_text or ""))
    if age_match:
        context["age"] = int(age_match.group(1))
    gender_match = _GENDER_PATTERN.search(str(report_text or ""))
    if gender_match:
        context["gender"] = gender_match.group(1).strip().title()
    return context


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantity_value(value: object) -> object:
    if isinstance(value, dict):
        return value.get("value")
    return value


def normalize_code(value: str) -> str:
    normalized = str(value or "").strip().lower()
    for char in ("(", ")", "/", "-", ",", "%"):
        normalized = normalized.replace(char, " ")
    return "_".join(part for part in normalized.split() if part)


def reference_range_from_any(raw_value: object) -> ReferenceRange:
    if isinstance(raw_value, ReferenceRange):
        return raw_value

    if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
        if isinstance(raw_value[0], dict) and ("low" in raw_value[0] or "high" in raw_value[0]):
            return _reference_range_from_fhir(raw_value)
        low = _to_float(raw_value[0])
        high = _to_float(raw_value[1])
        return ReferenceRange(low=low, high=high)

    if isinstance(raw_value, dict):
        low = raw_value.get("low")
        high = raw_value.get("high")
        if "as_list" in raw_value and isinstance(raw_value["as_list"], list):
            return reference_range_from_any(raw_value["as_list"])
        return ReferenceRange(
            low=_to_float(_quantity_value(low)),
            high=_to_float(_quantity_value(high)),
            text=str(raw_value.get("text")) if raw_value.get("text") else None,
        )

    return ReferenceRange()


def _display(concept: dict | None) -> str:
    if not isinstance(concept, dict):
        return ""
    if concept.get("text"):
        return str(concept["text"]).strip()
    for coding in concept.get("coding", []):
        if isinstance(coding, dict) and coding.get("display"):
            return str(coding["display"]).strip()
    for coding in concept.get("coding", []):
        if isinstance(coding, dict) and coding.get("code"):
            return str(coding["code"]).strip()
    return ""


def _code(concept: dict | None) -> str:
    if not isinstance(concept, dict):
        return ""
    for coding in concept.get("coding", []):
        if isinstance(coding, dict) and coding.get("code"):
            return str(coding["code"]).strip()
    if concept.get("text"):
        return str(concept["text"]).strip()
    return ""


def _quantity_unit(quantity: dict | None) -> str:
    if not isinstance(quantity, dict):
        return ""
    return str(quantity.get("unit") or quantity.get("code") or "")


def _reference_range_from_fhir(ref_ranges: object) -> ReferenceRange:
    if not isinstance(ref_ranges, list) or not ref_ranges:
        return ReferenceRange()
    first_range = ref_ranges[0]
    if not isinstance(first_range, dict):
        return ReferenceRange()
    return ReferenceRange(
        low=_to_float(_quantity_value(first_range.get("low"))),
        high=_to_float(_quantity_value(first_range.get("high"))),
        text=str(first_range.get("text")) if first_range.get("text") else None,
    )


def _observation_from_parts(
    code_source: str,
    display_name: str,
    value: object,
    unit: str,
    reference_range: ReferenceRange,
) -> dict[str, Any] | None:
    numeric_value = _to_float(value)
    if numeric_value is None:
        return None
    label = display_name or code_source or "Observation"
    return {
        "code": normalize_code(label or code_source) or "observation",
        "display_name": label,
        "value": numeric_value,
        "unit": unit,
        "reference_range": reference_range.as_list(),
        "reference_range_detail": reference_range.to_dict(),
        "referenceRange": reference_range.to_fhir(),
    }


def _observations_from_fhir_observation(resource: dict) -> list[dict[str, Any]]:
    display_name = _display(resource.get("code")) or str(resource.get("id") or "Observation")
    code_source = _code(resource.get("code")) or display_name
    observations: list[dict[str, Any]] = []

    value_quantity = resource.get("valueQuantity")
    if isinstance(value_quantity, dict):
        observation = _observation_from_parts(
            code_source=code_source,
            display_name=display_name,
            value=value_quantity.get("value"),
            unit=_quantity_unit(value_quantity),
            reference_range=_reference_range_from_fhir(resource.get("referenceRange")),
        )
        if observation:
            observations.append(observation)

    for component in resource.get("component", []):
        if not isinstance(component, dict):
            continue
        component_quantity = component.get("valueQuantity")
        if not isinstance(component_quantity, dict):
            continue
        component_name = _display(component.get("code")) or display_name
        component_code = _code(component.get("code")) or component_name
        observation = _observation_from_parts(
            code_source=component_code,
            display_name=component_name,
            value=component_quantity.get("value"),
            unit=_quantity_unit(component_quantity),
            reference_range=_reference_range_from_fhir(
                component.get("referenceRange", resource.get("referenceRange"))
            ),
        )
        if observation:
            observations.append(observation)

    return observations


def _age_from_birth_date(birth_date: object) -> int | None:
    if not isinstance(birth_date, str) or not birth_date.strip():
        return None
    parts = birth_date.strip().split("-")
    if not parts or len(parts) > 3:
        return None
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) >= 2 else 1
        day = int(parts[2]) if len(parts) >= 3 else 1
        datetime(year, month, day)
    except (TypeError, ValueError):
        return None
    today = datetime.today()
    age = today.year - year
    if (today.month, today.day) < (month, day):
        age -= 1
    return age if age >= 0 else None


def _patient_context(resource: dict) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if resource.get("gender"):
        context["gender"] = str(resource["gender"]).title()
    age = _age_from_birth_date(resource.get("birthDate"))
    if age is not None:
        context["age"] = age
    return context


def _resources_from_fhir(resource: object) -> list[dict]:
    if isinstance(resource, str):
        resource = json.loads(resource)
    if isinstance(resource, list):
        resources: list[dict] = []
        for item in resource:
            resources.extend(_resources_from_fhir(item))
        return resources
    if not isinstance(resource, dict):
        return []
    if resource.get("resourceType") == "Bundle":
        resources = []
        for entry in resource.get("entry", []):
            if isinstance(entry, dict):
                resources.extend(_resources_from_fhir(entry.get("resource")))
        return resources
    resources = [resource]
    for contained in resource.get("contained", []):
        if isinstance(contained, dict):
            resources.extend(_resources_from_fhir(contained))
    return resources


def _timestamp_from_resource(resource: dict) -> str | None:
    for key in ("effectiveDateTime", "issued", "date", "authoredOn"):
        if resource.get(key):
            return str(resource[key])
    period = resource.get("effectivePeriod")
    if isinstance(period, dict):
        return period.get("start") or period.get("end")
    return None


def fhir_to_internal_model(fhir_patient: dict) -> dict:
    """Directly map FHIR resources to the internal patient model without text parsing."""
    raw_fhir: object = json.loads(fhir_patient) if isinstance(fhir_patient, str) else fhir_patient
    resources = _resources_from_fhir(raw_fhir)
    patient_resource = next(
        (resource for resource in resources if resource.get("resourceType") == "Patient"),
        None,
    )
    observations: list[dict[str, Any]] = []
    timestamp: str | None = None

    for resource in resources:
        if timestamp is None:
            timestamp = _timestamp_from_resource(resource)
        if resource.get("resourceType") != "Observation":
            continue
        observations.extend(_observations_from_fhir_observation(resource))

    resource_dict = raw_fhir if isinstance(raw_fhir, dict) else {}
    patient_id = None
    demographics: dict[str, Any] = {}
    if patient_resource:
        patient_id = patient_resource.get("id")
        demographics = _patient_context(patient_resource)

    subject = resource_dict.get("subject") if isinstance(resource_dict, dict) else None
    if not patient_id and isinstance(subject, dict) and subject.get("reference"):
        patient_id = str(subject["reference"]).rstrip("/").split("/")[-1]

    return {
        "id": str(patient_id or resource_dict.get("id") or f"patient-{uuid4().hex[:8]}"),
        "demographics": demographics,
        "source": "fhir",
        "timestamp": timestamp,
        "observations": observations,
    }


def patient_from_dict(payload: dict) -> FHIRPatient:
    patient_payload = payload.get("patient", payload) if isinstance(payload, dict) else {}
    observations_payload = patient_payload.get("observations", [])
    observations: list[FHIRObservation] = []

    for observation in observations_payload:
        if not isinstance(observation, dict):
            continue
        value = _to_float(observation.get("value"))
        if value is None:
            continue
        observations.append(
            FHIRObservation(
                code=str(observation.get("code") or normalize_code(observation.get("display_name", "observation"))),
                display_name=str(observation.get("display_name") or observation.get("code") or "Observation"),
                value=value,
                unit=str(observation.get("unit", "")),
                reference_range=reference_range_from_any(
                    observation.get(
                        "reference_range_detail",
                        observation.get("reference_range", observation.get("referenceRange")),
                    )
                ),
                status=str(observation["status"]) if observation.get("status") else None,
            )
        )

    demographics = patient_payload.get("demographics", {})
    if not demographics:
        for field_name in ("age", "gender", "sex"):
            if field_name in patient_payload:
                demographics[field_name] = patient_payload[field_name]

    return FHIRPatient(
        id=str(patient_payload.get("id") or f"patient-{uuid4().hex[:8]}"),
        source=str(patient_payload.get("source", "structured-input")),
        demographics=dict(demographics),
        observations=observations,
        timestamp=patient_payload.get("timestamp"),
    )


def patient_from_report_text(report_text: str, patient_id: str | None = None) -> FHIRPatient:
    parameters = extract_parameters(report_text)
    context = extract_context(report_text)
    observations = [
        FHIRObservation(
            code=normalize_code(parameter["name"]),
            display_name=parameter["name"],
            value=float(parameter["value"]),
            unit=str(parameter.get("unit", "")),
            reference_range=ReferenceRange(
                low=parameter.get("low"),
                high=parameter.get("high"),
            ),
            status=str(parameter["status"]) if parameter.get("status") else None,
        )
        for parameter in parameters
    ]
    # Apply demographic reference range adjustments for well-known codes.
    adjusted_observations = [
        FHIRObservation(
            code=obs.code,
            display_name=obs.display_name,
            value=obs.value,
            unit=obs.unit,
            reference_range=apply_demographic_reference_range(
                obs.code, obs.reference_range, context
            ),
            status=obs.status,
        )
        for obs in observations
    ]
    return FHIRPatient(
        id=patient_id or f"patient-{uuid4().hex[:8]}",
        source="raw-report",
        demographics=context,
        observations=adjusted_observations,
    )


def patient_from_fhir_resource(resource: str | dict | list, patient_id: str | None = None) -> FHIRPatient:
    internal_model = fhir_to_internal_model(resource)  # type: ignore[arg-type]
    if patient_id:
        internal_model["id"] = patient_id
    return patient_from_dict(internal_model)


def patient_from_any(payload: object, patient_id: str | None = None) -> FHIRPatient:
    if isinstance(payload, FHIRPatient):
        return payload
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return patient_from_report_text(stripped, patient_id=patient_id)
            if isinstance(parsed, dict) and parsed.get("resourceType"):
                return patient_from_fhir_resource(parsed, patient_id=patient_id)
            return patient_from_dict(parsed)
        return patient_from_report_text(stripped, patient_id=patient_id)
    if isinstance(payload, list):
        return patient_from_fhir_resource(payload, patient_id=patient_id)
    if isinstance(payload, dict):
        if payload.get("resourceType"):
            return patient_from_fhir_resource(payload, patient_id=patient_id)
        return patient_from_dict(payload)
    raise TypeError("Unsupported patient payload.")


def _looks_like_report_payload(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    if "observations" in item:
        return True
    if "patient" in item and isinstance(item.get("patient"), dict):
        return True
    if "report_text" in item or "fhir_resource" in item or "timestamp" in item:
        return True
    return False


def _patient_from_report_entry(entry: dict) -> FHIRPatient:
    timestamp = entry.get("timestamp")
    if entry.get("patient") is not None:
        patient = patient_from_dict(entry["patient"])
    elif entry.get("fhir_resource") is not None:
        patient = patient_from_any(entry["fhir_resource"])
    elif entry.get("report_text"):
        patient = patient_from_report_text(str(entry["report_text"]), patient_id=entry.get("id"))
    else:
        patient = patient_from_dict(entry)
    patient.timestamp = timestamp or patient.timestamp
    if entry.get("id") and not patient.id:
        patient.id = str(entry["id"])
    return patient


def reports_from_any(payload: object) -> list[FHIRPatient]:
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [patient_from_report_text(stripped)]
            return reports_from_any(parsed)
        return [patient_from_report_text(stripped)]

    if isinstance(payload, dict):
        if payload.get("resourceType"):
            return [patient_from_fhir_resource(payload)]
        if isinstance(payload.get("reports"), list):
            return [_patient_from_report_entry(report) for report in payload["reports"] if isinstance(report, dict)]
        return [_patient_from_report_entry(payload)]

    if isinstance(payload, list):
        if payload and all(_looks_like_report_payload(item) for item in payload):
            return [_patient_from_report_entry(item) for item in payload if isinstance(item, dict)]
        return [patient_from_any(payload)]

    if isinstance(payload, FHIRPatient):
        return [payload]

    raise TypeError("Unsupported report payload.")
