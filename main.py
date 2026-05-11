from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from medexplainer_pro.agent import MedAgent
from medexplainer_pro.models import AgentConfig


ROOT = Path(__file__).resolve().parent
SAMPLE_PATIENT = ROOT / "samples" / "synthetic_patient.json"
SAMPLE_HISTORY = ROOT / "samples" / "synthetic_patient_history.json"


def _read_json(path: str | Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedExplainer Pro unified agent runner")
    parser.add_argument("--demo", action="store_true", help="Run the built-in synthetic demo patient.")
    parser.add_argument("--history-demo", action="store_true", help="Run the built-in synthetic longitudinal trend demo.")
    parser.add_argument("--patient-json", help="Path to a simplified FHIR-style patient JSON payload.")
    parser.add_argument("--fhir-json", help="Path to a FHIR resource JSON payload.")
    parser.add_argument("--report-text", help="Path to a plain-text lab report.")
    parser.add_argument("--question", help="Optional user question for the reasoning layer.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AgentConfig.from_env()
    if (args.demo or args.history_demo) and not config.llm.enabled:
        print(
            "Demo note: running in deterministic fallback mode (no active LLM provider). "
            "Set GROQ_API_KEY or OPENAI_API_KEY to demonstrate AI reasoning.",
            file=sys.stderr,
        )

    if args.patient_json:
        payload = _read_json(args.patient_json)
    elif args.fhir_json:
        payload = _read_json(args.fhir_json)
    elif args.report_text:
        payload = _read_text(args.report_text)
    elif args.history_demo:
        payload = _read_json(SAMPLE_HISTORY)
    else:
        payload = _read_json(SAMPLE_PATIENT)

    result = MedAgent(config=config).run(payload, user_question=args.question)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
