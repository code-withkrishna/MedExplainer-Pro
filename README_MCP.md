# MedExplainer Pro — SHARP-on-MCP Lab Intelligence Superpower

> **Agents Assemble Hackathon Submission · Option 1: Build a Superpower (MCP)**
> Published on the [Prompt Opinion Marketplace](https://app.promptopinion.ai)

---

## The Problem

Lab reports are the most ordered clinical tool in medicine, yet they remain the
most misunderstood by the people who need them most — patients and the
clinicians trying to explain them under time pressure. A typical discharge
summary hands a patient a table of numbers with no explanation of what is
abnormal, why it matters, or what to do next.

Existing AI assistants can read a lab report if you paste it in. But they
cannot pull a live patient record from an EHR mid-conversation, they cannot
track how a marker is trending across visits, and they cannot pass structured
clinical context through a chain of agents without bespoke glue code.

**MedExplainer Pro closes that gap** — as a composable MCP Superpower that any
agent on the Prompt Opinion platform can pick up and use.

---

## What It Does

MedExplainer Pro is an MCP server that exposes a six-tool pipeline for
end-to-end lab report intelligence. It accepts free-text lab reports, raw FHIR
JSON, or a patient ID forwarded via SHARP headers, and returns a structured
clinical assessment any downstream agent can reason over.

```
Input (text · FHIR JSON · SHARP patient ID)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  MedExplainer Pro MCP Server  (FastMCP + SHARP)     │
│                                                     │
│  1. extract_lab_values   → normalise to FHIR-style  │
│  2. analyze_abnormal_values → LOW / NORMAL / HIGH   │
│  3. compute_health_risk  → 0-100 Health Risk Score  │
│  4. generate_explanation → patient-safe narrative   │
│  5. generate_trend_analysis → longitudinal trajectory │
│  6. run_med_agent        → full pipeline in one call │
└─────────────────────────────────────────────────────┘
        │
        ▼
Structured JSON · Prompt Opinion Agent · Downstream Agents
```

### The six MCP tools

| Tool | What it does |
|------|-------------|
| `extract_lab_values` | Normalises raw text, FHIR resources, or structured JSON into the internal FHIR-style patient model |
| `analyze_abnormal_values` | Classifies every observation as LOW, NORMAL, or HIGH against reference ranges and computes deviation % |
| `compute_health_risk` | Combines abnormal findings into a 0–100 Health Risk Score and a risk tier (LOW / MODERATE / HIGH / CRITICAL) using a weighted severity model |
| `generate_explanation` | Produces a patient-safe narrative with pattern signals, clinical insight, and recommended next steps via hybrid LLM + rule-based reasoning |
| `generate_trend_analysis` | Analyses longitudinal marker trajectories across two or more reports — direction, severity change, risk delta, and volatility |
| `run_med_agent` | Orchestrates the full pipeline in a single call; SHARP-aware: resolves a live FHIR patient when `X-Patient-ID` is injected by the platform |

---

## SHARP Extension Spec Support

MedExplainer Pro is fully compliant with the **SHARP (Standardised Healthcare
Agent Remote Protocol)** extension spec.

Per SHARP §3.2, the MCP server never runs OAuth itself. The Prompt Opinion
platform bridges the EHR session credentials and forwards them as HTTP headers
on every tool call:

```
X-FHIR-Server-URL     → base URL of the patient's FHIR endpoint
X-FHIR-Access-Token   → bearer token from the EHR / SMART launch session
X-Patient-ID          → patient currently in context
```

The server advertises compliance on every `initialize` response:

```json
{
  "capabilities": {
    "experimental": {
      "fhir_context_required": true
    }
  }
}
```

Implementation note: this capability flag is currently injected via a guarded
FastMCP initialization-options compatibility patch. If a future FastMCP release
changes internals, the patch safely no-ops with a warning while MCP tools
continue to function.

A `SharpContextMiddleware` (Starlette) reads the three headers per request,
stores them in Python `ContextVar`s scoped to that request, and makes them
available to every tool without polluting the tool signatures. When Prompt
Opinion injects a patient ID, `run_med_agent` automatically fetches the live
FHIR patient record from the forwarded endpoint and feeds it into the same
pipeline — no code change required in the agent layer.

SHARP support is **fully backward-compatible**: all six tools function
identically when called without SHARP headers (local dev, direct curl, other
MCP clients).

---

## FHIR Integration

MedExplainer Pro works against any FHIR R4 endpoint. `fhir.py` handles:

- **FHIR R4 Observation bundles** — parses `component`, `valueQuantity`,
  `referenceRange` natively
- **Free-text lab reports** — regex-based extraction of common lab patterns
  (`Hemoglobin: 10.2 g/dL (Low)`) when structured FHIR is not available
- **Hybrid normalisation** — FHIR-first with graceful fallback to text
  extraction, so the same tool works whether the upstream agent passes a live
  EHR record or a pasted clinical note

Public HAPI FHIR R4 sandbox is supported out of the box (no credentials):

```
X-FHIR-Server-URL: https://hapi.fhir.org/baseR4
X-FHIR-Access-Token: anonymous
X-Patient-ID: 592
```

---

## Architecture

```
┌──────────────────────────────────────┐
│        Prompt Opinion Platform       │
│  (injects SHARP headers per call)    │
└─────────────────┬────────────────────┘
                  │ POST /mcp + SHARP headers
                  ▼
┌──────────────────────────────────────┐
│      SharpContextMiddleware          │
│  Parses X-FHIR-* + X-Patient-ID     │
│  Stores in ContextVar (per-request)  │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│        FastMCP Tool Registry         │
│  6 tools · JSON-RPC 2.0 · HTTP SSE  │
└─────────────────┬────────────────────┘
                  │
          ┌───────┴────────┐
          ▼                ▼
  ┌──────────────┐  ┌─────────────────┐
  │  MedAgent    │  │  FHIR Client    │
  │  Pipeline    │  │  (httpx async)  │
  └──────┬───────┘  └─────────────────┘
         │
   ┌─────┴──────────────────────┐
   │         ToolRegistry       │
   │  extract → analyze →       │
   │  risk → explain → trends   │
   └────────────────────────────┘
         │
   ┌─────┴──────────┐
   │ HybridReasoner │
   │  LLM (Groq)    │
   │  + rule-based  │
   │  fallback      │
   └────────────────┘
```

The hybrid reasoner uses the Groq API for explanation generation when
available, and falls back to a deterministic rule-based engine when the API
key is not present — ensuring the server is always functional.

For judge-facing demos where Generative AI usage is scored, run with a valid
LLM API key so `reasoning_mode` resolves to `llm` (instead of `fallback`).

---

## Output Contract

Every public entrypoint returns the same structured schema, making the output
composable with any downstream agent:

```json
{
  "health_risk_score": 52,
  "risk_level": "HIGH",
  "trend_analysis": "HbA1c rising across three visits. Glucose consistently above range.",
  "clinical_insight": "Three abnormal metabolic markers detected. Pattern is consistent with poorly controlled diabetes. This is educational support only.",
  "recommended_actions": [
    "Urgent follow-up with endocrinologist recommended.",
    "Medication review indicated given HbA1c trajectory.",
    "Review these findings with a licensed clinician."
  ],
  "contributing_factors": [
    "HbA1c: HIGH (7.1%)",
    "Glucose: HIGH (168 mg/dL)",
    "Hemoglobin: LOW (10.2 g/dL)"
  ],
  "sharp_context": {
    "patient_id": "592",
    "fhir_server_url": "https://hapi.fhir.org/baseR4"
  }
}
```

`risk_level` is always one of `LOW`, `MODERATE`, `HIGH`, or `CRITICAL`.
`sharp_context` is present only when called via the Prompt Opinion platform
with SHARP headers — confirming the context was received and used.
`reasoning_mode` and `ai_reasoning_enabled` indicate whether the explanation
was generated through remote LLM reasoning or deterministic fallback mode.

---

## Risk Scoring Model

The Health Risk Score (0–100) is computed from a weighted severity model:

| Component | Weight |
|-----------|--------|
| Critical parameter abnormal (Hemoglobin, Glucose, Troponin, Creatinine…) | 3 base |
| Moderate parameter abnormal (HbA1c, Cholesterol, eGFR…) | 2 base |
| Other parameter abnormal | 1 base |
| Deviation ≥ 10% from reference boundary | +1 |
| Deviation ≥ 25% from reference boundary | +2 |
| Abnormal marker burden (ratio × 35) | additive |
| Multiple abnormal markers together (≥ 3) | +8 pattern bonus |

Score thresholds: **LOW** < 20 · **MODERATE** 20–44 · **HIGH** 45–69 · **CRITICAL** ≥ 70

These weights are prototype heuristics for hackathon prioritization, not a
clinically validated scoring instrument.

---

## Potential Impact

**Who it helps:** Clinicians spending 30 seconds per lab result explaining
values to patients, care managers monitoring panels for chronic disease,
discharge nurses summarising results for post-acute follow-up.

**Hypothesis for outcomes improvement:**
- Reduces clinician explanation time per patient encounter by surfacing
  structured, patient-safe language that can be reviewed and approved rather
  than written from scratch.
- Enables proactive panel management — an agent using `generate_trend_analysis` across
  a population's lab history can flag patients whose markers are deteriorating
  before they become critical.
- Composable with prior auth agents, care gap agents, and documentation agents
  on the Prompt Opinion platform — the structured JSON output is designed to
  be a building block, not a terminal endpoint.

**Why traditional rule-based software cannot do this:** The explanation
generation step requires understanding which combination of findings is
clinically meaningful in context, not just which individual values are out of
range. A patient with low Hemoglobin, high HbA1c, and high Glucose represents
a different clinical picture than the same three values in isolation — pattern
recognition and natural language generation are not addressable with static
rules.

---

## Feasibility & Safety

- **No PHI is stored.** Every call is stateless. The SHARP token is held in
  a per-request `ContextVar` and discarded when the request completes.
- **The server never handles OAuth.** Token acquisition is the caller's
  responsibility per SHARP §3.2 — the server only reads a forwarded header.
- **All outputs include a safety disclaimer** and are framed as educational
  support, never diagnosis.
- **Works within existing FHIR infrastructure.** Any FHIR R4 endpoint — Epic,
  Cerner, HAPI, athenahealth — is supported without vendor-specific code.
- **No proprietary dependencies.** FastMCP, Starlette, httpx, and optionally
  the Groq API. Fully replaceable components.

---

## Project Structure

```
MedExplainer_Pro_MCP/
├── mcp_server.py              # FastMCP server · SHARP middleware · 6 tools
├── main.py                    # Local demo runner
├── requirements.txt
├── medexplainer_pro/
│   ├── agent.py               # MedAgent orchestrator
│   ├── tools.py               # ToolRegistry · 5 internal tools · scoring model
│   ├── reasoning.py           # HybridReasoner (Groq LLM + rule-based fallback)
│   ├── fhir.py                # FHIR R4 parser · text extraction · normalisation
│   ├── trends.py              # ClinicalTrendAnalyzer · longitudinal analysis
│   └── models.py              # Dataclasses · schemas · AgentConfig
├── frontend ui/               # Browser dashboard (calls /api/demo)
└── samples/                   # Synthetic patient JSON for local testing
```

---

## Run Locally

```bash
# Install
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Optional: add Groq API key for LLM-powered explanations
# Without it, the rule-based fallback is used automatically
export GROQ_API_KEY=your_key_here

# Start the MCP server
python mcp_server.py
# → http://0.0.0.0:5000/mcp

# Verify all 6 tools are registered
curl -X POST http://localhost:5000/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# Test the full pipeline (no SHARP headers — backward-compatible mode)
curl -X POST http://localhost:5000/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
        "name":"run_med_agent",
        "arguments":{"input_text":"Hemoglobin: 10.2 g/dL\nGlucose: 168 mg/dL\nHbA1c: 7.1%"}}}'

# Test with SHARP headers (HAPI public sandbox — no credentials needed)
curl -X POST http://localhost:5000/mcp \
  -H 'Content-Type: application/json' \
  -H 'X-FHIR-Server-URL: https://hapi.fhir.org/baseR4' \
  -H 'X-FHIR-Access-Token: anonymous' \
  -H 'X-Patient-ID: 592' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
        "name":"run_med_agent","arguments":{"input_text":"592"}}}'
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Optional | Enables LLM-powered explanation generation. Falls back to rule-based engine if absent. |
| `FHIR_BASE_URL` | Optional | Default FHIR endpoint for non-SHARP calls. Overridden by `X-FHIR-Server-URL` header when present. |
| `MCP_PORT` | Optional | Port for MCP runtime in env-based launches (see `.env.example`). |

Note: `mcp_server.py` currently starts on hardcoded port `5000` when run directly.

---

## Prompt Opinion Platform Integration

MedExplainer Pro is published to the Prompt Opinion Marketplace and connected
to a BYO agent configured to handle clinical lab queries.

The agent workflow on the platform:

1. Clinician or care manager opens a patient chart in context
2. Prompt Opinion bridges the EHR session into SHARP headers automatically
3. Agent calls `run_med_agent` with the patient ID — no manual data entry
4. MedExplainer Pro fetches the live FHIR record, runs the full pipeline,
   and returns a structured assessment
5. Agent surfaces the explanation, risk score, and recommended actions
   directly in the clinician workspace

---

## Built With

- [FastMCP](https://github.com/jlowin/fastmcp) — MCP server framework
- [Starlette](https://www.starlette.io/) — ASGI middleware (SharpContextMiddleware)
- [httpx](https://www.python-httpx.org/) — async FHIR client
- [Groq API](https://groq.com/) — LLM inference for explanation generation
- [Prompt Opinion](https://www.promptopinion.ai/) — agent platform and marketplace

---

## Disclaimer

MedExplainer Pro is an informational tool for educational and decision-support
purposes only. It does not provide medical diagnoses, treatment
recommendations, or clinical decisions. All outputs should be reviewed by a
licensed healthcare professional. Not intended for direct patient care without
clinician review.
