/**
 * MedExplainer Pro dashboard, hydrated from the unified MedAgent backend.
 */

const API_RUN_URL = "/api/run";
const RISK_LEVEL_ATTR = {
  LOW: "low",
  MODERATE: "medium",
  HIGH: "high",
  CRITICAL: "critical"
};

const EMPTY_DASHBOARD = {
  health_risk_score: 0,
  risk_level: "LOW",
  trend_analysis: "Submit input to run analysis.",
  clinical_insight:
    "MedExplainer Pro returns educational support only. It is not a diagnosis.",
  recommended_actions: ["Enter report text or patient JSON, then click Submit."]
};

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setDataStatus(isLive) {
  const host = document.querySelector(".header__actions");
  if (!host) return;

  let badge = document.getElementById("dataStatusBadge");
  if (!badge) {
    badge = document.createElement("span");
    badge.id = "dataStatusBadge";
    badge.className = "data-status-badge";
    host.appendChild(badge);
  }

  badge.dataset.source = isLive ? "live" : "offline";
  badge.textContent = isLive ? "Live Backend" : "No Result";
}

function normalizeRiskLevel(value) {
  const level = String(value ?? "LOW").trim().toUpperCase();
  if (level === "MEDIUM") return "MODERATE";
  return ["LOW", "MODERATE", "HIGH", "CRITICAL"].includes(level) ? level : "LOW";
}

function normalise(data) {
  return {
    health_risk_score: Number(data?.health_risk_score ?? 0),
    risk_level: normalizeRiskLevel(data?.risk_level),
    trend_analysis: String(data?.trend_analysis ?? ""),
    clinical_insight: String(data?.clinical_insight ?? ""),
    recommended_actions: Array.isArray(data?.recommended_actions)
      ? data.recommended_actions.filter(item => typeof item === "string" && item.trim())
      : []
  };
}

function setRiskCard(d) {
  const card = document.getElementById("riskCard");
  const level = normalizeRiskLevel(d.risk_level);
  card.dataset.riskLevel = RISK_LEVEL_ATTR[level] ?? "low";

  const score = Number.isFinite(d.health_risk_score) ? Math.round(d.health_risk_score) : 0;
  document.getElementById("scoreNumber").textContent = String(score);

  const pct = Math.min(100, Math.max(0, score));
  const ring = document.getElementById("riskRingProgress");
  const meter = document.getElementById("riskMeterFill");
  if (ring) {
    ring.style.strokeDasharray = "100";
    ring.style.strokeDashoffset = String(100 - pct);
  }
  if (meter) meter.style.width = pct + "%";

  document.getElementById("riskTierLabel").textContent = level + " RISK";
  document.getElementById("confPct").textContent = "Live";
  requestAnimationFrame(() => {
    document.getElementById("confBar").style.width = "100%";
  });
}

function trendDirectionFromText(text) {
  const value = String(text ?? "").toLowerCase();
  if (value.includes("unstable") || value.includes("volatile")) return "unstable";
  if (value.includes("worsen") || value.includes("increasing") || value.includes("rise")) return "increasing";
  if (value.includes("improv") || value.includes("decreasing") || value.includes("declin")) return "decreasing";
  if (value.includes("stable")) return "stable";
  return "unknown";
}

function overallTrendArrow(dir) {
  if (dir === "increasing") return "↑";
  if (dir === "decreasing") return "↓";
  if (dir === "unstable") return "↕";
  return "→";
}

function overallTrendLabel(dir) {
  if (dir === "increasing") return "WORSENING";
  if (dir === "decreasing") return "IMPROVING";
  if (dir === "unstable") return "UNSTABLE";
  if (dir === "stable") return "STABLE";
  return "REVIEW";
}

function trendHeroClassFromDir(dir) {
  if (dir === "increasing") return "trend-hero--worsening";
  if (dir === "decreasing") return "trend-hero--improving";
  if (dir === "unstable") return "trend-hero--unstable";
  if (dir === "stable") return "trend-hero--stable";
  return "trend-hero--unknown";
}

function renderTrend(text) {
  const dir = trendDirectionFromText(text);
  const hero = document.getElementById("trendDirection");
  hero.className = "trend-hero " + trendHeroClassFromDir(dir);
  document.getElementById("trendArrow").textContent = overallTrendArrow(dir);
  document.getElementById("trendText").textContent = overallTrendLabel(dir);
  document.getElementById("trajectoryValue").textContent = "MedAgent";
  requestAnimationFrame(() => {
    document.getElementById("trajectoryBar").style.width = dir === "unknown" ? "20%" : "70%";
  });

  document.getElementById("markersList").innerHTML = `<li class="trend-list__item trend-list__item--neutral">
    <span class="trend-list__arrow">→</span>
    <span class="trend-list__name">${esc(text || EMPTY_DASHBOARD.trend_analysis)}</span>
    <span class="trend-list__state">Summary</span>
  </li>`;
}

function renderBreakdown(d) {
  document.getElementById("breakdownList").innerHTML = `<li class="breakdown-list__item">
    <span class="impact-badge">${esc(String(Math.round(d.health_risk_score)))}</span>
    <span class="factor-chip">${esc(normalizeRiskLevel(d.risk_level))} risk from unified MedAgent output</span>
  </li>`;
}

function renderActions(actions) {
  const rows = actions.length ? actions : EMPTY_DASHBOARD.recommended_actions;
  document.getElementById("actionsList").innerHTML = rows
    .map(
      action => `<li class="actions-list__item">
      <span class="actions-list__glyph" aria-hidden="true">✚</span>
      <span class="actions-list__text">${esc(action)}</span>
    </li>`
    )
    .join("");
}

function formatInsightParagraphs(text) {
  const value = (text ?? "").trim() || EMPTY_DASHBOARD.clinical_insight;
  return value
    .split(/(?<=[.!?])\s+/)
    .filter(Boolean)
    .map(paragraph => `<p class="insight-para">${esc(paragraph.trim())}</p>`)
    .join("");
}

function render(data) {
  const d = normalise(data);
  document.getElementById("patientId").textContent = "Input";
  document.getElementById("patientSummary").textContent = "Run output from MedAgent orchestrator";
  document.getElementById("footerTs").textContent =
    "Generated " + new Date().toLocaleString("en-GB", { hour12: false });

  setRiskCard(d);
  renderTrend(d.trend_analysis);
  document.getElementById("clinicalInsight").innerHTML = formatInsightParagraphs(d.clinical_insight);
  document.getElementById("patternSignals").innerHTML = "";
  document.getElementById("disclaimer").textContent =
    "This output is educational support only, not a diagnosis. A licensed clinician should interpret these findings in clinical context.";
  renderBreakdown(d);
  renderActions(d.recommended_actions);
}

function renderDashboard(data) {
  render(data);
}

function renderError(payload) {
  const message = payload?.message ? String(payload.message) : "Request failed";
  document.getElementById("formStatus").textContent = message;
  document.getElementById("patientSummary").textContent = message;
  setDataStatus(false);
}

async function runAnalysis(inputText) {
  const response = await fetch(API_RUN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ input_text: inputText })
  });
  const payload = await response.json();
  if (!response.ok || payload?.status === "error") {
    throw payload;
  }
  return payload;
}

document.getElementById("analysisForm").addEventListener("submit", async event => {
  event.preventDefault();
  const submitBtn = document.getElementById("btnSubmit");
  const input = document.getElementById("inputText");
  const status = document.getElementById("formStatus");

  status.textContent = "Running...";
  submitBtn.disabled = true;

  try {
    const payload = await runAnalysis(input.value);
    renderDashboard(payload);
    setDataStatus(true);
    status.textContent = "Completed";
  } catch (errorPayload) {
    renderError(errorPayload);
  } finally {
    submitBtn.disabled = false;
  }
});

document.getElementById("btnRefresh").addEventListener("click", () => {
  document.getElementById("analysisForm").requestSubmit();
});

document.addEventListener("DOMContentLoaded", () => {
  renderDashboard(EMPTY_DASHBOARD);
  setDataStatus(false);
});
