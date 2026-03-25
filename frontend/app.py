"""
DefectSense — Gradio Dashboard  (Session 6)

Tabs:
  1. Live Monitor   — real-time sensor charts + anomaly gauge per machine
  2. Alerts         — pending/approved alerts table with approve/reject
  3. Root Cause     — full ReAct reasoning trace for any alert
  4. System Health  — service status + ML model + A-MEM stats

Run:
    python frontend/app.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Optional

import httpx
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE    = os.getenv("API_BASE", "http://localhost:8080")
REFRESH_SEC = 5       # auto-refresh interval
MAX_POINTS  = 50      # readings shown in sensor chart

SEVERITY_COLOUR = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MEDIUM":   "#eab308",
    "LOW":      "#22c55e",
}

STATUS_COLOUR = {
    "CRITICAL": "#ef4444",
    "WARNING":  "#f97316",
    "NORMAL":   "#22c55e",
}


# ── API helpers ────────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> dict | list:
    try:
        with httpx.Client(timeout=8) as c:
            r = c.get(f"{API_BASE}{path}", params=params or {})
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def _post(path: str, body: dict) -> dict:
    try:
        with httpx.Client(timeout=8) as c:
            r = c.post(f"{API_BASE}{path}", json=body)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        return {"error": str(exc)}


# ── Tab 1: Live Monitor ────────────────────────────────────────────────────────

def get_machine_list() -> list[str]:
    data = _get("/api/dashboard/machines")
    machines = data.get("machines", [])
    if not machines:
        return ["No machines yet — run the simulator"]
    return [m["machine_id"] for m in machines]


def build_sensor_chart(machine_id: str) -> go.Figure:
    if not machine_id or "No machines" in machine_id:
        fig = go.Figure()
        fig.update_layout(title="No machine selected", template="plotly_dark", height=350)
        return fig

    data = _get(f"/api/sensors/{machine_id}/history", {"n": MAX_POINTS})
    if not data or isinstance(data, dict) and "error" in data:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {machine_id}", template="plotly_dark", height=350)
        return fig

    readings = data if isinstance(data, list) else []
    if not readings:
        fig = go.Figure()
        fig.update_layout(title=f"No readings yet for {machine_id}", template="plotly_dark", height=350)
        return fig

    # Parse timestamps
    ts = []
    for r in readings:
        try:
            ts.append(datetime.fromisoformat(str(r.get("timestamp", "")).replace("Z", "+00:00")))
        except Exception:
            ts.append(datetime.utcnow())

    sensors = [
        ("air_temperature",     "Air Temp (K)",   "#60a5fa"),
        ("process_temperature", "Process Temp (K)","#f87171"),
        ("rotational_speed",    "RPM",            "#34d399"),
        ("torque",              "Torque (Nm)",    "#fbbf24"),
        ("tool_wear",           "Tool Wear (min)","#a78bfa"),
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[s[1] for s in sensors],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]

    for i, (key, label, colour) in enumerate(sensors):
        vals = [r.get(key) for r in readings]
        row, col = positions[i]
        fig.add_trace(
            go.Scatter(
                x=ts, y=vals,
                mode="lines",
                name=label,
                line=dict(color=colour, width=1.5),
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(text=f"Sensor History — {machine_id} (last {len(readings)} readings)",
                   font=dict(size=14)),
        template="plotly_dark",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def build_anomaly_gauge(machine_id: str) -> go.Figure:
    if not machine_id or "No machines" in machine_id:
        prob, status, colour = 0.0, "UNKNOWN", "#6b7280"
    else:
        machines = _get("/api/dashboard/machines").get("machines", [])
        m = next((x for x in machines if x["machine_id"] == machine_id), None)
        prob   = m["failure_probability"] if m else 0.0
        status = m["status"] if m else "NORMAL"
        colour = STATUS_COLOUR.get(status, "#6b7280")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 32}},
        title={"text": f"Failure Probability<br><b style='color:{colour}'>{status}</b>",
               "font": {"size": 13}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": colour},
            "steps": [
                {"range": [0,  50],  "color": "#1f2937"},
                {"range": [50, 70],  "color": "#374151"},
                {"range": [70, 100], "color": "#4b1c1c"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 3}, "value": 70},
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=240,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def refresh_live_monitor(machine_id: str):
    return build_sensor_chart(machine_id), build_anomaly_gauge(machine_id)


# ── Tab 2: Alerts ──────────────────────────────────────────────────────────────

def _alerts_to_table(
    severity_filter: str = "All",
    machine_filter:  str = "",
    status_filter:   str = "All",
) -> tuple[list[list], list[str]]:
    params: dict = {"limit": 50}
    if machine_filter.strip():
        params["machine_id"] = machine_filter.strip()
    if status_filter != "All":
        params["status"] = status_filter.lower()

    data    = _get("/api/alerts", params)
    alerts  = data.get("alerts", [])

    if severity_filter != "All":
        alerts = [
            a for a in alerts
            if a.get("root_cause_report", {}).get("severity") == severity_filter
        ]

    headers = ["Alert ID", "Machine", "Severity", "Root Cause", "Confidence", "Status", "Created"]
    rows    = []
    for a in alerts:
        rcr      = a.get("root_cause_report", {})
        approved = a.get("approved")
        status   = "PENDING" if approved is None else ("APPROVED" if approved else "REJECTED")
        rows.append([
            a.get("alert_id", "")[:8],
            a.get("machine_id", ""),
            rcr.get("severity", ""),
            (rcr.get("root_cause", "")[:60] + "…") if len(rcr.get("root_cause","")) > 60 else rcr.get("root_cause",""),
            f"{rcr.get('confidence', 0):.0%}",
            status,
            str(a.get("created_at", ""))[:16],
        ])
    return rows, headers


def fetch_alerts_table(sev_filter, machine_filter, status_filter):
    rows, headers = _alerts_to_table(sev_filter, machine_filter, status_filter)
    return rows


def get_pending_alert_ids() -> list[str]:
    data   = _get("/api/alerts", {"status": "pending", "limit": 20})
    alerts = data.get("alerts", [])
    return [a["alert_id"] for a in alerts] if alerts else []


def approve_alert(alert_id: str) -> str:
    if not alert_id:
        return "Select an alert ID first."
    resp = _post(f"/api/alerts/{alert_id}/approve", {"approved_by": "dashboard_user"})
    if "error" in resp:
        return f"Error: {resp['error']}"
    return f"Alert {alert_id[:8]}... APPROVED"


def reject_alert(alert_id: str, reason: str) -> str:
    if not alert_id:
        return "Select an alert ID first."
    if not reason.strip():
        return "Please enter a rejection reason."
    resp = _post(
        f"/api/alerts/{alert_id}/reject",
        {"rejection_reason": reason, "rejected_by": "dashboard_user"},
    )
    if "error" in resp:
        return f"Error: {resp['error']}"
    return f"Alert {alert_id[:8]}... REJECTED"


# ── Tab 3: Root Cause Analysis ─────────────────────────────────────────────────

def get_all_alert_ids() -> list[str]:
    data   = _get("/api/alerts", {"limit": 30})
    alerts = data.get("alerts", [])
    return [f"{a['alert_id']} | {a.get('machine_id','')} | {a.get('root_cause_report',{}).get('severity','')}"
            for a in alerts]


def load_root_cause(alert_choice: str) -> tuple[str, str, str, str, str]:
    """Returns: explanation, root_cause, evidence, reasoning_steps, memory_used."""
    if not alert_choice:
        return ("", "", "", "", "")

    alert_id = alert_choice.split(" | ")[0].strip()
    data     = _get(f"/api/alerts/{alert_id}")

    if "error" in data:
        return (f"Error: {data['error']}", "", "", "", "")

    rcr  = data.get("root_cause_report", {})
    expl = data.get("plain_language_explanation", "")

    root_cause = (
        f"**Root Cause:** {rcr.get('root_cause', 'N/A')}\n\n"
        f"**Severity:** {rcr.get('severity', 'N/A')}  |  "
        f"**Confidence:** {rcr.get('confidence', 0):.0%}\n\n"
        f"**Failure Type:** {rcr.get('anomaly_result', {}).get('failure_type_prediction', 'N/A')}"
    )

    evidence = "\n".join(
        f"• {e}" for e in rcr.get("evidence", [])
    ) or "No evidence recorded."

    reasoning = "\n\n".join(
        f"**Step {i+1}:** {step}"
        for i, step in enumerate(rcr.get("reasoning_steps", []))
    ) or "No reasoning trace recorded."

    memory = "\n".join(
        f"• {m}" for m in rcr.get("agent_memory_used", [])
    ) or "No A-MEM memories recalled."

    return expl, root_cause, evidence, reasoning, memory


# ── Tab 4: System Health ───────────────────────────────────────────────────────

def load_system_health() -> tuple[str, str, str]:
    health = _get("/health")
    stats  = _get("/api/dashboard/stats")

    # Service status table
    svc_lines = []
    icons     = {True: "OK", False: "DOWN"}
    for key, val in health.items():
        if key in ("status", "as_of"):
            continue
        label = key.replace("_", " ").title()
        icon  = icons.get(bool(val), str(val))
        svc_lines.append(f"| {label} | {icon} |")

    svc_md = (
        "| Service | Status |\n"
        "|---------|--------|\n" +
        "\n".join(svc_lines)
    )

    # Pipeline stats
    dist = stats.get("failure_type_distribution", {})
    dist_lines = "\n".join(f"  - {k}: {v}" for k, v in dist.items()) or "  No data yet"

    sev_dist  = stats.get("alerts_by_severity", {})
    sev_lines = "\n".join(f"  - {k}: {v}" for k, v in sev_dist.items()) or "  No data yet"

    stats_md = (
        f"**Machines Monitored:** {stats.get('total_machines_monitored', 0)}\n\n"
        f"**Anomalies (last 24h):** {stats.get('anomalies_last_24h', 0)}\n\n"
        f"**Alerts Pending Approval:** {stats.get('alerts_pending_approval', 0)}\n\n"
        f"**Avg Resolution Time:** "
        f"{stats.get('avg_resolution_time_minutes') or 'N/A'} minutes\n\n"
        f"**Failure Type Distribution:**\n{dist_lines}\n\n"
        f"**Alerts by Severity:**\n{sev_lines}"
    )

    # ML model info
    ml_md = (
        "**Models Loaded:** LSTM Autoencoder + Isolation Forest\n\n"
        "**Sequence Length:** 30 readings\n\n"
        "**Anomaly Detection:** High-confidence = LSTM above threshold AND IForest = -1\n\n"
        "**Failure Types:** TWF · HDF · PWF · OSF · RNF\n\n"
        "**Dataset:** AI4I 2020 — 10,000 rows · 339 failure rows (3.4%)"
    )

    return svc_md, stats_md, ml_md


# ── Build Gradio App ───────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="DefectSense — Manufacturing AI Dashboard",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        .tab-nav button { font-size: 15px !important; }
        .status-ok   { color: #22c55e !important; font-weight: bold; }
        .status-warn { color: #f97316 !important; font-weight: bold; }
        .status-crit { color: #ef4444 !important; font-weight: bold; }
        """,
    ) as demo:

        gr.Markdown(
            "# DefectSense — Manufacturing Defect Intelligence\n"
            "> Hybrid ML + GenAI multi-agent system · Real-time anomaly detection & root cause analysis"
        )

        with gr.Tabs():

            # ── Tab 1: Live Monitor ────────────────────────────────────────────
            with gr.TabItem("🏭 Live Monitor"):
                with gr.Row():
                    machine_dd = gr.Dropdown(
                        label="Select Machine",
                        choices=get_machine_list(),
                        interactive=True,
                        scale=2,
                    )
                    refresh_btn = gr.Button("Refresh Now", variant="secondary", scale=1)
                    gr.Button("Reload Machine List", variant="secondary", scale=1).click(
                        fn=lambda: gr.Dropdown(choices=get_machine_list()),
                        outputs=machine_dd,
                    )

                with gr.Row():
                    sensor_chart = gr.Plot(label="Sensor Readings", scale=3)
                    gauge_chart  = gr.Plot(label="Anomaly Score",    scale=1)

                timer1 = gr.Timer(value=REFRESH_SEC)
                timer1.tick(fn=refresh_live_monitor, inputs=machine_dd, outputs=[sensor_chart, gauge_chart])
                refresh_btn.click(fn=refresh_live_monitor, inputs=machine_dd, outputs=[sensor_chart, gauge_chart])
                machine_dd.change(fn=refresh_live_monitor, inputs=machine_dd, outputs=[sensor_chart, gauge_chart])

            # ── Tab 2: Alerts ──────────────────────────────────────────────────
            with gr.TabItem("🚨 Alerts"):
                with gr.Row():
                    sev_filter     = gr.Dropdown(
                        label="Filter by Severity",
                        choices=["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        value="All", scale=1,
                    )
                    machine_filter = gr.Textbox(label="Filter by Machine ID", scale=1)
                    status_filter  = gr.Dropdown(
                        label="Filter by Status",
                        choices=["All", "pending", "approved", "rejected"],
                        value="All", scale=1,
                    )
                    gr.Button("Apply Filters", variant="primary", scale=1).click(
                        fn=fetch_alerts_table,
                        inputs=[sev_filter, machine_filter, status_filter],
                        outputs=gr.Dataframe(
                            headers=["Alert ID","Machine","Severity","Root Cause","Confidence","Status","Created"],
                            interactive=False,
                            wrap=True,
                        ),
                    )

                alerts_table = gr.Dataframe(
                    headers=["Alert ID","Machine","Severity","Root Cause","Confidence","Status","Created"],
                    interactive=False,
                    wrap=True,
                    label="Recent Alerts",
                )

                timer2 = gr.Timer(value=REFRESH_SEC)
                timer2.tick(
                    fn=fetch_alerts_table,
                    inputs=[sev_filter, machine_filter, status_filter],
                    outputs=alerts_table,
                )

                gr.Markdown("### Approve / Reject")
                with gr.Row():
                    pending_dd = gr.Dropdown(
                        label="Pending Alert ID",
                        choices=get_pending_alert_ids(),
                        interactive=True,
                        scale=3,
                    )
                    gr.Button("Refresh Pending List", scale=1).click(
                        fn=lambda: gr.Dropdown(choices=get_pending_alert_ids()),
                        outputs=pending_dd,
                    )

                with gr.Row():
                    approve_btn = gr.Button("Approve", variant="primary",  scale=1)
                    reject_reason = gr.Textbox(label="Rejection Reason", scale=3)
                    reject_btn  = gr.Button("Reject",  variant="stop",    scale=1)

                action_result = gr.Textbox(label="Action Result", interactive=False)
                approve_btn.click(fn=approve_alert, inputs=pending_dd,                        outputs=action_result)
                reject_btn.click( fn=reject_alert,  inputs=[pending_dd, reject_reason],       outputs=action_result)

            # ── Tab 3: Root Cause Analysis ─────────────────────────────────────
            with gr.TabItem("🧠 Root Cause Analysis"):
                with gr.Row():
                    alert_dd = gr.Dropdown(
                        label="Select Alert",
                        choices=get_all_alert_ids(),
                        interactive=True,
                        scale=4,
                    )
                    gr.Button("Refresh Alert List", scale=1).click(
                        fn=lambda: gr.Dropdown(choices=get_all_alert_ids()),
                        outputs=alert_dd,
                    )
                    gr.Button("Load Analysis", variant="primary", scale=1).click(
                        fn=load_root_cause,
                        inputs=alert_dd,
                        outputs=["rca_expl", "rca_root", "rca_evidence", "rca_steps", "rca_memory"],
                    )

                rca_expl = gr.Textbox(
                    label="Plain-Language Explanation (for factory floor)",
                    lines=3, interactive=False, elem_id="rca_expl",
                )
                with gr.Row():
                    rca_root     = gr.Markdown(label="Root Cause Summary", elem_id="rca_root")
                    rca_evidence = gr.Textbox(label="Supporting Evidence", lines=6,
                                              interactive=False, elem_id="rca_evidence")
                with gr.Row():
                    rca_steps  = gr.Textbox(label="ReAct Reasoning Steps", lines=12,
                                            interactive=False, elem_id="rca_steps")
                    rca_memory = gr.Textbox(label="A-MEM Memories Recalled", lines=6,
                                            interactive=False, elem_id="rca_memory")

                alert_dd.change(
                    fn=load_root_cause,
                    inputs=alert_dd,
                    outputs=[rca_expl, rca_root, rca_evidence, rca_steps, rca_memory],
                )

            # ── Tab 4: System Health ───────────────────────────────────────────
            with gr.TabItem("System Health"):
                with gr.Row():
                    gr.Button("Refresh Health", variant="primary").click(
                        fn=load_system_health,
                        outputs=["health_svc", "health_stats", "health_ml"],
                    )

                with gr.Row():
                    health_svc   = gr.Markdown(label="Service Status",  elem_id="health_svc")
                    health_stats = gr.Markdown(label="Pipeline Stats",  elem_id="health_stats")
                    health_ml    = gr.Markdown(label="ML Model Info",   elem_id="health_ml")

                timer4 = gr.Timer(value=30)   # health refreshes every 30s
                timer4.tick(fn=load_system_health, outputs=[health_svc, health_stats, health_ml])

        # Initial load on page open
        demo.load(
            fn=load_system_health,
            outputs=[health_svc, health_stats, health_ml],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"DefectSense Dashboard — connecting to {API_BASE}")
    print("Open http://localhost:7860 in your browser")
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
