"""
DefectSense — Gradio Dashboard  (Session 6)

Tabs:
  1. Live Monitor   — real-time sensor charts + anomaly gauge per machine
  2. Alerts         — pending/approved alerts table with approve/reject
  3. Root Cause     — full ReAct reasoning trace for any alert
  4. System Health  — service status + pipeline stats + ML info

Run:
    python frontend/app.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import httpx
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE    = os.getenv("API_BASE", "http://localhost:8080")
REFRESH_SEC = 5
MAX_POINTS  = 50

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
    return [m["machine_id"] for m in machines] if machines else ["(no machines yet)"]


def build_sensor_chart(machine_id: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["Air Temp (K)", "Process Temp (K)", "RPM",
                        "Torque (Nm)", "Tool Wear (min)", ""],
        vertical_spacing=0.18, horizontal_spacing=0.08,
    )

    if not machine_id or "no machines" in machine_id.lower():
        fig.update_layout(title="Select a machine", template="plotly_dark", height=420)
        return fig

    data = _get(f"/api/sensors/{machine_id}/history", {"n": MAX_POINTS})
    readings = data if isinstance(data, list) else []

    if not readings:
        fig.update_layout(title=f"No readings for {machine_id}", template="plotly_dark", height=420)
        return fig

    ts = []
    for r in readings:
        try:
            ts.append(datetime.fromisoformat(str(r.get("timestamp", "")).replace("Z", "+00:00")))
        except Exception:
            ts.append(datetime.utcnow())

    sensors = [
        ("air_temperature",     "#60a5fa", 1, 1),
        ("process_temperature", "#f87171", 1, 2),
        ("rotational_speed",    "#34d399", 1, 3),
        ("torque",              "#fbbf24", 2, 1),
        ("tool_wear",           "#a78bfa", 2, 2),
    ]
    for key, colour, row, col in sensors:
        vals = [r.get(key) for r in readings]
        fig.add_trace(
            go.Scatter(x=ts, y=vals, mode="lines",
                       line=dict(color=colour, width=1.5), showlegend=False),
            row=row, col=col,
        )

    fig.update_layout(
        title=f"Sensor History — {machine_id} (last {len(readings)} readings)",
        template="plotly_dark", height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def build_anomaly_gauge(machine_id: str) -> go.Figure:
    prob, status, colour = 0.0, "UNKNOWN", "#6b7280"

    if machine_id and "no machines" not in machine_id.lower():
        machines = _get("/api/dashboard/machines").get("machines", [])
        m = next((x for x in machines if x["machine_id"] == machine_id), None)
        if m:
            prob   = m.get("failure_probability", 0.0)
            status = m.get("status", "NORMAL")
            colour = STATUS_COLOUR.get(status, "#6b7280")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 32}},
        title={"text": f"Failure Probability<br><b>{status}</b>", "font": {"size": 13}},
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
    fig.update_layout(template="plotly_dark", height=240,
                      margin=dict(l=20, r=20, t=40, b=10))
    return fig


def refresh_monitor(machine_id: str):
    return build_sensor_chart(machine_id), build_anomaly_gauge(machine_id)


# ── Tab 2: Alerts ──────────────────────────────────────────────────────────────

def fetch_alerts_table(sev_filter: str, machine_filter: str, status_filter: str):
    params: dict = {"limit": 50}
    if machine_filter.strip():
        params["machine_id"] = machine_filter.strip()
    if status_filter != "All":
        params["status"] = status_filter.lower()

    data   = _get("/api/alerts", params)
    alerts = data.get("alerts", [])

    if sev_filter != "All":
        alerts = [a for a in alerts
                  if a.get("root_cause_report", {}).get("severity") == sev_filter]

    rows = []
    for a in alerts:
        rcr      = a.get("root_cause_report", {})
        approved = a.get("approved")
        status   = "PENDING" if approved is None else ("APPROVED" if approved else "REJECTED")
        rc       = rcr.get("root_cause", "")
        rows.append([
            a.get("alert_id", "")[:8],
            a.get("machine_id", ""),
            rcr.get("severity", ""),
            rc[:60] + ("…" if len(rc) > 60 else ""),
            f"{rcr.get('confidence', 0):.0%}",
            status,
            str(a.get("created_at", ""))[:16],
        ])
    return rows


def get_pending_ids() -> list[str]:
    data = _get("/api/alerts", {"status": "pending", "limit": 20})
    return [a["alert_id"] for a in data.get("alerts", [])]


def approve_alert(alert_id: str):
    if not alert_id:
        return "Select an alert first.", gr.Dropdown(choices=get_pending_ids())
    resp = _post(f"/api/alerts/{alert_id}/approve", {"approved_by": "dashboard_user"})
    if "error" in resp:
        err = resp["error"]
        if "409" in err:
            msg = f"Already decided — click Reload to refresh the list."
        else:
            msg = f"Error: {err}"
    else:
        msg = f"Approved: {alert_id[:8]}…"
    return msg, gr.Dropdown(choices=get_pending_ids())


def reject_alert(alert_id: str, reason: str):
    if not alert_id:
        return "Select an alert first.", gr.Dropdown(choices=get_pending_ids())
    if not reason.strip():
        return "Enter a rejection reason.", gr.Dropdown(choices=get_pending_ids())
    resp = _post(f"/api/alerts/{alert_id}/reject",
                 {"rejection_reason": reason, "rejected_by": "dashboard_user"})
    if "error" in resp:
        err = resp["error"]
        msg = "Already decided — click Reload." if "409" in err else f"Error: {err}"
    else:
        msg = f"Rejected: {alert_id[:8]}…"
    return msg, gr.Dropdown(choices=get_pending_ids())


# ── Tab 3: Root Cause Analysis ─────────────────────────────────────────────────

def get_alert_choices() -> list[str]:
    data = _get("/api/alerts", {"limit": 30})
    return [
        f"{a['alert_id']} | {a.get('machine_id','')} | {a.get('root_cause_report',{}).get('severity','')}"
        for a in data.get("alerts", [])
    ]


def load_root_cause(choice: str):
    empty = ("", "", "", "", "")
    if not choice:
        return empty

    alert_id = choice.split(" | ")[0].strip()
    data     = _get(f"/api/alerts/{alert_id}")
    if "error" in data:
        return (f"Error: {data['error']}", "", "", "", "")

    rcr  = data.get("root_cause_report", {})
    expl = data.get("plain_language_explanation", "")

    summary = (
        f"**Root Cause:** {rcr.get('root_cause', 'N/A')}\n\n"
        f"**Severity:** {rcr.get('severity', '')}  |  "
        f"**Confidence:** {rcr.get('confidence', 0):.0%}\n\n"
        f"**Failure Type:** {rcr.get('anomaly_result', {}).get('failure_type_prediction', 'N/A')}"
    )

    evidence = "\n".join(f"• {e}" for e in rcr.get("evidence", [])) or "No evidence recorded."
    steps    = "\n\n".join(
        f"Step {i+1}: {s}" for i, s in enumerate(rcr.get("reasoning_steps", []))
    ) or "No reasoning trace recorded."
    memory   = "\n".join(f"• {m}" for m in rcr.get("agent_memory_used", [])) or "No memories recalled."

    return expl, summary, evidence, steps, memory


# ── Tab 4: System Health ───────────────────────────────────────────────────────

def load_health():
    health = _get("/health")
    stats  = _get("/api/dashboard/stats")

    icons = {True: "OK", False: "DOWN"}
    rows  = "\n".join(
        f"| {k.replace('_',' ').title()} | {icons.get(bool(v), str(v))} |"
        for k, v in health.items() if k not in ("status",)
    )
    svc_md = f"| Service | Status |\n|---------|--------|\n{rows}"

    dist = stats.get("failure_type_distribution", {})
    sev  = stats.get("alerts_by_severity", {})
    stats_md = (
        f"**Machines monitored:** {stats.get('total_machines_monitored', 0)}\n\n"
        f"**Anomalies (last 24h):** {stats.get('anomalies_last_24h', 0)}\n\n"
        f"**Pending alerts:** {stats.get('alerts_pending_approval', 0)}\n\n"
        f"**Avg resolution:** {stats.get('avg_resolution_time_minutes') or 'N/A'} min\n\n"
        f"**Failure types:**\n" + "\n".join(f"  - {k}: {v}" for k, v in dist.items()) + "\n\n"
        f"**By severity:**\n"  + "\n".join(f"  - {k}: {v}" for k, v in sev.items())
    )

    ml_md = (
        "**Models:** LSTM Autoencoder + Isolation Forest\n\n"
        "**Sequence length:** 30 readings\n\n"
        "**Detection:** High-conf = LSTM above threshold AND IForest = -1\n\n"
        "**Failure types:** TWF · HDF · PWF · OSF · RNF\n\n"
        "**Dataset:** AI4I 2020 — 10,000 rows · 339 failure rows (3.4%)"
    )

    return svc_md, stats_md, ml_md


# ── Build app ──────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks() as demo:

        gr.Markdown(
            "# DefectSense — Manufacturing Defect Intelligence\n"
            "> Hybrid ML + GenAI multi-agent system · Real-time anomaly detection & root cause analysis"
        )

        with gr.Tabs():

            # ── Tab 1 ──────────────────────────────────────────────────────────
            with gr.TabItem("Live Monitor"):
                with gr.Row():
                    machine_dd  = gr.Dropdown(label="Machine", choices=get_machine_list(),
                                              interactive=True, scale=3)
                    reload_btn  = gr.Button("Reload Machines", scale=1)
                    refresh_btn = gr.Button("Refresh Now", variant="secondary", scale=1)

                with gr.Row():
                    sensor_chart = gr.Plot(label="Sensor Readings", scale=3)
                    gauge_chart  = gr.Plot(label="Anomaly Score",    scale=1)

                t1 = gr.Timer(value=REFRESH_SEC)
                t1.tick(fn=refresh_monitor,  inputs=machine_dd, outputs=[sensor_chart, gauge_chart])
                refresh_btn.click(fn=refresh_monitor, inputs=machine_dd, outputs=[sensor_chart, gauge_chart])
                machine_dd.change(fn=refresh_monitor, inputs=machine_dd, outputs=[sensor_chart, gauge_chart])
                reload_btn.click(fn=lambda: gr.Dropdown(choices=get_machine_list()), outputs=machine_dd)

            # ── Tab 2 ──────────────────────────────────────────────────────────
            with gr.TabItem("Alerts"):
                with gr.Row():
                    sev_f    = gr.Dropdown(label="Severity", choices=["All","CRITICAL","HIGH","MEDIUM","LOW"],
                                          value="All", scale=1)
                    mach_f   = gr.Textbox(label="Machine ID", scale=1)
                    stat_f   = gr.Dropdown(label="Status", choices=["All","pending","approved","rejected"],
                                          value="All", scale=1)
                    filt_btn = gr.Button("Filter", variant="primary", scale=1)

                alerts_tbl = gr.Dataframe(
                    headers=["ID","Machine","Severity","Root Cause","Conf","Status","Created"],
                    interactive=False, wrap=True, label="Alerts",
                )

                t2 = gr.Timer(value=REFRESH_SEC)
                t2.tick(fn=fetch_alerts_table, inputs=[sev_f, mach_f, stat_f], outputs=alerts_tbl)
                filt_btn.click(fn=fetch_alerts_table, inputs=[sev_f, mach_f, stat_f], outputs=alerts_tbl)

                gr.Markdown("### Approve / Reject a pending alert")
                with gr.Row():
                    pending_dd   = gr.Dropdown(label="Pending Alert ID",
                                               choices=get_pending_ids(), interactive=True, scale=3)
                    reload_pend  = gr.Button("Reload", scale=1)

                with gr.Row():
                    approve_btn  = gr.Button("Approve", variant="primary", scale=1)
                    reject_text  = gr.Textbox(label="Rejection Reason", scale=3)
                    reject_btn   = gr.Button("Reject",  variant="stop",   scale=1)

                action_out = gr.Textbox(label="Result", interactive=False)
                approve_btn.click(fn=approve_alert, inputs=pending_dd,                    outputs=[action_out, pending_dd])
                reject_btn.click( fn=reject_alert,  inputs=[pending_dd, reject_text],     outputs=[action_out, pending_dd])
                reload_pend.click(fn=lambda: gr.Dropdown(choices=get_pending_ids()),       outputs=pending_dd)

            # ── Tab 3 ──────────────────────────────────────────────────────────
            with gr.TabItem("Root Cause Analysis"):
                with gr.Row():
                    alert_dd   = gr.Dropdown(label="Alert", choices=get_alert_choices(),
                                             interactive=True, scale=4)
                    reload_rca = gr.Button("Reload List", scale=1)
                    load_btn   = gr.Button("Load Analysis", variant="primary", scale=1)

                expl_box = gr.Textbox(label="Plain-Language Explanation", lines=3, interactive=False)

                with gr.Row():
                    summary_md = gr.Markdown()
                    evidence_box = gr.Textbox(label="Supporting Evidence", lines=6, interactive=False)

                with gr.Row():
                    steps_box  = gr.Textbox(label="ReAct Reasoning Steps",    lines=12, interactive=False)
                    memory_box = gr.Textbox(label="A-MEM Memories Recalled",  lines=6,  interactive=False)

                rca_outputs = [expl_box, summary_md, evidence_box, steps_box, memory_box]
                load_btn.click( fn=load_root_cause, inputs=alert_dd, outputs=rca_outputs)
                alert_dd.change(fn=load_root_cause, inputs=alert_dd, outputs=rca_outputs)
                reload_rca.click(fn=lambda: gr.Dropdown(choices=get_alert_choices()), outputs=alert_dd)

            # ── Tab 4 ──────────────────────────────────────────────────────────
            with gr.TabItem("System Health"):
                health_refresh_btn = gr.Button("Refresh Health", variant="primary")
                with gr.Row():
                    svc_md   = gr.Markdown(label="Services")
                    stats_md = gr.Markdown(label="Pipeline Stats")
                    ml_md    = gr.Markdown(label="ML Models")

                health_outputs = [svc_md, stats_md, ml_md]
                health_refresh_btn.click(fn=load_health, outputs=health_outputs)

                t4 = gr.Timer(value=30)
                t4.tick(fn=load_health, outputs=health_outputs)

        demo.load(fn=load_health, outputs=[svc_md, stats_md, ml_md])

    return demo


if __name__ == "__main__":
    print(f"DefectSense Dashboard — connecting to {API_BASE}")
    print("Open http://localhost:7860")
    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    )
