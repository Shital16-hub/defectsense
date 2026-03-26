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


# ── Tab 5: Log Maintenance Report ─────────────────────────────────────────────

def submit_maintenance_log(
    machine_id: str,
    failure_type: str,
    machine_type: str,
    symptoms: str,
    root_cause: str,
    action_taken: str,
    resolution_hours: float,
    technician: str,
    notes: str,
):
    """Validate and submit a maintenance log via the API."""
    missing = []
    if not machine_id.strip():
        missing.append("Machine ID")
    if not symptoms.strip():
        missing.append("Symptoms")
    if not root_cause.strip():
        missing.append("Root Cause")
    if not action_taken.strip():
        missing.append("Action Taken")
    if not technician.strip():
        missing.append("Technician Name")

    if missing:
        msg = (
            f'<span style="color:#ef4444;font-weight:bold;">'
            f"Missing required fields: {', '.join(missing)}"
            f"</span>"
        )
        return msg, *_get_recent_logs_table()

    body = {
        "machine_id":            machine_id.strip(),
        "date":                  datetime.utcnow().isoformat(),
        "failure_type":          failure_type,
        "machine_type":          machine_type or None,
        "symptoms":              symptoms.strip(),
        "root_cause":            root_cause.strip(),
        "action_taken":          action_taken.strip(),
        "resolution_time_hours": float(resolution_hours or 0.0),
        "technician":            technician.strip(),
        "notes":                 notes.strip() or None,
    }

    resp = _post("/api/maintenance-logs/add", body)
    if "error" in resp:
        msg = (
            f'<span style="color:#ef4444;font-weight:bold;">'
            f"Submission failed: {resp['error']}"
            f"</span>"
        )
    else:
        log_id = resp.get("log_id", "unknown")
        msg = (
            f'<span style="color:#22c55e;font-weight:bold;">'
            f"Report submitted successfully. Log ID: {log_id}"
            f"</span>"
        )

    return msg, *_get_recent_logs_table()


def _get_recent_logs_table(limit: int = 5) -> tuple:
    """Fetch recent logs for the summary table. Returns (rows,) tuple."""
    data = _get("/api/maintenance-logs", {"limit": limit})
    logs = data.get("logs", []) if isinstance(data, dict) else []
    rows = []
    for lg in logs:
        date_raw = str(lg.get("date", lg.get("saved_at", "")))
        date_str = date_raw[:10] if date_raw else ""
        rc = lg.get("root_cause", "")
        rows.append([
            lg.get("machine_id", ""),
            lg.get("failure_type", ""),
            rc[:50] + ("…" if len(rc) > 50 else ""),
            lg.get("technician", ""),
            date_str,
        ])
    return (rows,)


def refresh_recent_logs():
    (rows,) = _get_recent_logs_table()
    return rows


def clear_form():
    """Return empty values for all form fields after successful submission."""
    return "", "HDF", "M", "", "", "", 0.0, "", ""


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


# ── Tab 6: Evaluation ──────────────────────────────────────────────────────────

def _score_colour(score: float) -> str:
    if score >= 0.7:
        return "#22c55e"   # green
    if score >= 0.4:
        return "#f97316"   # orange
    return "#ef4444"       # red


def _score_card(label: str, score: float) -> str:
    colour = _score_colour(score)
    pct    = f"{score * 100:.1f}%"
    return (
        f'<div style="border:1px solid #334155;border-radius:8px;padding:10px;'
        f'text-align:center;min-width:100px;">'
        f'<div style="font-size:11px;color:#94a3b8;">{label}</div>'
        f'<div style="font-size:22px;font-weight:bold;color:{colour};">{pct}</div>'
        f"</div>"
    )


def run_evaluation_now() -> str:
    resp = _get("/api/evaluation/run")
    if "error" in resp:
        return f"Error: {resp['error']}"
    return "✓ Evaluation started — refresh in ~2 minutes"


def load_evaluation_results():
    """Return (status_text, scores_html, history_rows, sample_rows)."""
    latest  = _get("/api/evaluation/latest")
    history = _get("/api/evaluation/history", {"limit": 20})

    # ── Status text ──────────────────────────────────────────────────────────
    rag_doc = latest.get("rag")   if isinstance(latest, dict) else None
    llm_doc = latest.get("llm_judge") if isinstance(latest, dict) else None

    if rag_doc:
        run_at   = rag_doc.get("run_at", "")[:19].replace("T", " ")
        status_t = f"Last evaluation: {run_at} UTC"
    else:
        status_t = "No evaluation results yet — click 'Run Evaluation Now'"

    # ── Score cards HTML ─────────────────────────────────────────────────────
    cards = []
    if rag_doc:
        s = rag_doc.get("scores", {})
        cards.append("<b>RAG Pipeline</b>&nbsp;&nbsp;")
        for label, key in [
            ("Context Precision", "context_precision"),
            ("Faithfulness",      "faithfulness"),
            ("Answer Relevancy",  "answer_relevancy"),
        ]:
            cards.append(_score_card(label, s.get(key, 0.0)))
        cards.append("&nbsp;&nbsp;<b>LLM Judge</b>&nbsp;&nbsp;")
    if llm_doc:
        s = llm_doc.get("scores", {})
        for label, key in [
            ("Root Cause",   "root_cause_correctness"),
            ("Severity",     "severity_accuracy"),
            ("Actions",      "action_quality"),
            ("Reasoning",    "reasoning_quality"),
            ("Calibration",  "confidence_calibration"),
        ]:
            cards.append(_score_card(label, s.get(key, 0.0)))
    scores_html = (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">'
        + "".join(cards)
        + "</div>"
    ) if cards else "<i>No scores yet.</i>"

    # ── History table ─────────────────────────────────────────────────────────
    history_rows = []
    for r in (history.get("results", []) if isinstance(history, dict) else []):
        s      = r.get("scores", {})
        run_at = r.get("run_at", "")[:19].replace("T", " ")
        et     = r.get("eval_type", "")
        if et == "rag":
            label   = "RAG"
            cp      = f"{s.get('context_precision', 0) * 100:.1f}%"
            fa      = f"{s.get('faithfulness',      0) * 100:.1f}%"
            ar      = f"{s.get('answer_relevancy',  0) * 100:.1f}%"
            overall = f"{s.get('overall', 0) * 100:.1f}%"
        else:
            label   = "LLM Judge"
            cp      = "—"
            fa      = "—"
            ar      = "—"
            overall = f"{s.get('overall', 0) * 100:.1f}%"
        history_rows.append([run_at, label, cp, fa, ar, overall])

    # ── Sample scores from latest RAG eval ────────────────────────────────────
    sample_rows = []
    if rag_doc:
        for s in rag_doc.get("sample_scores", []):
            sample_rows.append([
                s.get("machine_id", ""),
                f"{s.get('context_precision', 0)*100:.1f}%",
                f"{s.get('faithfulness',      0)*100:.1f}%",
                f"{s.get('answer_relevancy',  0)*100:.1f}%",
            ])

    return status_t, scores_html, history_rows, sample_rows


def refresh_evaluation():
    return load_evaluation_results()


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

            # ── Tab 5 ──────────────────────────────────────────────────────────
            with gr.TabItem("Log Maintenance Report"):
                gr.Markdown(
                    "### Submit a Maintenance Report\n"
                    "Log a resolved incident to keep the RAG knowledge base up to date. "
                    "Submitted reports are immediately searchable by the AI system."
                )

                with gr.Row():
                    t5_machine_id  = gr.Textbox(label="Machine ID *", placeholder="e.g. M042", scale=2)
                    t5_failure_type = gr.Dropdown(
                        label="Failure Type *",
                        choices=["TWF", "HDF", "PWF", "OSF", "RNF"],
                        value="HDF", scale=1,
                    )
                    t5_machine_type = gr.Dropdown(
                        label="Machine Type",
                        choices=["L", "M", "H"],
                        value="M", scale=1,
                    )

                t5_symptoms    = gr.Textbox(label="Symptoms *",    lines=3,
                                            placeholder="Observable symptoms before/during failure")
                t5_root_cause  = gr.Textbox(label="Root Cause *",  lines=3,
                                            placeholder="Identified root cause of the failure")
                t5_action_taken = gr.Textbox(label="Action Taken *", lines=3,
                                             placeholder="Maintenance action performed to resolve the issue")

                with gr.Row():
                    t5_resolution_hours = gr.Number(
                        label="Resolution Hours", value=0.0, minimum=0, maximum=720, scale=1,
                    )
                    t5_technician = gr.Textbox(label="Technician Name *",
                                               placeholder="e.g. J. Smith", scale=2)

                t5_notes = gr.Textbox(label="Notes (optional)", lines=2,
                                      placeholder="Any additional observations or context")

                t5_submit_btn = gr.Button("Submit Maintenance Report", variant="primary")

                t5_status = gr.HTML(value="")

                gr.Markdown("---\n### Recent Reports (last 5)")
                t5_recent_tbl = gr.Dataframe(
                    headers=["Machine ID", "Failure Type", "Root Cause", "Technician", "Date"],
                    interactive=False,
                    wrap=True,
                    label="Recent Maintenance Logs",
                )

                # Form inputs list — used for both submit and clear
                _t5_inputs = [
                    t5_machine_id, t5_failure_type, t5_machine_type,
                    t5_symptoms, t5_root_cause, t5_action_taken,
                    t5_resolution_hours, t5_technician, t5_notes,
                ]
                _t5_outputs = [t5_status, t5_recent_tbl]

                def _submit_and_maybe_clear(*args):
                    status_html, rows = submit_maintenance_log(*args)
                    # If submission succeeded (green), also clear the form
                    if "22c55e" in status_html:
                        return status_html, rows, "", "HDF", "M", "", "", "", 0.0, "", ""
                    return status_html, rows, *args

                t5_submit_btn.click(
                    fn=submit_maintenance_log,
                    inputs=_t5_inputs,
                    outputs=_t5_outputs,
                )
                t5_submit_btn.click(
                    fn=refresh_recent_logs,
                    outputs=t5_recent_tbl,
                )

            # ── Tab 6 ──────────────────────────────────────────────────────────
            with gr.TabItem("Evaluation"):
                gr.Markdown(
                    "### LLM-as-Judge Evaluation\n"
                    "Scores the RAG pipeline (context precision, faithfulness, answer relevancy) "
                    "and root-cause reports (correctness, severity, actions, reasoning, calibration) "
                    "using Groq llama-3.3-70b as judge. Runs nightly at 02:00 UTC."
                )

                with gr.Row():
                    t6_run_btn     = gr.Button("Run Evaluation Now", variant="primary", scale=1)
                    t6_refresh_btn = gr.Button("Refresh Results",    variant="secondary", scale=1)
                    t6_status      = gr.Textbox(label="Status", interactive=False, scale=3)

                t6_scores_html = gr.HTML(value="<i>Loading…</i>")

                gr.Markdown("#### Evaluation History")
                t6_history_tbl = gr.Dataframe(
                    headers=["Run Time", "Type", "Ctx Precision", "Faithfulness", "Ans Relevancy", "Overall"],
                    interactive=False,
                    wrap=True,
                    label="History (last 20)",
                )

                gr.Markdown("#### Sample Scores — Latest RAG Evaluation")
                t6_sample_tbl = gr.Dataframe(
                    headers=["Machine ID", "Context Precision", "Faithfulness", "Answer Relevancy"],
                    interactive=False,
                    wrap=True,
                    label="Per-Alert RAG Scores",
                )

                _t6_outputs = [t6_status, t6_scores_html, t6_history_tbl, t6_sample_tbl]

                def _run_and_refresh():
                    msg = run_evaluation_now()
                    return (msg,) + tuple(load_evaluation_results()[1:])

                t6_run_btn.click(fn=_run_and_refresh,     outputs=_t6_outputs)
                t6_refresh_btn.click(fn=refresh_evaluation, outputs=_t6_outputs)

        demo.load(fn=load_health, outputs=[svc_md, stats_md, ml_md])
        demo.load(fn=refresh_recent_logs, outputs=t5_recent_tbl)
        demo.load(fn=refresh_evaluation,  outputs=[t6_status, t6_scores_html, t6_history_tbl, t6_sample_tbl])

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
