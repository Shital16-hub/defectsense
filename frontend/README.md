---
title: DefectSense Dashboard
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
short_description: Manufacturing defect root-cause intelligence dashboard
---

# DefectSense Dashboard

Real-time manufacturing defect monitoring dashboard.

**Backend required:** Set `API_BASE` environment variable to your DefectSense API URL.

```
API_BASE=https://your-defectsense-api.railway.app
```

## Tabs

- **Live Monitor** — Real-time sensor charts + anomaly probability gauge
- **Alerts** — Pending alerts with approve/reject controls
- **Root Cause Analysis** — Full ReAct reasoning trace per alert
- **System Health** — Service status + pipeline statistics

## Source

[github.com/Shital16-hub/defectsense](https://github.com/Shital16-hub/defectsense)
