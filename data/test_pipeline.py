"""
End-to-end pipeline test — Session 5.

Sends one known HDF failure reading directly to the live server and
polls for the resulting alert, then approves it.

Usage:
    python data/test_pipeline.py
"""
import asyncio
import time
import sys

import httpx

BASE = "http://localhost:8080"

# Known HDF failure row values from AI4I dataset
HDF_READING = {
    "machine_id":          "TEST_M001",
    "air_temperature":     300.6,
    "process_temperature": 312.1,   # +11.5 K delta triggers HDF
    "rotational_speed":    1305,
    "torque":              52.1,
    "tool_wear":           108,
    "timestamp":           "2026-03-25T10:00:00Z",
}


async def main():
    async with httpx.AsyncClient(timeout=120) as client:

        # ── 1. Health check ────────────────────────────────────────────────────
        r = await client.get(f"{BASE}/health")
        health = r.json()
        print("\n=== Health ===")
        for k, v in health.items():
            print(f"  {k}: {v}")
        if not health.get("orchestrator_ready"):
            print("\nERROR: orchestrator not ready — is the server running?")
            sys.exit(1)

        # ── 2. Ingest a failure reading ────────────────────────────────────────
        print("\n=== Ingesting HDF failure reading ===")
        r = await client.post(f"{BASE}/api/sensors/ingest", json=HDF_READING)
        result = r.json()
        print(f"  anomaly detected: {result.get('is_anomaly')}")
        print(f"  failure_probability: {result.get('failure_probability')}")
        print(f"  failure_type: {result.get('failure_type_prediction')}")

        if not result.get("is_anomaly"):
            print("\n  No anomaly — pipeline won't run. Try increasing process_temperature.")
            print("  (The ML model needs 30 readings of history first — try after running the simulator for a bit)")
            sys.exit(0)

        # ── 3. Poll for alert (up to 60s — LLM calls take time) ───────────────
        print("\n=== Polling for alert (up to 60s) ===")
        alert_id = None
        for attempt in range(24):   # 24 * 2.5s = 60s
            await asyncio.sleep(2.5)
            r = await client.get(f"{BASE}/api/alerts", params={"machine_id": "TEST_M001", "limit": 5})
            alerts = r.json().get("alerts", [])
            if alerts:
                alert_id = alerts[0]["alert_id"]
                alert = alerts[0]
                print(f"  [attempt {attempt+1}] ALERT FOUND: {alert_id[:8]}...")
                print(f"    severity:  {alert.get('root_cause_report', {}).get('severity')}")
                print(f"    approved:  {alert.get('approved')}")
                print(f"    explanation: {alert.get('plain_language_explanation', '')[:120]}")
                break
            else:
                print(f"  [attempt {attempt+1}] no alert yet...")

        if not alert_id:
            print("\nERROR: no alert created after 60s — check server logs for errors")
            sys.exit(1)

        # ── 4. Approve the alert ───────────────────────────────────────────────
        if alert.get("approved") is None:
            print(f"\n=== Approving alert {alert_id[:8]} ===")
            r = await client.post(
                f"{BASE}/api/alerts/{alert_id}/approve",
                json={"approved_by": "test_engineer"},
            )
            print(f"  response: {r.json()}")

            # Verify
            r = await client.get(f"{BASE}/api/alerts/{alert_id}")
            final = r.json()
            print(f"  final approved: {final.get('approved')}")
            print(f"  approved_by:    {final.get('approved_by')}")
        else:
            print(f"\n=== Alert already decided: approved={alert.get('approved')} (auto) ===")

        # ── 5. Stats ───────────────────────────────────────────────────────────
        print("\n=== Alert Stats ===")
        r = await client.get(f"{BASE}/api/alerts/stats")
        import json
        print(json.dumps(r.json(), indent=2))

        print("\nPipeline test PASSED")


if __name__ == "__main__":
    asyncio.run(main())
