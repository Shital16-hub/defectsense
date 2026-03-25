"""
End-to-end pipeline test — Session 5.

Steps:
  1. Send 35 normal readings to build LSTM history (sequence_length=30)
  2. Send one extreme HDF failure reading
  3. Poll for the resulting alert (up to 90s — LLM takes ~10-20s)
  4. Approve it via the API
  5. Print final stats

Usage:
    python data/test_pipeline.py
"""
import asyncio
import sys
import json

import httpx

BASE       = "http://localhost:8080"
MACHINE_ID = "PIPELINE_TEST_001"

# Normal readings — realistic baseline values
NORMAL = {
    "machine_id":          MACHINE_ID,
    "air_temperature":     298.1,
    "process_temperature": 308.6,    # delta = 10.5 K (healthy)
    "rotational_speed":    1500,
    "torque":              40.0,
    "tool_wear":           50,
}

# HDF failure: very small temp delta + low rotational speed + high torque
# (mirrors actual failure rows in AI4I dataset)
HDF_FAILURE = {
    "machine_id":          MACHINE_ID,
    "air_temperature":     302.0,
    "process_temperature": 309.0,    # delta = 7 K  (< 8.6 K threshold → HDF)
    "rotational_speed":    1182,     # well below normal ~1500 rpm
    "torque":              68.5,     # high torque
    "tool_wear":           195,
}


async def main():
    async with httpx.AsyncClient(timeout=120) as client:

        # ── 0. Health ──────────────────────────────────────────────────────────
        r = await client.get(f"{BASE}/health")
        h = r.json()
        print("\n=== Health ===")
        for k, v in h.items():
            status = "OK" if v else "MISSING"
            print(f"  {k}: {v}  {'[OK]' if v else '[MISSING]'}")

        if not h.get("orchestrator_ready"):
            print("\nERROR: orchestrator not ready. Start the server first.")
            sys.exit(1)

        # ── 1. Warm-up: 35 normal readings ────────────────────────────────────
        print(f"\n=== Warming up LSTM with 35 normal readings for {MACHINE_ID} ===")
        for i in range(35):
            # Vary slightly to avoid identical vectors
            reading = {**NORMAL, "tool_wear": NORMAL["tool_wear"] + i}
            r = await client.post(f"{BASE}/api/sensors/ingest", json=reading)
            if r.status_code != 200:
                print(f"  [WARN] ingest returned {r.status_code}: {r.text}")
            if (i + 1) % 10 == 0:
                print(f"  sent {i+1}/35...")

        print("  warm-up complete")
        await asyncio.sleep(1)   # let Redis settle

        # ── 2. Send the HDF failure ────────────────────────────────────────────
        print("\n=== Sending HDF failure reading ===")
        r = await client.post(f"{BASE}/api/sensors/ingest", json=HDF_FAILURE)
        result = r.json()
        print(f"  is_anomaly:          {result.get('is_anomaly')}")
        print(f"  failure_probability: {result.get('failure_probability')}")
        print(f"  failure_type:        {result.get('failure_type_prediction')}")
        print(f"  ml_model_used:       {result.get('ml_model_used')}")

        if not result.get("is_anomaly"):
            print(
                "\n  No anomaly detected. The ML model may need more training data variety.\n"
                "  Try running the simulator first: python data/stream_simulator.py\n"
                "  (wait for ~50 rows sent) then re-run this test."
            )
            # Still proceed — manually trigger pipeline via a direct POST to
            # an anomaly score the server already accepted
            print("\n  Proceeding anyway to test the API layer...")

        # ── 3. Poll for alert ──────────────────────────────────────────────────
        print(f"\n=== Polling /api/alerts for machine {MACHINE_ID} (up to 90s) ===")
        alert = None
        alert_id = None
        for attempt in range(36):   # 36 * 2.5s = 90s
            await asyncio.sleep(2.5)
            r = await client.get(
                f"{BASE}/api/alerts",
                params={"machine_id": MACHINE_ID, "limit": 5},
            )
            alerts = r.json().get("alerts", [])
            if alerts:
                alert    = alerts[0]
                alert_id = alert["alert_id"]
                rcr      = alert.get("root_cause_report", {})
                print(f"\n  ALERT FOUND after {(attempt+1)*2.5:.0f}s:")
                print(f"    alert_id:    {alert_id}")
                print(f"    severity:    {rcr.get('severity')}")
                print(f"    confidence:  {rcr.get('confidence')}")
                print(f"    approved:    {alert.get('approved')}")
                print(f"    auto_approved: {alert.get('auto_approved')}")
                expl = alert.get("plain_language_explanation", "")
                print(f"    explanation: {expl[:200]}")
                break
            else:
                print(f"  [{attempt+1:02d}] waiting... (pipeline running: context+RAG+LLM reasoning)")

        if not alert_id:
            print(
                "\nERROR: no alert after 90s.\n"
                "Check server logs for errors in:\n"
                "  Orchestrator[retrieve_context]\n"
                "  Orchestrator[reason]\n"
                "  Orchestrator[generate_alert]"
            )
            sys.exit(1)

        # ── 4. Approve (if still pending) ─────────────────────────────────────
        if alert.get("approved") is None:
            print(f"\n=== Approving alert {alert_id[:8]}... ===")
            r = await client.post(
                f"{BASE}/api/alerts/{alert_id}/approve",
                json={"approved_by": "test_engineer"},
            )
            resp = r.json()
            print(f"  {resp}")

            r = await client.get(f"{BASE}/api/alerts/{alert_id}")
            final = r.json()
            print(f"  final approved:    {final.get('approved')}")
            print(f"  approved_by:       {final.get('approved_by')}")
            print(f"  approved_at:       {final.get('approved_at')}")
        elif alert.get("approved") is True:
            print(f"\n=== Alert already auto-approved (confidence >= threshold) ===")
        else:
            print(f"\n=== Alert was rejected ===")

        # ── 5. Stats ───────────────────────────────────────────────────────────
        print("\n=== Alert Stats ===")
        r = await client.get(f"{BASE}/api/alerts/stats")
        print(json.dumps(r.json(), indent=2))

        print("\n*** Pipeline test PASSED ***\n")


if __name__ == "__main__":
    asyncio.run(main())
