import httpx
import asyncio
import json

async def test():
    async with httpx.AsyncClient(timeout=60) as client:
        # Fill buffer with normal readings
        print("Filling buffer with 200 normal readings...")
        for i in range(200):
            payload = {
                "machine_id": "M001",
                "volt": 176.0,
                "rotate": 418.0,
                "pressure": 113.0,
                "vibration": 45.0
            }
            r = await client.post("http://localhost:8080/api/sensors/ingest", json=payload)
            await asyncio.sleep(0.05)  # 50ms between requests
            if i % 50 == 0:
                d = json.loads(r.text)
                print(f"  [{i}] ml_model_used: {d['ml_model_used']} | score: {d['anomaly_score']}")

        # Now send anomalous readings - vibration spike
        print()
        print("Sending anomalous readings (vibration spike to 150)...")
        for i in range(10):
            payload = {
                "machine_id": "M001",
                "volt": 176.0,
                "rotate": 418.0,
                "pressure": 113.0,
                "vibration": 150.0
            }
            r = await client.post("http://localhost:8080/api/sensors/ingest", json=payload)
            d = json.loads(r.text)
            print(f"  [{i}] ml_model_used: {d['ml_model_used']} | score: {d['anomaly_score']} | is_anomaly: {d['is_anomaly']} | failure_type: {d['failure_type_prediction']}")

asyncio.run(test())