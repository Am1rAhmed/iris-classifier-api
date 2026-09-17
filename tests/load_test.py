import asyncio
import httpx
import time
from app.config import settings 

URL = settings.API_URL
HEADERS = {"X-API-Key": settings.API_KEY}
PAYLOAD = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
NUM_REQUESTS = 100

async def send_request(client, i):
    start = time.time()
    try:
        response = await client.post(URL, json=PAYLOAD, headers=HEADERS)
        duration = time.time() - start
        return {"id": i, "status": response.status_code, "duration": duration}
    except Exception as e:
        return {"id": i, "status": "error", "error": str(e)}

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        start_time = time.time()
        tasks = [send_request(client, i) for i in range(NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    successes = [r for r in results if r["status"] == 200]
    failures = [r for r in results if r["status"] != 200]
    durations = [r["duration"] for r in successes]

    print(f"\n--- Load Test Results ---")
    print(f"Total requests: {NUM_REQUESTS}")
    print(f"Successful: {len(successes)}")
    print(f"Failed: {len(failures)}")
    print(f"Total time: {total_time:.2f}s")
    if durations:
        print(f"Avg response time: {sum(durations)/len(durations):.4f}s")
        print(f"Max response time: {max(durations):.4f}s")
        print(f"Min response time: {min(durations):.4f}s")
    if failures:
        print(f"Failure details: {failures[:5]}")

if __name__ == "__main__":
    asyncio.run(main())