
import asyncio
import httpx
import time
import base64
import os
from statistics import mean

# ==== CONFIG ====
API_URL = "http://localhost:8000/api/search/base64"
# API_URL = "https://ccdf-103-197-153-124.ngrok-free.app/api/search/base64"
IMAGE_PATH = "polo.png"
NUM_REQUESTS = 100   # total requests
CONCURRENCY = 80     # simultaneous requests

# ==== Load and encode image ====
with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()
base64_image = base64.b64encode(image_bytes).decode()

# ==== Task ====
async def send_request(client):
    start_time = time.time()
    try:
        response = await client.post(API_URL, data={"base64_image": base64_image})
        elapsed = time.time() - start_time
        return response.status_code, elapsed
    except Exception as e:
        return 0, time.time() - start_time  # Treat exception as failed request

# ==== Main ====
async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [send_request(client) for _ in range(NUM_REQUESTS)]
        
        print(f"Sending {NUM_REQUESTS} requests with concurrency {CONCURRENCY}...")
        start = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start

        # Analysis
        response_times = [r[1] for r in results if r[0] == 200]
        failures = [r for r in results if r[0] != 200]

        print("\n📊 === Load Test Results ===")
        print(f"Total requests:     {NUM_REQUESTS}")
        print(f"Successful:         {len(response_times)}")
        print(f"Failed:             {len(failures)}")
        print(f"Total duration:     {total_time:.2f}s")
        if response_times:
            print(f"Average time:       {mean(response_times):.3f}s")
            print(f"Fastest response:   {min(response_times):.3f}s")
            print(f"Slowest response:   {max(response_times):.3f}s")

        # Optional: save results to file
        with open("load_results.csv", "w") as f:
            f.write("status,response_time\n")
            for status, elapsed in results:
                f.write(f"{status},{elapsed:.3f}\n")

if __name__ == "__main__":
    asyncio.run(main())
