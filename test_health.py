import requests
import time

url = "http://localhost:8000/health"

for i in range(60):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            print(f"Server is ready: {r.json()}")
            break
    except:
        print(f"Waiting for server... ({i+1}/60)")
        time.sleep(5)
else:
    print("Server did not start within 5 minutes")
