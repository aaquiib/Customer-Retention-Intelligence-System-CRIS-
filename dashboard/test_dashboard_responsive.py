"""Quick test that dashboard is responsive."""
import requests
import time

print("Testing Streamlit dashboard...")
time.sleep(2)  # Give it a moment to fully start

for i in range(3):
    try:
        response = requests.get("http://localhost:8501", timeout=3)
        if response.status_code == 200:
            print(f"✅ Dashboard is RESPONSIVE (HTTP {response.status_code})")
            print(f"✅ Available at: http://localhost:8501")
            print(f"\n✅ You can now access the dashboard!")
            break
    except requests.exceptions.ConnectionError:
        if i < 2:
            print(f"Attempt {i+1}: Waiting for app to start...")
            time.sleep(2)
        else:
            print("❌ Dashboard is not accessible")
    except requests.exceptions.Timeout:
        print("❌ Dashboard timed out (still loading)")
