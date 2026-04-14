"""Test if model_info call is hanging."""
import time
from dashboard.utils.api_client import APIClient

print("Testing if get_model_info() hangs...")
print("Starting at:", time.strftime("%H:%M:%S"))

client = APIClient()

print("\nCalling get_model_info()...")
start = time.time()
try:
    success, data, error = client.get_model_info()
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Success: {success}")
    print(f"Error: {error}")
    if success:
        print(f"Data keys: {list(data.keys())[:5]}...")
except Exception as e:
    elapsed = time.time() - start
    print(f"EXCEPTION after {elapsed:.2f}s: {e}")
