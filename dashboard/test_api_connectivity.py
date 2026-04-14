"""Test API connectivity."""
import requests
import sys

API_URL = "http://localhost:8000/api"

print("\n" + "="*60)
print("API CONNECTIVITY TEST")
print("="*60)

endpoints = [
    ("/health", "Health Check"),
    ("/explanations/model-info", "Model Info"),
]

for endpoint, description in endpoints:
    url = API_URL + endpoint
    print(f"\n📡 Testing: {description}")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=5)
        status = response.status_code
        
        if status == 200:
            print(f"   ✅ SUCCESS (HTTP {status})")
            try:
                data = response.json()
                print(f"   Response: {str(data)[:100]}...")
            except:
                print(f"   Response: {response.text[:100]}...")
        else:
            print(f"   ⚠️  Status {status}: {response.text[:100]}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ CONNECTION ERROR")
        print(f"   Details: {e}")
    except requests.exceptions.Timeout:
        print(f"   ❌ TIMEOUT - Server not responding")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)
print("\nIF CONNECTION FAILS:")
print("1. Check if FastAPI backend is running:")
print("   - Look for another terminal/process running the API")
print("   - Run: python run_pipeline.py")
print("\n2. Check if API is listening on correct port:")
print("   - netstat -ano | findstr :8000")
print("\n3. Check firewall/network settings")
print("\nIP CONFIGURATION:")
import socket
try:
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(f"   Hostname: {hostname}")
    print(f"   Local IP: {ip}")
except Exception as e:
    print(f"   Error getting IP: {e}")
