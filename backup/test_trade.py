import requests
import json

# Execute a test trade
url = "http://localhost:8080/api/execute-trade"
data = {
    "pair": "EURUSD",
    "action": "BUY",
    "lot_size": 0.01
}

print("Executing test trade...")
response = requests.post(url, json=data)
print(f"Response status: {response.status_code}")
print(f"Response content: {response.text}")

# Get trading status
url = "http://localhost:8080/api/trading-status"
response = requests.get(url)
print("\nTrading Status:")
print(json.dumps(response.json(), indent=2))
