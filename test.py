import requests

url = "http://127.0.0.1:5000/api/ml/analyze"

data = {
    "sample":"test"
}

response = requests.post(url, json=data)

print(response.json())