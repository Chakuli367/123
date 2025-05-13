import requests

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "llama3",  # Use the correct model name
    "prompt": "Give me one productivity tip.",
    "stream": False
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print(f"Failed to connect. Status code: {response.status_code}")
