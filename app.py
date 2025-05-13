import requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3"
PROMPT = "Give me one productivity tip."
TIMEOUT = 10  # seconds

def model_exists(model_name):
    try:
        tags_response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT)
        tags_response.raise_for_status()
        tags = tags_response.json().get("models", [])
        return any(tag.get("name") == model_name for tag in tags)
    except requests.exceptions.RequestException as e:
        print(f"Error checking model existence: {e}")
        return False

def generate_response(prompt):
    url = f"{OLLAMA_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": llama3,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        print("🧠 AI Response:", result.get("response", "No response field in JSON."))
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e} - Status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Is the Ollama server running at http://localhost:11434?")
    except requests.exceptions.Timeout:
        print("⌛ Request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except ValueError:
        print("⚠️ Failed to decode JSON response.")

# Optional: check if the model exists
if model_exists(MODEL_NAME):
    generate_response(PROMPT)
else:
    print(f"❌ Model '{MODEL_NAME}' not found in Ollama. Use `ollama run {MODEL_NAME}` first.")
