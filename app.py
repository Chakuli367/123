import os
from flask import Flask, jsonify, request
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Use environment variable for OLLAMA_URL for flexibility
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # Default to localhost if not set
MODEL_NAME = "llama3"
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
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response field in JSON.")
    except requests.exceptions.HTTPError as e:
        return f"HTTP error: {e} - Status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ Connection error. Is the Ollama server running?"
    except requests.exceptions.Timeout:
        return "⌛ Request timed out."
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except ValueError:
        return "⚠️ Failed to decode JSON response."

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if not model_exists(MODEL_NAME):
        return jsonify({"error": f"Model '{MODEL_NAME}' not found. Run `ollama run {MODEL_NAME}` first."}), 404

    ai_response = generate_response(prompt)
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)
