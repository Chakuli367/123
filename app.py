import os
from flask import Flask, jsonify, request
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Use environment variable if available, fallback to default
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY", 
    "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"
)

MODEL_NAME = "meta-llama/llama-3-8b-instruct"
TIMEOUT = 10
MAX_TOKENS = 500  # Stay under OpenRouter free tier limits

def generate_response(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.HTTPError as e:
        return f"❌ HTTP error: {e} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {e}"
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    ai_response = generate_response(prompt)
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)
