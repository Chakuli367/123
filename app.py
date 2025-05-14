import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Hugging Face model and token
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")  # Replace or use .env

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Modify the prompt to include goal-specific action list generation
    prompt_for_model = (
        f"My goal is: {prompt}. "
        "Generate a simple daily action list with 3 to 5 practical steps I can take today to move toward this goal. "
        "Make it encouraging, achievable, and personalized. Return only the action list."
    )

    payload = {
        "inputs": prompt_for_model,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        generated = result[0].get("generated_text", "⚠️ No output generated.")
        return jsonify({"response": generated})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e), "details": response.text}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

