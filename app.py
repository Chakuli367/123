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
    # Modify the prompt to include goal-specific action list generation
    prompt_for_model = (
        f"My goal is: {prompt}.\n\n"
        "Please respond in the following exact format:\n\n"
        "💡 Tip:\n"
        "[Write one short, practical, and encouraging tip to help me achieve this goal.]\n\n"
        "📅 Daily Action List:\n"
        "1. [Step 1 - clear, small action I can do today]\n"
        "2. [Step 2]\n"
        "3. [Step 3]\n"
        "4. [Step 4]\n"
        "5. [Step 5 - optional]\n\n"
        "Make it personalized, achievable, and motivating. Return only the tip and action list using the exact format above."
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

