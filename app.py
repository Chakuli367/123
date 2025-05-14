import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Hugging Face model and token
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")  # Replace this in production

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    goal = data.get("prompt", "").strip()

    if not goal:
        return jsonify({"error": "Prompt is required"}), 400

    # REVISED PROMPT: instruct format without inserting the actual words "💡 Tip:" or "📅 Daily Action List:" in the prompt
    prompt_for_model = (
        f"The user's goal is: {goal}\n"
        "Give a short, practical tip to help them achieve it.\n"
        "Then give a 3–5 step action list for today that will move them toward their goal.\n"
        "Format your response using the following structure:\n"
        "1. Start the tip section with the emoji and label for 'Tip'.\n"
        "2. Then, add a new section labeled 'Daily Action List' with numbered items.\n"
        "Only return the final formatted response. Do not repeat the user's goal or these instructions."
    )

    payload = {
        "inputs": prompt_for_model,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.3
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        generated = result[0].get("generated_text", "").strip()

        # Extract response starting from the first occurrence of 💡 Tip:
        tip_index = generated.find("💡 Tip:")
        if tip_index != -1:
            cleaned_output = generated[tip_index:].strip()
        else:
            cleaned_output = generated  # fallback if emoji is missing

        return jsonify({"response": cleaned_output})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
