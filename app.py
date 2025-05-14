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

    # REVISED PROMPT: Avoid repeating the goal or prompt instructions
    prompt_for_model = (
        f"The user's goal is: {goal}\n"
        "Provide a **short, actionable tip** to help them achieve it.\n"
        "Follow it with a **Daily Action List** containing 3-5 concrete steps they can take today toward their goal.\n"
        "Format the response exactly as follows:\n"
        "- Tip (preceded by the emoji 💡): short and actionable advice.\n"
        "- Daily Action List (preceded by 📅): numbered steps (3-5).\n"
        "Please avoid repeating the user's goal or your instructions. Return only the formatted output."
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

        # Remove the prompt entirely from the output wherever it appears
        if f"The user's goal is: {goal}" in generated:
            generated = generated.replace(f"The user's goal is: {goal}", "").strip()

        return jsonify({"response": generated})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

