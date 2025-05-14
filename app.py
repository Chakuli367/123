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

    # Compose structured prompt to discourage model from echoing
    prompt_for_model = (
        f"You are a helpful assistant.\n\n"
        f"User's goal: {goal}\n\n"
        "Your task:\n"
        "- Do NOT repeat the user's goal.\n"
        "- Give exactly 1 short practical tip.\n"
        "- Then, provide a 3–5 step daily action list.\n"
        "- Format it EXACTLY like this:\n\n"
        "💡 Tip:\n"
        "Your tip here\n\n"
        "📅 Daily Action List:\n"
        "1. First actionable step\n"
        "2. Second step\n"
        "3. Third step\n"
        "4. Fourth (optional)\n"
        "5. Fifth (optional)\n\n"
        "Only return the response in this format. Do not repeat the user's goal or instructions."
    )

    payload = {
        "inputs": prompt_for_model,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.3
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()

        generated = result[0].get("generated_text", "").strip()

        # Strip echoed goal if included
        if goal in generated:
            generated = generated.split(goal, 1)[-1].strip()

        generated = generated.replace("User's goal:", "").strip()

        return jsonify({"response": generated})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

