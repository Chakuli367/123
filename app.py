import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Hugging Face API info
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")  # Caution: Use env var securely

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

@app.route('/personalize', methods=['POST'])
def personalize():
    data = request.get_json()
    goal = data.get("goal", "").strip()

    if not goal:
        return jsonify({"error": "'goal' is required"}), 400

    base_text = (
        "Day 1 – Become Genuinely Interested in Others\n"
        "Principle: Show genuine interest in people.\n\n"
        "Action Plan:\n"
        "Ask about their day — listen actively, don’t interrupt.\n"
        "Remember and use their name at least once in the conversation.\n"
        "Compliment someone on something you admire — like “their energy“.\n"
        "Write down the names and one thing you learned about each person."
    )

    personalized_content = personalize_text(goal, base_text)
    return jsonify({"response": personalized_content})


def personalize_text(goal, base_text):
    prompt = (
        f"The user's goal is: {goal}\n"
        "Please rewrite the following daily mission text to make it more personal and aligned with this goal.\n"
        "Keep the core principle but adjust the action steps and tone so that it's directly relevant and motivating for the user.\n\n"
        f"Text to personalize:\n{base_text.strip()}"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.4
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        full_text = result[0].get("generated_text", "").strip()

        # Remove the prompt from the output
        if full_text.startswith(prompt):
            ai_only_response = full_text[len(prompt):].strip()
        else:
            ai_only_response = full_text

        return ai_only_response

    except Exception as e:
        return f"Error generating content: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)



