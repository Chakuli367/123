import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Hugging Face model info
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")  # Secure this!

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

@app.route('/personalize', methods=['POST'])
def personalize():
    data = request.get_json()
    goal = data.get("goal", "").strip()
    html_content = data.get("html_content", "").strip()

    if not goal or not html_content:
        return jsonify({"error": "Both 'goal' and 'html_content' are required"}), 400

    # Generate personalized content based on the goal and existing HTML block content
    personalized_content = personalize_html_content(goal, html_content)

    return jsonify({"response": personalized_content})

def personalize_html_content(goal, html_content):
    # Create personalized prompt for the model based on the user's goal and HTML content
    instructions = (
        f"The user's goal is: {goal}\n"
        "Personalize the following HTML content based on this goal.\n"
        "You should focus on providing **actionable advice** and make sure the content aligns with the user's objective.\n"
        "Here's the existing content:\n"
        f"{html_content}\n"
        "Update the content to make it more relevant and tailored to the goal.\n"
        "Ensure the new content is actionable and motivating.\n"
        "Format the response exactly as the original HTML block, but with personalized content."
    )

    payload = {
        "inputs": instructions,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.3
        }
    }

    # Request personalized content from the model
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        raw_output = result[0].get("generated_text", "").strip()

        # Return the personalized HTML content
        return raw_output

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)


