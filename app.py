import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Hugging Face API settings
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")  # Store securely

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

    # Base HTML block (this matches what your page shows)
    html_content = """
    <div>
        <p><strong>Principle:</strong> Show genuine interest in people.</p>
        <p>Tips for the day:</p>
        <ul>
            <li>Ask someone about their day and listen actively.</li>
            <li>Use their name during the conversation.</li>
            <li>Compliment something you genuinely admire.</li>
            <li>Reflect on what you learned about others today.</li>
        </ul>
    </div>
    """

    personalized_content = personalize_html_content(goal, html_content)
    return jsonify({"response": personalized_content})


def personalize_html_content(goal, html_content):
    instructions = (
        f"The user's goal is: {goal}\n"
        "Personalize the following HTML block to better fit this goal.\n"
        "Keep it actionable, motivational, and relevant to the user's intent.\n"
        f"HTML to personalize:\n{html_content}"
    )

    payload = {
        "inputs": instructions,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.3
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()
        return result[0].get("generated_text", "").strip()

    except Exception as e:
        return f"<div>Error generating content: {str(e)}</div>"

if __name__ == "__main__":
    app.run(debug=True)


