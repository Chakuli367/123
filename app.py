from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Groq API key setup from environment variable
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

@app.route('/')
def index():
    return "✅ Groq LLaMA 3 Backend is running."

@app.route('/submit', methods=['POST'])
def handle_form():
    data = request.get_json()

    goal_name = data.get("goal_name", "")
    why_it_matters = data.get("why_it_matters", "")
    current_obstacle = data.get("current_obstacle", "")
    available_time = data.get("available_time", "")
    desired_outcome = data.get("desired_outcome", "")

    if not all([goal_name, why_it_matters, current_obstacle, available_time, desired_outcome]):
        return jsonify({"error": "Missing one or more fields"}), 400

    # Read prompt from external file
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        return jsonify({"error": "prompt.txt file not found"}), 500

    # Inject user inputs into the prompt
    prompt = prompt_template.format(
        goal_name=goal_name,
        why_it_matters=why_it_matters,
        current_obstacle=current_obstacle,
        available_time=available_time,
        desired_outcome=desired_outcome
    )

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )

        result = response.choices[0].message.content.strip()
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
