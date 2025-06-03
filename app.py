from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Groq API client using environment variable
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def load_prompt(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

@app.route('/')
def index():
    return "✅ Groq LLaMA 3 Backend is running."

@app.route('/ask-questions', methods=['POST'])
def ask_questions():
    data = request.get_json()
    goal_name = data.get("goal_name", "").strip()

    if not goal_name:
        return jsonify({"error": "Missing goal_name"}), 400

    prompt_template = load_prompt("prompt_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_questions.txt file not found"}), 500

    prompt = prompt_template.format(goal_name=goal_name)

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )

        result = response.choices[0].message.content.strip()
        return jsonify({"questions": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/final-plan', methods=['POST'])
def final_plan():
    data = request.get_json()
    goal_name = data.get("goal_name", "").strip()
    user_answers = data.get("user_answers", [])

    if not goal_name or not isinstance(user_answers, list):
        return jsonify({"error": "Missing or invalid goal_name or user_answers"}), 400

    # Convert list of answers into a formatted string
    formatted_answers = "\n".join(
        [f"{i+1}. {answer.strip()}" for i, answer in enumerate(user_answers) if isinstance(answer, str)]
    )

    prompt_template = load_prompt("prompt_plan.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_plan.txt file not found"}), 500

    prompt = prompt_template.format(goal_name=goal_name, user_answers=formatted_answers)

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )

        result = response.choices[0].message.content.strip()
        return jsonify({"plan": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

