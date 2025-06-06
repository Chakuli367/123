from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Groq API client
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

def save_user_data(goal_name, user_answers, user_info, ai_response):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user": user_info,
        "goal_name": goal_name,
        "answers": user_answers,
        "ai_response": ai_response
    }

    logs_path = "user_logs.json"

    # Append to file (one JSON object per line)
    with open(logs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

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

    formatted_answers = "\n".join(
        [f"{i+1}. {answer.strip()}" for i, answer in enumerate(user_answers) if isinstance(answer, str)]
    )

    prompt_template = load_prompt("prompt_plan.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_plan.txt file not found"}), 500

    prompt = prompt_template.replace("<<goal_name>>", goal_name).replace("<<user_answers>>", formatted_answers)

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1600
        )

        result = response.choices[0].message.content.strip()

        # Try parsing as JSON
        try:
            parsed_plan = json.loads(result)
        except json.JSONDecodeError as json_err:
            return jsonify({
                "error": f"Failed to parse plan as JSON: {str(json_err)}",
                "raw_response": result
            }), 500

        # Get user info from headers (set this on frontend)
        user_info = request.headers.get("X-User-Name", "Anonymous")

        # Save everything to file
        save_user_data(goal_name, user_answers, user_info, parsed_plan)

        return jsonify({"plan": parsed_plan})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
