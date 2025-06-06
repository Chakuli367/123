from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os, json, datetime

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

LOG_FILE = "logs.json"

def load_prompt(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def save_log(entry):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

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
    avatar = data.get("avatar", "Unknown")
    user_email = data.get("user_email", "anonymous@example.com")
    label = data.get("label", f"Session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

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

        try:
            parsed_plan = json.loads(result)
        except Exception:
            return jsonify({
                "error": "Failed to parse AI response as JSON",
                "raw_response": result
            }), 500

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_email": user_email,
            "goal_name": goal_name,
            "answers": user_answers,
            "avatar": avatar,
            "label": label,
            "ai_response": parsed_plan
        }
        save_log(log_entry)

        return jsonify({"plan": parsed_plan})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/view-logs', methods=['POST'])
def view_logs():
    data = request.get_json()
    email = data.get("user", "").strip().lower()

    if not email:
        return jsonify({"error": "Missing user email"}), 400

    if not os.path.exists(LOG_FILE):
        return jsonify({"logs": []})

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError:
            return jsonify({"logs": []})

    user_logs = [log for log in logs if log.get("user_email", "").lower() == email]

    return jsonify({"logs": user_logs})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
