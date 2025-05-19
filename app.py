from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

# Groq API key setup
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY") or "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",
    base_url="https://api.groq.com/openai/v1"
)

@app.route('/')
def index():
    return "✅ Groq LLaMA 3 Backend is running."

@app.route('/submit', methods=['POST'])
def handle_form():
    data = request.get_json()

    person_name = data.get("person_name", "")
    relationship = data.get("relationship", "")
    interaction_goal = data.get("interaction_goal", "")
    compliment = data.get("compliment", "")
    curiosity = data.get("curiosity", "")

    if not all([person_name, relationship, interaction_goal, compliment, curiosity]):
        return jsonify({"error": "Missing one or more fields"}), 400

    # Read prompt from external file
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        return jsonify({"error": "prompt.txt file not found"}), 500

    # Inject user inputs into the prompt
    prompt = prompt_template.format(
        person_name=person_name,
        relationship=relationship,
        interaction_goal=interaction_goal,
        compliment=compliment,
        curiosity=curiosity
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



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
