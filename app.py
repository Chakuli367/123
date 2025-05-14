import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)

# Load lightweight model
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = app.make_response()
        response.headers['Access-Control-Allow-Origin'] = 'https://goalgrid.wpcomstaging.com'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

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

    prompt = (
        f"The user's goal is: {goal}\n"
        "Please rewrite the following daily mission text to align with this goal.\n"
        f"Text to personalize:\n{base_text.strip()}"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=300,
            temperature=0.7,
            num_return_sequences=1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

