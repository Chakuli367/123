import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

app = Flask(__name__)
CORS(app)

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"  # You can choose a different size/model if needed
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

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

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    # Generate the personalized text
    with torch.no_grad():
        output = model.generate(inputs['input_ids'], max_length=500, temperature=0.7, num_return_sequences=1)

    # Decode and clean up the output
    personalized_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return personalized_text

if __name__ == "__main__":
    app.run(debug=True)




