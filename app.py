from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

# Create Flask app
app = Flask(__name__)
CORS(app)

# Load Groq API credentials
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or "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"
GROQ_API_BASE = "https://api.groq.com/openai/v1"

openai.api_key = GROQ_API_KEY
openai.api_base = GROQ_API_BASE

@app.route("/")
def index():
    return "✅ Groq LLaMA 3 Backend is running."

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "").strip()

        if not user_input:
            return jsonify({"error": "Prompt is required"}), 400

        # Personalization prompt
        system_prompt = f"""
You are a personal development coach. Personalize the following Day 1 text based on the user's specific interests, goals, and current challenges. Make it practical, motivating, and aligned with their journey.

Here is the base text for Day 1:

---

**Day 1: Become Genuinely Interested in Others**

Principle: Show genuine interest in people.

Today’s principle is about making authentic connections with others. Building relationships that are meaningful starts with curiosity about others. People appreciate when we take the time to listen, remember their names, and show interest in their lives. 

**Action Plan:**
1. **Ask about their day** — Show genuine curiosity by asking how their day went. Make sure to listen actively without interrupting.
2. **Remember and use their name** — It’s a small but powerful gesture to use someone’s name during conversation. It helps make the interaction more personal and shows you value them.
3. **Compliment someone** — Compliment them on something you admire. For example, you might say, “I really admire your energy today!” or “Your positivity is contagious.”
4. **Write it down** — After each interaction, jot down the names and one thing you learned about the person. This will help you remember and appreciate them even more.

---

User Input: {user_input}

Personalized Day 1 Message:
"""

        # Call Groq API (OpenAI-compatible)
        response = openai.ChatCompletion.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.3,
            max_tokens=300
        )

        result = response['choices'][0]['message']['content'].strip()
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))  # Render sets PORT
    app.run(host="0.0.0.0", port=port)

