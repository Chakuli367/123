import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "https://goalgrid.wpcomstaging.com"}})

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint

@app.route('/')
def index():
    return "LLaMA 3 Backend is running."

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_input = data.get("prompt", "").strip()

    if not user_input:
        return jsonify({"error": "Prompt is required"}), 400

    instructions = f'''
You are a personal development coach. The following text is Day 1 of a user's self-improvement journey based on the principle of "Become Genuinely Interested in Others." Your task is to personalize the message, focusing on the user's specific interests, goals, and current challenges. You should make the message feel engaging and practical, motivating them to take action. Make sure the action plan is personalized and aligned with their personal journey.

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
'''

    try:
        payload = {
            "model": "llama3",
            "prompt": instructions,
            "temperature": 0.3,
            "max_tokens": 250,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        ai_output = result.get("response", "").strip()
        return jsonify({"response": ai_output})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    port = 5000
    app.run(host="0.0.0.0", port=port)
