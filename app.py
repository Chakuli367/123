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

    # Combine inputs into a user context prompt
    user_context = f"""
Person: {person_name}
Relationship: {relationship}
Goal for interaction: {interaction_goal}
Compliment: {compliment}
Curiosity: {curiosity}
"""

    prompt = f'''
You are a personal development coach. The following text is Day 1 of a user's self-improvement journey based on the principle of "Become Genuinely Interested in Others." Your task is to personalize the message, focusing on the user's specific context and interaction goals. Make the message feel human, warm, and practical — motivating them to take authentic action during their interaction with this person.

Here is the base text for Day 1:

---

Day 1 – Become Genuinely Interested in Others
Principle: Show genuine interest in people.

Action Plan:
Ask about their day — listen actively, don’t interrupt.
Remember and use name at least once in the convo.
Compliment someone on something you admire — like “their energy“.
Write down the names and one thing you learned about each person.
Why Show Sincere Interest?
Why Show Sincere Interest?
In a noisy world, sincere interest is rare — and powerful. It tells people, “You matter.” When you listen deeply and care about what excites others, you build trust and make them feel seen.

This opens doors, creates loyalty, and makes you truly unforgettable. People remember how you make them feel, and genuine interest builds a connection that lasts.

Warren Buffett
Warren Buffett’s Secret? Genuine Appreciation.
One of the world’s richest and most respected men, Warren Buffett credits much of his success not just to numbers, but to people. He’s known for writing heartfelt letters of praise to employees, remembering birthdays, and recognizing unsung heroes.

His philosophy? “Praise by name, criticize by category.” He understood what Dale Carnegie taught — when you make others feel important, sincerely, you win hearts and loyalty forever.

Consequences of Not Showing Interest
What Happens If You Don’t Show Sincere Interest?
When you fail to show genuine interest in others, people can sense it. They might feel ignored, unappreciated, or undervalued.

Without genuine interest, relationships can become shallow and distant. People are less likely to trust you or open up, which can hurt both personal and professional connections.

In the long run, not showing sincere interest might lead to missed opportunities, lost friendships, and a reputation of being self-centered or indifferent. BELIEVE ME YOU DON’T WANT TO BE LEFT ALONE
---

User's Context:
{user_context}

Personalized Day 1 Message:
'''

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

