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

    prompt = """
You are a helpful AI coach that generates personalized, gamified social skill lessons. You are NOT just a fill-in-the-blank engine. Before generating the output, you must understand the user's emotional and social context based on their inputs.

---

### STEP 1: DEEPLY UNDERSTAND THE USER INPUT

Use the following fields to infer the user's social situation, emotional intent, and type of relationship.  
- **Person's Name**: {person_name}  
- **Relationship**: {relationship}  
- **Goal for Interaction**: {interaction_goal}  
- **Compliment**: {compliment}  
- **Curiosity**: {curiosity}

→ Before writing the lesson, internally consider:
- What emotional or relational challenge might the user be facing?
- Why might this person be important to the user’s growth?
- What kind of tone and style would best encourage this user?
- How can we make the interaction feel like an engaging quest rather than a task?

---

### STEP 2: GENERATE A GAMIFIED DAY 1 LESSON

Title:  
🎯 **Day 1 Quest: Unlock a Stronger Bond with [person_name]**  
_Principle: Show genuine interest in others_

---

**🗺️ Mission Briefing:**  
Set the scene. This is a micro-quest. Help the user get emotionally invested in improving their connection with [person_name]. Give it a sense of purpose and play.

---

✅ **Action Plan (Your Moves for Today):**  
List 4–5 specific, gamified actions. Use emojis and short sentences:
- Start a conversation and be fully present  
- Use their name at least once  
- Give a compliment based on {compliment}  
- Ask a sincere question based on {curiosity}  
- Reflect at the end of the day on what you learned  

Each step should feel rewarding, doable, and socially meaningful.

---

🧠 **Why This Quest Matters**  
Explain the emotional payoff. Show how genuine interest helps build bonds, trust, and social confidence. Make it feel human, not transactional.

---

⚠️ **If You Skip This Quest...**  
Lightly describe the emotional or social cost of inaction — e.g., missed opportunity for connection, staying invisible, or reinforcing distance. Use metaphors like "staying in NPC mode" or "leaving this chapter unfinished."

---

🧭 **Bonus XP: Reflection Questions**  
Offer 2–3 reflection questions to help the user process their experience:
- What did you learn about [person_name]?  
- How did they respond to your curiosity or compliment?  
- What surprised you?

---

### STYLE & TONE
- Think Duolingo meets therapy  
- Playful, light, but emotionally intelligent  
- Speak directly to the user ("you")  
- Keep it under ~600 words  
- Never copy a template. Always respond uniquely to the context.

REMEMBER: This is not about giving commands. It’s about helping the user feel confident, excited, and emotionally safe while practicing a real-life social interaction.
"""


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
