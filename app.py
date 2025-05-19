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

    prompt = f"""
**Day 1 – Become Genuinely Interested in {person_name}**  
_Principle: Show genuine interest in people._

---

**Definitions of User Input:**  
- **Person**: {person_name}  
- **Relationship**: {relationship}  
- **Goal for interaction**: {interaction_goal}  
- **Compliment**: {compliment}  
- **Curiosity**: {curiosity}  

---

**Day 1 – Become Genuinely Interested in Others**  
_Principle: Show genuine interest in people._

**Action Plan:**  
Today, you’ll focus on {person_name}, your {relationship}. Your mission? Be *present*.  
Ask {person_name} how their day is going and really *listen* — no interruptions, no distractions.  
Use their name at least once during the conversation to build warmth and connection.  
Make sure to genuinely compliment them on something you admire — like how {compliment}.  
Then, ask about something you’re curious about — for example, {curiosity}.  

End the day by writing down what you learned about {person_name}. Let that learning deepen your connection tomorrow.

**Why Show Sincere Interest?**  
{person_name} is more than just a {relationship} — they’re someone who can help shape your growth.  
By showing true curiosity and appreciation, you’re saying: “You matter to me.”  
When someone feels seen and valued, they open up. That opens doors — to trust, to deeper bonds, and to unexpected growth.  

Sincere interest in {person_name} will leave a mark. They’ll remember how you made them feel, and in return, they may begin to see you as someone they truly *want* to be around.

**Warren Buffett’s Secret? Genuine Appreciation.**  
Even one of the richest men on Earth built his legacy through relationships, not just business savvy.  
Warren Buffett made it a point to express appreciation, write praise-filled letters, and remember the little things.  
Like Carnegie, he understood the magic of making people feel significant.  

When you compliment {person_name} on how {compliment}, you’re tapping into that same power.  
You’re building something real — and that’s worth more than any transaction.

**What Happens If You Don’t Show Sincere Interest?**  
Imagine if {person_name} walks away from today feeling unseen.  
They might feel like your {relationship} label is just surface-level. That stings — and it distances people.  
Without genuine interest, the connection weakens. They may not trust you. They may not open up.  

Over time, you risk being someone others simply tolerate — or worse, forget.  
BELIEVE ME YOU DON’T WANT TO BE LEFT ALONE.  
So today, take the leap. Make {person_name} feel important, for real.  
Because your journey toward self-growth starts not with yourself — but with how you treat others.

---

Now use the above text of Day 1 as format and create a similar  CUSTOMIZED lesson for the user and RETURN ONLY THE NEWLY CUSTOMIZED LESSON FOR THE USER


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
