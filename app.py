from flask import Flask
import requests

app = Flask(__name__)

# Function to generate advice using local LLaMA 3 via Ollama
def generate_advice(goal):
    prompt = f"Give practical, encouraging advice to help someone achieve this goal: '{goal}'"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json().get("response", "No response from model.")
        else:
            return f"Error from LLaMA model: {response.text}"
    except Exception as e:
        return f"Error connecting to LLaMA: {str(e)}"

@app.route('/')
def index():
    test_goal = "I want to wake up at 5 AM every day and build a consistent morning routine."
    advice = generate_advice(test_goal)
    return f"<h2>Goal:</h2><p>{test_goal}</p><h2>Advice:</h2><p>{advice}</p>"

if __name__ == '__main__':
    app.run(debug=True)
