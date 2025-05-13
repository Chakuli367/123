from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask is working!"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    # You can do some processing here
    return jsonify({"message": "Data received", "data": data})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
