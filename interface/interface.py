from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import requests

app = Flask(__name__)

# Define the base URL for the ngrok endpoint
NGROK_URL = "https://55ad-35-247-33-107.ngrok-free.app"  # Replace this with the new URL when it changes

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the ChatHistory model
class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sender = db.Column(db.String(50))
    message = db.Column(db.Text)
    def __repr__(self):
        return f"<ChatHistory {self.sender}: {self.message}>"

# Initialize the database
db.create_all()

def add_to_history(sender, message):
    entry = ChatHistory(sender=sender, message=message)
    db.session.add(entry)
    db.session.commit()

def search_history(input_text, limit=3):
    keywords = input_text.split()
    query = db.session.query(ChatHistory).filter(
        ChatHistory.message.ilike(f"%{keywords[0]}%")
    )
    for keyword in keywords[1:]:
        query = query.union_all(
            db.session.query(ChatHistory).filter(ChatHistory.message.ilike(f"%{keyword}%"))
        )
    query = query.order_by(ChatHistory.timestamp.desc()).limit(limit)
    return query.all()

def build_context(input_text, history_entries):
    context = ""
    for entry in history_entries:
        if entry.sender == 'user':
            context += f"User: {entry.message}\n"
        else:
            context += f"Seraphina: {entry.message}\n"
    context += f"User: {input_text}\nSeraphina:"
    return context

def chat_with_seraphina(message):
    if "/tarot" in message:
        context = f"User: {message}\nSeraphina:"
    else:
        history_entries = search_history(message)
        context = build_context(message, history_entries)
    url = f"{NGROK_URL}/generate"
    payload = {
        'message': context,
        'history': []  # Not sending history to the API
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get('response')
    else:
        print("Failed to get a response:", response.status_code, response.text)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    response = chat_with_seraphina(message)
    if response:
        if "/tarot" not in message:
            # Add user message and AI response to history
            add_to_history('user', message)
            add_to_history('ai', response)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'Failed to get a response from the server'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    search_query = request.args.get('query', '')
    if search_query:
        filtered_history = search_history(search_query)
    else:
        filtered_history = ChatHistory.query.all()
    return jsonify([{'timestamp': entry.timestamp, 'sender': entry.sender, 'message': entry.message} for entry in filtered_history])

if __name__ == '__main__':
    app.run(debug=True)
