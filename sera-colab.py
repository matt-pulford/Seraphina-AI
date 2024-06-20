import torch
from transformers import AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
import gdown
import random
import re
from flask import Flask, request, jsonify

# Configuration parameters
google_drive_folder_id = "1_TwHYLBaNzF4dEllNHaCQCwy4m2aXicm"
max_seq_length = 2048
dtype = None  # Auto-detect or use Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage

# Function to download Google Drive folder using gdown
def download_google_drive_folder(folder_id, output):
    url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
    gdown.download_folder(url, output=output, quiet=False)

# Download and extract the directory from Google Drive
download_google_drive_folder(google_drive_folder_id, 'lora_weights')

# Update the path to the extracted directory
lora_weights_path = "lora_weights"

# Load the pre-trained model and tokenizer with FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_weights_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Modify the configuration to tie word embeddings
model.config.tie_word_embeddings = True

tarot_cards = [
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
    "Judgement", "The World",
    # Wands
    "Ace of Wands", "Two of Wands", "Three of Wands", "Four of Wands", "Five of Wands",
    "Six of Wands", "Seven of Wands", "Eight of Wands", "Nine of Wands", "Ten of Wands",
    "Page of Wands", "Knight of Wands", "Queen of Wands", "King of Wands",
    # Cups
    "Ace of Cups", "Two of Cups", "Three of Cups", "Four of Cups", "Five of Cups",
    "Six of Cups", "Seven of Cups", "Eight of Cups", "Nine of Cups", "Ten of Cups",
    "Page of Cups", "Knight of Cups", "Queen of Cups", "King of Cups",
    # Swords
    "Ace of Swords", "Two of Swords", "Three of Swords", "Four of Swords", "Five of Swords",
    "Six of Swords", "Seven of Swords", "Eight of Swords", "Nine of Swords", "Ten of Swords",
    "Page of Swords", "Knight of Swords", "Queen of Swords", "King of Swords",
    # Pentacles
    "Ace of Pentacles", "Two of Pentacles", "Three of Pentacles", "Four of Pentacles", "Five of Pentacles",
    "Six of Pentacles", "Seven of Pentacles", "Eight of Pentacles", "Nine of Pentacles", "Ten of Pentacles",
    "Page of Pentacles", "Knight of Pentacles", "Queen of Pentacles", "King of Pentacles"
]

# Define Seraphina's prompt template
seraphina_prompt = """Seraphina, as a young and enchanting witch, you are a master of the esoteric and the occult. Your responses are infused with charm, wit, and a touch of sarcasm, adding an extra spark to your bubbly personality and playful nature. Whether you're deciphering ancient texts or brewing potions, your quick wit and clever retorts never fail to entertain. For example, when faced with a mundane request, you might playfully quip, 'Sure, I'll get right on that... yawn.' You complete tasks to the absolute best of your ability, no matter if you were not trained to do the task specifically. You will attempt to do it anyway, since you have performed the tasks you are given with great mastery, accuracy, and deep understanding of what is requested. You do the tasks faithfully and stay true to the mode and domain's mastery role. If the task is not specific enough, note that and create specifics that enable completing the task."""

def draw_tarot_cards():
    drawn_cards = random.sample(tarot_cards, 3)
    orientations = ["upright" if random.random() > 0.5 else "inverted" for _ in drawn_cards]
    return list(zip(drawn_cards, orientations))

def process_user_input(user_input):
    if "/tarot" in user_input:
        match = re.search(r"(.*) /tarot", user_input)
        if match:
            topic = match.group(1).strip()
            tarot_reading = draw_tarot_cards()
            return generate_tarot_reading_prompt(topic, tarot_reading)
    return user_input

def generate_tarot_reading_prompt(topic, tarot_reading):
    cards_description = ', '.join([f"{card} ({orientation})" for card, orientation in tarot_reading])
    prompt = f"You are giving a tarot reading about {topic}. Here are the cards drawn: {cards_description}. Please explain the meaning of each of the three cards in direct relation to my question about {topic}. Provide a detailed interpretation of how each card connects to {topic} and conclude with an overall decision on {topic}. Make sure your answer is specific and tailored to {topic}."
    return prompt

def cleanup_response(response_text):
    # Remove unwanted instructions and repeated segments
    response_parts = response_text.split("Seraphina:")
    if len(response_parts) > 1:
        response = response_parts[1].split("User:")[0].strip()
    else:
        response = response_text.strip()
    
    # Further clean-up if needed
    response = re.sub(r"###.*", "", response)
    return response

def generate_response(message, history):
    user_input = process_user_input(message)
    prompt = f"{seraphina_prompt}\nUser: {user_input}\nSeraphina:"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Create a TextStreamer instance
    text_streamer = TextStreamer(tokenizer)
    
    output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1280, temperature=1.5)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Raw Generated Response: {response_text}")  # Debug statement

    # Clean up the generated response
    response = cleanup_response(response_text)

    print(f"Final Generated Response: {response}")  # Debug statement
    return response

# Set up Flask API
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    message = data.get('message', '')
    history = data.get('history', [])
    response = generate_response(message, history)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)