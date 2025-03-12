from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the fine-tuned model, tokenizer, and label mapping
model_path = "model/sentiment_model.pth"
tokenizer_path = "model/tokenizer"
label_mapping_path = "model/label_mapping.pkl"

model = AutoModelForSequenceClassification.from_pretrained(tokenizer_path, num_labels=4)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load label mapping
with open(label_mapping_path, "rb") as f:
    label_mapping = pickle.load(f)

reverse_mapping = {v: k for k, v in label_mapping.items()}

# Sentiment prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    sentiment = reverse_mapping[prediction]
    return sentiment

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
