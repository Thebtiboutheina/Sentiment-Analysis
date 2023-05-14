import pandas as pd
from flask import request, jsonify, Flask, render_template
from transformers import pipeline

app = Flask(__name__, template_folder='templates')

# Load the sentiment-analysis pipeline using the RoBERTa model
sentiment_analysis_pipeline = pipeline('sentiment-analysis', model='roberta-base')

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def my_form_post():
    text = request.form['text']

    # Use the RoBERTa model for sentiment analysis
    sentiment = sentiment_analysis_pipeline(text)[0]
    label = sentiment['label']
    score = sentiment['score']

    # Map the labels to human-readable labels
    if label == 'POSITIVE':
        label = 'This sentence is positive'
    elif label == 'NEGATIVE':
        label = 'This sentence is negative'
    else:
        label = 'This sentence is neutral'

    # Round the score and multiply it by 100 to get the percentage
    score_percentage = round(score * 100)

    return jsonify({"label": label, "score": score_percentage})

if __name__ == "__main__":
    app.run(port='8088', threaded=False)
