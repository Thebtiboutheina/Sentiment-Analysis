import pandas as pd
import nltk
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from tqdm import tqdm
from flask import request
from flask import jsonify
from flask import Flask, render_template


@app.route('/')
def my_form():
    return render_template('index.html')
app = Flask(__name__,template_folder='template')

def roberta_model ():
   task='sentiment'
   MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
   tokenizer = AutoTokenizer.from_pretrained(MODEL)
   model = AutoModelForSequenceClassification.from_pretrained(MODEL)
   model.save_pretrained(MODEL)
   return model

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    model=roberta_model()
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict
def roberta_class(text):
     roberta_result = polarity_scores_roberta(text)
     if roberta_result['roberta_pos'] > roberta_result['roberta_neg'] and roberta_result['roberta_pos'] > roberta_result['roberta_neu']:
        pred_class = 'positive'
     elif roberta_result['roberta_neg'] > roberta_result['roberta_pos'] and roberta_result['roberta_neg'] > roberta_result['roberta_neu']:
        pred_class = 'negative'
     else:
        pred_class = 'neutral'
     return pred_class, roberta_result


def my_form_post():
    text = request.form['text']
    class = roberta_class (text)
    return(render_template('index.html', variable=class))
if __name__ == "__main__":
    app.run(port='8088',threaded=False)
