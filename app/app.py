import os
from flask import jsonify, Flask, render_template, request, redirect, url_for, send_from_directory
import six.moves.urllib as urllib
import numpy as np
import requests
import urllib.request, json 
import time
from pytorch_pretrained_bert.tokenization import BertTokenizer
from flask_cors import CORS
import datetime
import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from helper import bert_predict

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

label_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
max_seq_length = 256
id_to_label = {0: 'AskIndia',
 1: 'Business/Finance',
 2: 'Food',
 3: 'Non-Political',
 4: 'Photography',
 5: 'Policy/Economy',
 6: 'Politics',
 7: 'Scheduled',
 8: 'Science/Technology',
 9: 'Sports',
 10: '[R]eddiquette'}

bert_model = 'bert-base-uncased'
model_file = 'pytorch_model.bin'

tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
model_state_dict = torch.load(model_file, map_location='cpu')
model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=model_state_dict, num_labels=11)

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')

@app.route('/render', methods=['POST'])
def render():
    check = ['https']
    url = request.data.decode('utf-8')
    if url:
        try:
            r = requests.get(url + '.json', headers = {'User-agent': 'your bot 0.1'})
            data = r.json()
            comments = list()

            x = data[0]['data']['children'][0]['data']
            y = data[1]['data']['children']

            author = x['author']
            date = x['created']
            title = x['title']
            selftext = x['selftext']
            flair = x["link_flair_text"]
            score = x["score"]
            num_comments = x["num_comments"]
            image_src = x['url']
            print (image_src)
            if ".jpg" or ".png" in image_src:
            	image_src = image_src.split('/')[-1].split('.')
       	    
       	    print (image_src)
       	    date = datetime.datetime.fromtimestamp(date).date().strftime("%Y-%m-%d %H:%M")
            for comment in y:
                c = comment['data']['body']
                comments.append(c)

            label = bert_predict(model, title, selftext, tokenizer, id_to_label, label_list, max_seq_length)

            features = {
            'title': title,
            'date': date,
            'author': author,
            'selftext': selftext,
            'flair': flair,
            'score': score,
            'num_comments': num_comments,
            'image_src': image_src,
            'comments': comments,
            'label': label
            }

            return jsonify(features)
        except:
            return jsonify('Invalid Input')

    else:
        return jsonify('Invalid Input')


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=80)