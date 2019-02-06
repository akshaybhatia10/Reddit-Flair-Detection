import os
from flask import jsonify, Flask, render_template, request, redirect, url_for, send_from_directory
import six.moves.urllib as urllib
import numpy as np
import requests
import urllib.request, json 
import time
from flask_cors import CORS
import datetime

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

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

            if ".jpg" in image_src:
            	image_src = image_src.split('/')[-1].split('.')[0]
       	    
       	    date = datetime.datetime.fromtimestamp(date).date().strftime("%Y-%m-%d %H:%M")
            for comment in y:
                c = comment['data']['body']
                comments.append(c)

            features = {
            'title': title,
            'date': date,
            'author': author,
            'selftext': selftext,
            'flair': flair,
            'score': score,
            'num_comments': num_comments,
            'image_src': image_src,
            'comments': comments
            }

            print (features)
            return jsonify(features)
        except:
            return jsonify('Data is Corrupted')

    else:
        return jsonify('Invalid Input')


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')