#Web app

This folder contained files to deploy the bert model as a web app on GCE in a Flask environment

To run locally:

```bash
git clone https://github.com/akshaybhatia10/Reddit-Flair-Detection
cd Reddit-Flair-Detection/app
pip install -r requirements.txt
wget --no-check-certificate --no-proxy "https://s3.amazonaws.com/redditdata2/pytorch_model.bin" 
python app.py
```
