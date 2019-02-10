# Reddit-Flair-Detection

## Table of Contents

0. [About](#about)
0. [Installation](#installation)
0. [Data Aquisition](#data-aquisition)
0. [Flair Classification](#flair-classification)
0. [Deploying as a Web Service](#deploying-as-a-web-service)
0. [Build on Google Colab](#build-on-google-colab)
0. [References](#references)

## About

This repo illustrates the task of data aquisition of reddit posts from the [/india](https://www.reddit.com/r/india/) subreddit, classification of the posts into 11 different flairs and deploying the best model as a web service.
 
## Installation

NOTE: In case the installation does work as expected, move to [Build on Google Colab](#build-on-google-colab) to try the project without installing locally. All results can be replicated on google colab easily.

The following installation has been tested on MacOSX 10.13.6 and Ubuntu 16.04.

This project requires **Python 3** and the following Python libraries installed(plus a few other s depending on task):

- [sklearn](http://scikit-learn.com/)
- [Pytorch](http://pytorch.org/)
- [pandas](pandas.pydata.org/)
- [Numpy](http://numpy.org/)
- [Matplotlib](https://matplotlib.org/) 
- [Torchtext](https://torchtext.readthedocs.io/en/latest/data.html)

1. Clone the repo

```bash
git clone https://github.com/akshaybhatia10/Reddit-Flair-Detection/-.git
cd Reddit-Flair-Detection/
```

2. Run
```bash
pip install -r requirements
```

## Data Aquisition

**Note: The notebook requires a GCP account, a reddit account and CloudSDK installed. If you want to use the dataset to get started with running the models instead building the dataset yourself, download the datasets using:

To download the datasets from s3
```bash
wget --no-check-certificate --no-proxy "https://s3.amazonaws.com/redditdata2/train.json"
wget --no-check-certificate --no-proxy "https://s3.amazonaws.com/redditdata2/test.json"
```

We will reference the publically available Reddit dump to [here](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/). The dataset is publically available on Google BigQuery and is divided across months from December 2015 - October 2018. BigQuery allows us to perform low latency queries on massive datasets. One example is [this](https://bigquery.cloud.google.com/table/fh-bigquery:reddit_posts.2018_08). Unfortunately the posts have not been tagged with their comments. To extract this information, in addition to BigQuery, we will use [PRAW](https://praw.readthedocs.io/en/latest/) for this task. 

The idea is to randomly query a subset of posts from December 2015 - October 2018. Then for each of the post, use praw to get comments for each one. To build a balanced dataset, we will limit the number of samples for each flair at 2000 and further randomly sample from the queried records. 

To get started, follow [here](https://github.com/akshaybhatia10/Reddit-Flair-Detection/blob/master/notebooks/data_aquisition.ipynb). (**Note: The notebook requires a GCP account, a reddit account and CloudSDK installed.)**


| Feature Name       | Type            | Description                           |
| ---                | ---             | ---                                   | 
| author             | STR             | author name                           |
| comments           | LIST            | list of top comments(LIMIT 10)        |
| created_utc        | INT             | timestamp of post                     |
| link_flair_text    | STR             | flair of the post                     |
| num_comments       | INT             | number of comments on the post        |
| score              | INT             | score of the post (upvotes-downvotes) |
| over_18            | BOOL            | whether post is age restricted or not |
| selftext           | STR             | description of the post               |
| title              | STR             | title of the post                     |
| url                | STR             | url associated with the post          |

This stores the queried records in a mongoDB database 'dataset' within the collection 'reddit_data'. To export the mongoDB colection to json, run:

```bash
mongoexport --db dataset -c reddit_dataset --out ./reddit_data.json
```

To import this json to your system, run:

```bash
mongoimport --db dataset --collection reddit_dataset --file ./reddit_data.json
```


| Description | Size  | Samples  |
| --- | --- | --- |  
|dataset/train | 31MB | ~16400 |
|dataset/test  | 7MB | ~5000   |

We are considering 11 flairs. The number of samples per set is:

| Label | Flair              | Train Samples  | Test Samples  |
| ---   | ---                | ---            | ---           | 
| 1.    | AskIndia           | 1523            | 477          |
| 2.    | Politics           | 1587            | 413          |
| 3.    | Sports             | 1719            | 281          |
| 4.    | Food               | 1656            | 344          |
| 5.    | [R]eddiquette      | 1604            | 396          |
| 6.    | Non-Political      | 1586            | 414          |
| 7.    | Scheduled          | 1596            | 372          |
| 8.    | Business/Finance   | 1604            | 396          |
| 9.    | Science/Technology | 1626            | 374          |
| 10.   | Photography        | 554             | 86           |
| 11.   | Policy/Economy     | 1431            | 569          |


## Flair Classification

**Note: The notebooks download the training and test set automatically.**

This section describes different models implemeted for the task of flair classification. We ideally want to classify the post as soon as it is created so we mostly use the title and body of the post as inputs to the models. 

#### 1. [Data Exploration and Baseline Implementations](https://github.com/akshaybhatia10/Reddit-Flair-Detection/blob/master/notebooks/Data_Analysis_and_Baseline_Models.ipynb)

In this example, we perform a basic exploration of all features. We then run a simple XGBoost model over some meta features. This is followed by running 3 simple baseline algorithms using TFIDF on title and post as features.

#### 2. [Simple GRU/LSTM, GRU with Concat Pooling and GRU/LSTM with Self Attention](https://github.com/akshaybhatia10/Reddit-Flair-Detection/blob/master/notebooks/RNN_LSTM.ipynb)

In this section, we implement various RNN based architectures on the reddit post title concatenated with the post body feature. Also, each model uses the pretrained glove embeddings as inputs to the model.(without fine tuning)
    
###### a) Simple GRU/LSTM

This model consists of a single layer vanilla GRU with a softmax classifier.

<p align="center">
  <img width="460" height="400" src="https://cdn-images-1.medium.com/max/1600/0*7CvKTm5BHkjV_jrt.png">
</p>

###### b) GRU with Concat Pooling

Here we implement concat pooling with a GRU. (See notebook for more details.)

<p align="center">
  <img width="460" height="400" src="https://cdn-images-1.medium.com/max/1400/1*qJggHpIPUkkzG0KQZ-EUcQ.jpeg">
</p>

###### c) GRU/LSTM with Self Attention

This model uses a RNN encoder conditioned with a scaled dot product self attention layer with a softmax classifier.

<p align="center">
  <img width="460" height="400" src="https://camo.githubusercontent.com/c2fb353e0d05d634ea93e0cb0f6dd2ccd226af04/687474703a2f2f7777772e77696c646d6c2e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031352f31322f53637265656e2d53686f742d323031352d31322d33302d61742d312e31362e30382d504d2e706e67">
</p>

#### 3. [Classification using BERT](https://github.com/akshaybhatia10/Reddit-Flair-Detection/blob/master/notebooks/BERT.ipynb)

NOTE: Most of the code in this notebook is referenced from the pytorch implementation of BERT in [this](https://github.com/huggingface/pytorch-pretrained-BERT) repo.

In this notebook, we use the pretrained language model BERT for the flair classification. BERT fine-tuning requires only a few new parameters added. For our purpose, we get the prediction by taking the final hidden state of the special first token [CLS], and multiplying it with a small weight matrix, and then applying softmax .Specifically, we use the uncased 12 head, 768 hidden model.

<p align="center">
  <img width="460" height="400" src="https://gluon-nlp.mxnet.io/_images/bert-sentence-pair.png">
</p>

The BERT model gives and best performance and is used in the deployed web app. Download the trained model using:

```bash
wget --no-check-certificate --no-proxy "https://s3.amazonaws.com/redditdata2/pytorch_model.bin"
```

### Results

The results of different models on test set:

| Model | Accuracy             |       |
| ---   | ---                  | ---   |  
| Logistic Regression          | 55.45 |
| MultinomialNB                | 54.70 |
| SGD                          | 56.30 |
| Single Layer Simple GRU/LSTM | 59.36 |
| GRU with Concat Pooling      | 61.62 |
| Bi-GRU with Self Attention    | 54.36 |
| BERT                         | 67.1  |

## Deploying as a Web Service

The best model - BERT is deployed as a web app. Check the live demo [here](http://104.196.19.204/). Due to the large model size, the app was deployed using Google Compute Engine platform rather a free service like heroku(due to its limited slug size). All the required files can be found [here](https://github.com/akshaybhatia10/Reddit-Flair-Detection/tree/master/app)

### Webapp demo

 1                         |                      2    |
:-------------------------:|:-------------------------:|
![](app/1.png?raw=True) |![](app/2.png?raw=true) |

## Build on Google Colab

Google Colab lets us build the project without installing it locally. Installation of some libraries may take some time depending on your internet connection.

To get started, open the notebooks in playground mode and run the cells(You must be logged in with your google account and provide additional authorization). Also since mongoDB cannot be run in a Colab environment, the data aquisition notebook cannot run in Google Colab.

1. [Data Exploration and Baseline Implementations](https://colab.research.google.com/drive/1N4nZozJg7SO_qLZ7-kpkp2JIl5IsSDqf)
2. [Simple GRU/LSTM, GRU with Concat Pooling and GRU/LSTM with Self Attention](https://colab.research.google.com/drive/1gLwrx5a1j_QdnmRBUnd5JiEnFh2MndBp)
3. [Classification using BERT](https://colab.research.google.com/drive/1msACJKPhXDdNsbAS2FhRIIlKOtyGdPKf)

## References 

- [Cho et. al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. 2014](https://arxiv.org/pdf/1406.1078.pdf)
- [Devlin et. al. BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding. 2018 ](https://arxiv.org/pdf/1810.04805)
- [Jeremy Howard, Sebastian Ruder. ULMFIT. 2018](https://arxiv.org/pdf/1801.06146.pdf)
- [ Vasvani et. al. Attention is all you need. Nips 2017](https://arxiv.org/pdf/1706.03762)
- Pytorch Implementation of BERT - [HuggingFace Github repo](https://github.com/huggingface/pytorch-pretrained-BERT)
