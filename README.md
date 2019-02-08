# Reddit-Flair-Detection

This repo illustrates the task of data aquisition of reddit posts from the [/india](https://www.reddit.com/r/india/) subreddit, classification of the posts into 11 different flairs and deploying the best model as a web service.

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

NOTE: In case the installation does work as expected, move to [Build on Google Colab](#colab) to try the project without installing locally. All results can be replicated on google colab easily.

The following installation has been tested on MacOSX 10.13.6 and Ubuntu 16.04. (This project requires Python 3.6 and Conda which is an open source package management system and environment management system that runs on Windows, macOS and Linux. Before starting, make sure both are installed or follow the instructions below.)

This project requires **Python 3** and the following Python libraries installed:

- [sklearn](http://scikit-learn.com/)
- [Pytorch](http://pytorch.org/)
- [pandas](pandas.pydata.org/)
- [Numpy](http://numpy.org/)
- [Scipy](http://scipy.org/)
- [Matplotlib](https://matplotlib.org/) 

1. Clone the repo

```bash
git clone https://github.com/akshaybhatia10/Reddit-Flair-Detection/-.git
cd Reddit-Flair-Detection/
```

2. 
```bash
pip install -r requirements
```

## Data Aquisition

**Note: The notebook requires a GCP account, a reddit account and CloudSDK installed. If you want to use the dataset to get started with running the models, download the datasets using:

```bash
wget 
wget 
```

We will reference the publically available Reddit dump to [here](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/). The dataset is publically available on Google BigQuery and is divided across months from December 2015 - October 2018. BigQuery allows us to perform low latency queries on massive datasets. One example is [this](https://bigquery.cloud.google.com/table/fh-bigquery:reddit_posts.2018_08). Unfortunately the posts have not been tagged with their comments. To extract this information, in addition to BigQuery, we will use [PRAW](https://praw.readthedocs.io/en/latest/) for this task. 

The idea is to randomly query a subset of posts from December 2015 - October 2018. Then for each of the post, use praw to get comments for each one. To build a balanced dataset, we will limit the number of samples for each flair at 2000 and randomly sample from the extracted dataset. 

To get started, follow [here](). (**Note: The notebook requires a GCP account, a reddit account and CloudSDK installed.)


| Description | Size  | Samples  |
| --- | --- | --- |  
|dataset/train | 31MB | ~16400 |
|dataset/test  | 7MB | ~5000   |

We are considering 11 flairs. The number of samples per set is:

| Label | Flair              | Train Samples  | Test Samples |
| ---   | ---                | ---            | ---          | 
| 1.    | AskIndia           | 1523            | 477          |
| 2.    | Politics           | 1587            | 413          |
| 3.    | Sports             | 1719            | 281          |
| 4.    | Food | 1656        | 344             | 1312         |
| 5.    | [R]eddiquette      | 1604            | 396          |
| 6.    | Non-Political      | 1586            | 414          |
| 7.    | Scheduled          | 1596            | 372          |
| 8.    | Business/Finance   | 1604            | 396          |
| 9.    | Science/Technology | 1626            | 374          |
| 10.   | Photography        | 554             | 86           |
| 11.   | Policy/Economy     | 1431            | 569          |


## Flair Classification

#### 1. Data Exploration and Baseline Implementations

In this example, we perform a basic exploration of all features. We then run a simple XGBoost model over some meta features. This is followed by running 3 simple baseline algorithms using TFIDF as features.

#### 2. Simple GRU/LSTM, GRU with Concat Pooling and GRU/LSTM with Self Attention

Here we use pretrained glove embeddings for all the models (without fine tuning)
    
###### a) Simple GRU/LSTM
![GRU Cell](https://cdn-images-1.medium.com/max/1600/0*7CvKTm5BHkjV_jrt.png)

###### b) GRU with Concat Pooling
![Concat Pooling](https://cdn-images-1.medium.com/max/1400/1*qJggHpIPUkkzG0KQZ-EUcQ.jpeg)

###### c) GRU/LSTM with Self Attention
[Self Attention](https://camo.githubusercontent.com/c2fb353e0d05d634ea93e0cb0f6dd2ccd226af04/687474703a2f2f7777772e77696c646d6c2e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031352f31322f53637265656e2d53686f742d323031352d31322d33302d61742d312e31362e30382d504d2e706e67)

#### 3. Classification using BERT

In this notebook, we use the pretrained language model BERT for the flair classification. BERT fine-tuning requires only a few new parameters added. For our purpose, we get the prediction by taking the final hidden state of the special first token [CLS], and multiplying it with a small weight matrix, and then applying softmax .Specifically, we use the uncased 12 head, 768 hidden model.

### Results

| Model | Accuracy             |       |
| ---   | ---                  | ---  |  
| Logistic Regression          |      |
| MultinomialNB                |      |
| SGD                          |      |
| Single Layer Simple GRU/LSTM |      |
| GRU with Concat Pooling      |      |
| GRU/LSTM with Self Attention |      |
| BERT                         | 67.5 |

## Deploying as a Web Service

The best model BERT is deployed as a web app. Check the live demo [here](https://reddit-flair.herokuapp.com)

## Build on Google Colab

Google Colab lets us build the project without installing it locally. Installation of some libraries may take some time depending on your internet connection.

To get started, open the notebooks in playground mode and run the cells(You must be logged in with your google account and provide additional authorization). Also since mongoDB cannot be run in a Colab environment, the data aquisition notebook cannot run in Google Colab.

1. [Data Exploration and Baseline Implementations]()
2. [Simple GRU/LSTM, GRU with Concat Pooling and GRU/LSTM with Self Attention]()
3. [Classification using BERT]()

## References 

- Jeremy Howard, Sebastian Ruder. [ULMFIT](https://arxiv.org/pdf/1801.06146.pdf)
