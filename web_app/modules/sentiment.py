#!/usr/bin/env python3
#function to compute sentiment score

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords


from tensorflow import keras
import json
from nltk.tokenize import RegexpTokenizer

import warnings
warnings.filterwarnings("ignore")
#function to compute sentiment score
def get_sentiment(prd_model, text_data, word_idx):

    live_list = []
    batchSize = len(text_data)
    live_list_np = np.zeros((56,batchSize))

    mask = [isinstance(item, (str, bytes)) for item in text_data['text']]
    text_data = text_data.loc[mask]

    for index, row in text_data.iterrows():
        

        text_data_sample = text_data['text'][index]
        
        # split the sentence into its words and remove any punctuations.
        tokenizer = RegexpTokenizer(r'\w+')
        text_data_list = tokenizer.tokenize(text_data_sample)
        if len(text_data_list)>56:
          text_data_list = text_data_list[:56]

        labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
        
        # get index for the live stage
        data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in text_data_list])
        data_index_np = np.array(data_index)

        # padded with zeros of length 56 i.e maximum length
        padded_array = np.zeros(56)
        padded_array[:data_index_np.shape[0]] = data_index_np
        data_index_np_pad = padded_array.astype(int)


        live_list.append(data_index_np_pad)

    live_list_np = np.asarray(live_list)
    score = prd_model.predict(live_list_np, batch_size=64, verbose=0)
    single_score = np.round(np.dot(score, labels)/10,decimals=2)
    score_all  = []
    for each_score in score:

        top_3_index = np.argsort(each_score)[-3:]
        top_3_scores = each_score[top_3_index]
        top_3_weights = top_3_scores/np.sum(top_3_scores)
        single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)
        score_all.append(single_score_dot)

    text_data['Sentiment_Score'] = score_all
    
    return text_data

    
