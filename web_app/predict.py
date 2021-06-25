#importing libraries
import pandas as pd
import numpy as np
from modules import sentiment
from modules import extract_text
from modules.extract_image_features import image_features
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import os
import json
import pickle
from skimage import io
from PIL import Image
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


#model to get sentiment of text
sentiment_model = load_model('/home/saurabh/Desktop/web_app/models/best_model.hdf5')
word_idx = json.load(open("/home/saurabh/Desktop/web_app/models/word_idx.txt"))

#function for data preparation
def data_preparation(image, time, caption):
    st.text('Preparing data...')
    im = Image.open(image)
    im = np.array(im)
    
    df = pd.DataFrame()
    
    df['hour'] = [time]
    #extracting text
    df['text'] = [extract_text.extract_and_clean(im) + caption]
    
    #feature containing number of words in text
    df['num_words']   = [df['text'].str.split().apply(len)]
    
    #getting sentiment score of the text data
    df = sentiment.get_sentiment(sentiment_model, df, word_idx)
    
    img_feature = image_features(im)
    
    #extracting HSV
    h_s_v = img_feature.hsv()
    df['avg_h'] = [h_s_v[0]]
    df['avg_s'] = [h_s_v[1]]
    df['avg_v'] = [h_s_v[2]]
    
    #extracting colors from image
    clr_pixel = img_feature.color()
    all_colors = clr_pixel
    
    #colors
    colors = ['white','black','gray','brown','off-white','dark red','light red','dark orange','light orange',
              'goldish','yellow','dark green','dark cyan','cyan','dark blue','light blue','faded colors']
    #adding new color features to the dataset
    for i, color in enumerate(colors):
        df[color] = [all_colors[i]]
    
    
    #getting height and width of the image
    ht_wt = img_feature.dimension()
    df['thumbnail_height'] = [ht_wt[0]]
    df['thumbnail_width']  = [ht_wt[1]]
    
    #extracting objects from image using VGG16
    labels = img_feature.objects()
    
    df['img_feature_pred_1'] = [labels[0][0][2]]
    df['img_feature_pred_2'] = [labels[0][1][2]]
    df['img_feature_pred_3'] = [labels[0][2][2]]
    
    website    =0
    book_jacket=0
    packet     =0
    mud_turtle =0

    objects = [labels[0][0][1],labels[0][1][1],labels[0][2][1]]
    for j in ['website','book_jacket','packet','mud_turtle']:
        if j in objects:
          exec("%s = %d" % (j,1))
        else:
          exec("%s = %d" % (j,0))
    df['web_site']     = [website]
    df['book_jacket'] = [book_jacket]
    df['packet']      = [packet]
    df['mud_turtle']  = [mud_turtle]
    print(df)
    
    #dataset for ML model
    X_ml = df[['img_feature_pred_1','img_feature_pred_2','img_feature_pred_3','avg_h','avg_s','avg_v','Sentiment_Score','hour',
               'num_words','thumbnail_height','thumbnail_width','gray', 'white', 'faded colors', 'black', 'dark blue', 'goldish',
               'light blue', 'brown', 'yellow', 'dark cyan', 'light orange', 'dark green', 'cyan', 'off-white', 'dark red', 
               'dark orange', 'light red','web_site','book_jacket','packet','mud_turtle']]
    
    #dataset for NLP model
    X_bert = df['text']
    
    return [X_ml, X_bert]

#function to load models
def load_models():
    st.text("Loading models...")
    ml_model = pickle.load(open("/home/saurabh/Desktop/web_app/models/dankornot_ml.pkl", 'rb'))
    cnn_model = load_model("/home/saurabh/Desktop/web_app/models/resnet_model.h5")
    nlp_model = tf.saved_model.load('/home/saurabh/Desktop/web_app/models/bert_model')
    st.text("Models loaded.")
    return [ml_model, cnn_model, nlp_model]

#function to predict
def prediction(X_ml, image, X_bert):
    models = load_models()
    #predicting using ML model
    ml_pred_prob = models[0].predict_proba(X_ml)[:,-1][0]
    
    #predicting using CNN model
    if image is not None:
        img = Image.open(image)
        pixels = np.asarray(img)
        pixels = pixels.astype('float32')
        pixels /= 255.0
        pixels.resize(224,224,3)
        pixels = np.expand_dims(pixels, axis=0)
        cnn_prediction = models[1].predict(pixels)
        cnn_pred_prob = cnn_prediction[0][0]
        
    #predicting using NLP model
    for text in X_bert:
        try:
          bert_predict = tf.sigmoid(models[2](tf.constant([text])))
          nlp_pred_prob = np.array(bert_predict)[0][0]
        except:
          nlp_pred_prob = 0
          
    st.text('Done...here is the prediction')
          
    return [ml_pred_prob, cnn_pred_prob, nlp_pred_prob]

  
def main():
    st.title("Is your Meme dank? Let's predict!")

    uploaded_image = st.file_uploader("Choose an image", type = ['png','jpg','jpeg'])
    time = st.number_input('When you want to post your meme?',0,23, value=1)
    caption = st.text_input('Enter caption for meme, if you have','Enter caption')
    if caption == 'Enter caption':
        caption=" "
        
    if st.button('Predict'):
        img = Image.open(uploaded_image)
        st.image(img, caption='Uploaded image')
        data = data_preparation(uploaded_image, time, caption) 
        st.text('Data preparation done.')
        X_ml = data[0]
        image = uploaded_image
        X_bert = data[1]
        st.text('Predicting...')
        predict = prediction(X_ml, image, X_bert)
        
        dank_percentage = (np.array(predict).mean()*100).round(2)
        
        if dank_percentage>=80:
            st.success('Hooray!!! Chance of your meme being dank is '+str(dank_percentage)+'%. '+'Just post it!')
        elif (dank_percentage>=70) & (dank_percentage<80):                     
            st.success('WoW!!! Chance of your meme being dank is : '+str(dank_percentage)+'%')
        elif (dank_percentage>=60) & (dank_percentage<70):
            st.success('Not bad!!! Chance of your meme being dank is : '+str(dank_percentage)+'%')
        elif (dank_percentage>=50) & (dank_percentage<60):
            st.success('Ummm!!! Chance of your meme being dank is : '+str(dank_percentage)+'%')
        else:
            st.success('Chance of your meme being dank is : '+str(dank_percentage)+'%. Needs some improvement.')
            
            
            

if __name__ ==  '__main__':
    main()
    


    
