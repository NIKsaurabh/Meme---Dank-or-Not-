#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:38:18 2021

@author: saurabh
"""
import streamlit as st
def main():
    st.title('Analysis of memes')
    st.subheader('Time of post of memes')
    st.image("/home/saurabh/Desktop/web_app/images/time.png")
    st.text("From the plot above we can see that the memes which are posted between 12am to 4am have")
    st.text("highest score values followed by 4am to 8am. Reason for this can be that most of the")
    st.text("users scrolls social networking sites just before sleeping for the case of 12am to 4am")
    st.text("and just after waking up for case of 4am to 8 am. One more reason can be, people need")
    st.text("humor just before sleeping and just after waking up.Memes have least score when posted")
    st.text("between 4pm to 8pm.")
    
    st.subheader('Sentiment of text of memes')
    st.image("/home/saurabh/Desktop/web_app/images/sentiment.png")
    st.text("Here we can see that,neutral texts have highest chances of becoming dank.")
    st.text("Between positive and negative sentiments, text with negative sentiment in the meme have")
    st.text("higher chances of becoming dank.")
    
    st.subheader('Number of words of memes')
    st.image("/home/saurabh/Desktop/web_app/images/num_words.png")
    st.text("Text with word length less than 12 is most preferable and have high chances of becoming")
    st.text("dank and it's obvious, peoplr don't like to read long texts for a pinch of humor.Texts")
    st.text("with word length greater than 11 is least preferable.")
    
    st.subheader('Word cloud of memes')
    st.image("/home/saurabh/Desktop/web_app/images/word_cloud.png")
    st.text("Some of the most frequent words in the memes are 'people','think','because','someone',")
    st.text("'something','still','really','friend','teacher' etc.")
    
    st.subheader('Colors of memes')
    st.image("/home/saurabh/Desktop/web_app/images/color.png")
    st.text("In most of the memes white, faded and dark colors are used. Light and bright colors are")
    st.text("rarely used in the memes.In general, muted colors are more abundant than bright colors in")
    st.text("memes. Perhaps because memes tend to be mundane photos, often blurry in self-made way,")
    st.text(" unlike professional photography.")
    
    st.subheader('Colors of memes')
    st.image("/home/saurabh/Desktop/web_app/images/hsv.png")
    st.text("From above plots we can see that, for a meme to get high score (upvotes minus downvotes) hue")
    st.text("should be low (approximately between 10 to 125 in opencv). For a meme to get high score the")
    st.text("saturation value should also be low (approximately between 0 to 140 in opencv).For a meme to")
    st.text("get high score the value should be higher (approximately between 100 to 200 in opencv).")
    
    st.subheader('Thumbnail heights and widths of memes')
    st.image("/home/saurabh/Desktop/web_app/images/thumbnail_h_w.png")
    st.text("Mostly thumbnail with height greater than 80 have high scores and almost all of the")
    st.text("thumbnails have 140 width means there is no correlation between score and thumbnail width.")
    
    
    
