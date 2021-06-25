#!/usr/bin/env python3
#function to extract and clean text
#installing pytesseract

import cv2 
import re
import nltk
from nltk.corpus import stopwords
import tensorflow_text as text
import pytesseract

nltk.download('punkt')
nltk.download('stopwords')

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def extract_and_clean(image):
    #extracting text
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) 
    custom_config = r"--oem 3 --psm 11 -c tessedit_char_whitelist= 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '"
    text = pytesseract.image_to_string(image, lang='eng', config=custom_config)
    text = text.replace('\n', ' ')

    #cleaning text
    text = re.sub('[^A-Za-z]',' ',text).lower()
    words = nltk.word_tokenize(text)
    stopWords = set(stopwords.words('english'))
    words = [w for w in words if w not in stopWords and len(w)>3]
    text = ' '.join(words)
    return text