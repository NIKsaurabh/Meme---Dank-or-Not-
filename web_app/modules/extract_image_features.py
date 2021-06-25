#!/usr/bin/env python3
import numpy as np
import cv2 
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

class image_features:
    def __init__(self, image):
        self.img_feature_model = load_model("/home/saurabh/Desktop/web_app/models/img_feature_vgg.h5")#VGG16()
        self.image = image

    def hsv(self):
        #extracting HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        avg_h = h.mean()
        avg_s = s.mean()
        avg_v = v.mean()

        return [avg_h, avg_s, avg_v]

    def color(self):
        #range of HSV values to extract colors from the image
        self.boundries = [([0,0,200],[180,25,255]),      #white
                    ([0,0,0],[180,255,3]),        #black
                    ([0,0,100],[180,20,180]),        #gray
                    ([0,90,115],[17,255,190]),     #brown
                    ([20,50,240],[30,75,255]),     #off-white
                    ([0,140,155],[12,255,230]),      #dark red
                    ([0,140,230],[12,255,255]),     #light red
                    ([13,190,155],[17,255,230]),     #dark orange
                    ([13,140,230],[115,255,255]),    #light orange
                    ([18,140,155],[140,255,230]),     #goldish
                    ([23,140,230],[165,255,255]),    #yellow
                    ([28,90,155],[80,255,230]),    #dark green
                    ([85,77,153],[93,255,230]),   #dark cyan
                    ([85,77,230],[93,255,255]),  #cyan
                    ([100,128,90],[125,255,190]),   #dark blue
                    ([100,128,193],[125,255,255]),  #light blue
                    ([0,0,255],[180,25,255])      #faded colors
        ]
        num_pixel = self.image.shape[0] * self.image.shape[1]
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        clr_pixel = []

        for (lower, upper) in self.boundries:
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
            mask = cv2.inRange(hsv, lower, upper)
            clr_pixel.append(round(((mask==255).sum())/num_pixel,5)) #counting and normalizing number of pixels

        return clr_pixel

    def dimension(self):
        height = self.image.shape[0]
        width = self.image.shape[1]
        return [height, width]

    def objects(self):
        #extracting objects from image using VGG16
        
        pixels = np.asarray(self.image)
        pixels = pixels.astype('float32')
        pixels.resize(224,224,3)
        pixels = np.expand_dims(pixels, axis=0)
        pixels = preprocess_input(pixels)
        im_prediction = self.img_feature_model.predict(pixels)
        labels = decode_predictions(im_prediction, top=3)
        return labels
