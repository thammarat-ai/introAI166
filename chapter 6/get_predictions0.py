#Example 6.15b
#Modified from:
#https://github.com/akash18tripathi/TCS-HumAIn-2019-Age-Emotions-Ethnicity-Gender-Predictions-Using-Computer-Vision/blob/master/get_predictions.py

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as img
import cv2
import matplotlib.pyplot as plt

age_dict={
    0:'below_20',
    1:'20-30',
    2:'30-40',
    3:'40-50',
    4:'above_50'
}

gender_dict = {
    0:'Male',
    1:'Female'
}
ethnicity_dict={
    0:'arab',
    1:'asian',
    2:'black',
    3:'hispanic',
    4:'indian',
    5:'white'
}
emotion_dict={
    0:'angry',
    1:'happy',
    2:'neutral',
    3:'sad'
}

def predict(file):
    
    #image = img.load_img(file)
    #image = np.array(image,dtype='uint8')
    image = cv2.imread(file)
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,10)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        roi_color = image[y:y+h,x:x+w]
        roi_color = cv2.resize(roi_color, (70,70), interpolation = cv2.INTER_AREA)
        face = np.array(roi_color,dtype='uint8')
        face = face/255
        face = face.reshape(1,70,70,3)
        #Predicting
        g_pred = gender_pred(face)
        e_pred = ethnicity_pred(face)
        a_pred = age_pred(face)
        emo_pred = emotions_pred(face)
        cv2.putText(image,"Age:"+str(a_pred),(x,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(image,"Gender:  "+str(g_pred),(x,y+h+40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(image,"Emotion: "+str(emo_pred),(x,y+h+60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(image,"Ethnicity: "+str(e_pred),(x,y+h+80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
    cv2.imwrite('output.jpg',image)
    cv2.imshow('output',image)

def gender_pred(image):
        gen = gender_model.predict(image).argmax()
        return gender_dict[gen]


def emotions_pred(image):
        emotion = emotions_model.predict(image).argmax()
        return emotion_dict[emotion]


def ethnicity_pred(image):
        eth = ethnicity_model.predict(image).argmax()
        return ethnicity_dict[eth]

def age_pred(image):
        age = age_model.predict(image).argmax()
        return age_dict[age]

ethnicity_model = load_model('ethnicity_model.h5')
gender_model = load_model('gender_model.h5')
age_model = load_model('age_model.h5')
emotions_model = load_model('emotions_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

file = 'image2.jpeg'
predict(file)    
