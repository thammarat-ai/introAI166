#Example 6.19 Face Detection in webcam - web app with Streamlit
#pip install --upgrade protobuf
#pip install streamlit

import streamlit as st
import cv2 
import numpy as np

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


@st.cache(allow_output_mutation=True)
def get_cap():
    #from webcam
    return cv2.VideoCapture(0)
    #from a video file
    # return cv2.VideoCapture('Roller Coaster.mp4')

cap = get_cap()

frameST = st.empty()
st.subheader("Webcam Beauty App")
st.sidebar.markdown("# Brightness Contrast ")
brightness = st.sidebar.slider("Brightness", 0, 255, 0, 1)
contrast = st.sidebar.slider("Contrast", 0, 255, 0, 1)
smothness = st.sidebar.slider("Smothness", 0.0, 5.0, 1.0, 0.1)

st.sidebar.markdown("# Alpha * frame + Beta")
alpha = st.sidebar.slider("Alpha", 0.0, 5.0, 2.0, 0.1)
beta = st.sidebar.slider("Beta", 0, 100, 10, 1)

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0,
gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    
    return buf

while True:
    ret, frame = cap.read()
    # Stop the program if reached end of video
    if not ret:
        cv2.waitKey(3000)
        cap.release()
        break

    #Alpha and Beta =================================================
    frame2 = np.uint8(np.clip((alpha * frame + beta), 0, 255))

    #Ajust the Smoothness of the Image===============================
    level = int(smothness*10)
    frame2 = cv2.bilateralFilter(frame2, level, 75, 75)

    #Adjust the brightness and contrast ==============================
    frame2 = apply_brightness_contrast(frame2, brightness , contrast )

    #concatanate image Vertically ===================================
    result=np.concatenate((frame,frame2),axis=0)
    
    frameST.image(result, channels="BGR")


