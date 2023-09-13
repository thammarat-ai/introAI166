#Example 6.1 OpenCV face detection
import cv2

img = cv2.imread('face0.jpeg')
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces= cascade_classifier.detectMultiScale(gray, minNeighbors=5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.putText(img,'face', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        
cv2.imshow('face', img)
cv2.waitKey(0)
