#Example 6.x Face Recognition System OpenCV
#pip install opencv-contrib-python
import cv2
import numpy as np
import PySimpleGUI as sg
import sys
import os
from PIL import Image
import pandas as pd
import csv

dataPath = "data"
databaseFile = "database.txt"

register = False
recognizeFrame = False
sampleNum = 20
name = ""
Id = ""
course = ""

faces=[]
Id =[]
# Local Binary Pattern Histogram is an Face Recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer = cv2.face.FisherFaceRecognizer_create()

# creating detector for faces 
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

def createImages(frame,count):		 
    global dataPath,name,Id,detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = detector.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(dataPath+"\\"+name +"."+Id +'.'+ str(count) + ".jpg", gray[y:y + h, x:x + w])
    return frame
def writeDatabase(databaseFile,row):
    #Create a student record file if does not exist
    if not os.path.exists(databaseFile):
        with open(databaseFile, 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Id","Name","Course"])
        csvFile.close()
    with open(databaseFile, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()    
    return "Saved to database"
def getImagesAndLabels(path): 
    imagePaths =[os.path.join(path, f) for f in os.listdir(path)] 
    faces =[] 
    Ids =[] 
    for imagePath in imagePaths:       
        extension = os.path.splitext(imagePath)[1]
        if extension != '.jpg':
            print(extension)
            continue
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def Train(path): 
    global recognizer,detector
    faces, Id = getImagesAndLabels(path) 
    recognizer.train(faces, np.array(Id))	 
    recognizer.save("Trainner.yml") 
    return "Training finished..."

# For testing phase 
def TrackImages(frame): 
    global recognizer,detector
    recognizer.read("Trainner.yml")
    df = pd.read_csv(databaseFile,delimiter=',')
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5)
    person = ""
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        #print("ID:" + str(Id) + " conf:" + str(conf))
        #print(df['Id'] )

	#If confidence is less than 100 ==>"0" perfect match
        if(conf < 50):
            person = df.loc[df['Id'] == Id]['Name'].values
            cr = df.loc[df['Id'] == Id]['Course'].values
            print(person)
            person = str(Id)+"-"+person+"-"+cr
        else:
            Id ='Unknown'
            person = str(Id)
        person = (str(person)+" " +str(int(conf))+"%")
        #cv2.putText(frame, person, (x, y + h), 
	#            font, 1, (255, 255, 255), 2)
    return frame,person
count = 0			
def main():
    global register,sampleNum,dataPath,name,Id,course,recognizeFrame
    sg.ChangeLookAndFeel('LightGreen')

    # define the window layout
    leftpanel = [
                 [sg.Text('Face Recognition', size=(30, 1), justification='center', font='Helvetica 20',key='title')],
                 [sg.Image(filename='', key='image')],
                ]
    rightpanel =[[sg.Text('Id:', size =(10, 1))],
                 [sg.InputText("", key="Id")],
                 [sg.Text('Name:', size =(10, 1))],
                 [sg.InputText("", key="name")],
                 [sg.Text('Course:', size =(10, 1))],
                 [sg.InputText("", key="course")],
                 [sg.Button('1.Register', size=(15, 1), font='Helvetica 14')],
                 [sg.Button('2.Train', size=(15, 1), font='Any 14')],
                 [sg.Button('3.Recognize', size=(15, 1), font='Helvetica 14')],
                ]
    layout = [
        [   sg.Column(leftpanel),
            sg.VSeperator(),
            sg.Column(rightpanel),
        ]
    ]
    window = sg.Window('Face Recognition System',location=(100, 100))
    window.Layout(layout).Finalize()

        
    info =""
    cap = cv2.VideoCapture(0)
    while True:
        event, values = window.read(timeout=20, timeout_key='timeout')
        if event == sg.WIN_CLOSED:      
            break
        elif event == '1.Register':
            register = True
            count = 0
        elif event == '2.Train':   
            info = Train(dataPath)
        elif event == '3.Recognize':   
            recognizeFrame = True

        name = values["name"]
        Id = values["Id"]
        course = values["course"]
        
        ret, frame = cap.read()

        if register:
            createImages(frame,count)
            info = "Saving "+str(count)
            count = count + 1
            if count > sampleNum:
                row = [Id, name,course]
                info = writeDatabase(databaseFile,row)
                register = False
        if recognizeFrame:
            frame,info =  TrackImages(frame)
            
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  
        window['image'].update(data=imgbytes)
        window['title'].update(info)
            
    # Release the webcam and close the window   
    cap.release()
    cv2.destroyAllWindows()
    window.close()
    
# Start the main program =============
main()
