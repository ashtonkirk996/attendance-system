import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# create a list of filenames in the face image folder
path = 'attendance-system/Face_Images'
images = []
classNames = []
fileNameList = os.listdir(path)

# iterate through the names in the list and store the images and names in separate arrays
for cl in fileNameList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# iterate through the imgages and get the encodings from the face_recognition library
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# record the name and time of a detected person inside a csv file
def markAttendance(name):
    with open('attendance-system/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            

# generate a list of encodings for the images
encodeList =  findEncodings(images)

# turn on webcam video capture
cap = cv2.VideoCapture(0)

# read each frame of the webcam input, encode it and compare it to the list of encoded images on file
while True:
    success, img = cap.read()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(img)
    encodingsCurFrame = face_recognition.face_encodings(img,facesCurFrame)

# compare the current encoding with those in the list and find the closest match
    for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeList,encodeFace)
        faceDis = face_recognition.face_distance(encodeList,encodeFace)
        matchIndex = np.argmin(faceDis)

# if the closest match is found then display a green box with the name of the person
        if matches[matchIndex]:
           name = classNames[matchIndex].upper()
           y1,x2,y2,x1 = faceLoc
           cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,0),2)
           cv2.rectangle(img,(x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
           cv2.putText(img, name,(x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
           markAttendance(name)

#show the webcam on-screen
    cv2.imshow('WebCam',img)
    cv2.waitKey(1)