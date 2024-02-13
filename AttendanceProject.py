from datetime import datetime
import os
import customtkinter
from tkinter import *
import cv2 
from PIL import Image, ImageTk 
import face_recognition
import numpy as np
  
# Define a video capture object 
vid = cv2.VideoCapture(0) 
  
# Declare the width and height in variables 
width, height = 800, 600
  
# Set the width and height 
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
  
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")



# set app frame
app = customtkinter.CTk()
app.geometry("720x480")
app.title("Attendance Portal")

# add some UI elements
title = customtkinter.CTkLabel(app,text="Please Look Into The Camera")
title.pack(padx=10, pady=10)
  
# Bind the app with Escape keyboard to 
# quit app whenever pressed 
app.bind('<Escape>', lambda e: app.quit()) 
  
# Create a label and display it on app 
label_widget = Label(app) 
label_widget.pack() 

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
  
# Create a function to open camera and 
# display it in the label_widget on app 
  
  
def open_camera(): 
  
    # Capture the video frame by frame 
    _, frame = vid.read() 
  
    # Convert image from one color space to other 
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

           # Redraw the canvas to update the rectangles
          
           cv2.rectangle( frame,(x1,y1),(x2,y2), (0,255,0),2)
           cv2.rectangle(frame,(x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
           cv2.putText(frame, name,(x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
           markAttendance(name)

    # Capture the latest frame and transform to image 
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
    captured_image = Image.fromarray(opencv_image) 
  
    # Convert captured image to photoimage 
    photo_image = ImageTk.PhotoImage(image=captured_image) 
  
    # Displaying photoimage in the label 
    label_widget.photo_image = photo_image 
  
    # Configure image in the label 
    label_widget.configure(image=photo_image) 
  
    # Repeat the same process after every 10 seconds 
    label_widget.after(10, open_camera) 
  
  
# Create a button to open the camera in GUI app 
# button1 = Button(app, text="Open Camera", command=open_camera) 
# button1.pack() 
    
open_camera()
# Create an infinite loop for displaying app on screen 
app.mainloop() 















