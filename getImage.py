
# coding: utf-8

# In[3]:


import cv2
import sys
import logging as log
import datetime as dt
from time import sleep


# In[4]:


#load OpenCV face detector, I am using LBP which is fast
#there is also a more accurate but slow Haar classifier
face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')


# In[5]:


#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# In[6]:


def drawRectangle(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    
    try:
        if rect == None:
            return False, img
    except(ValueError):
        pass
    
    #draw a rectangle around face detected
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return True, img


# In[7]:


video_capture = cv2.VideoCapture(0)
picCount = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if ret:
        isRact, img = drawRectangle(frame)
        # Display the resulting frame
        cv2.imshow('Video', img)
    
        inputKey = cv2.waitKey(1)
        if isRact:
            status = cv2.imwrite('/home/user/Project/faceRag/my-NewData/img'+str(picCount)+'.jpg',frame)
            print('/home/user/Project/faceRag/my-NewData/img'+str(picCount)+'.jpg have been saved.')
            picCount += 1
            inputKey = -2
        if inputKey == ord('q') or picCount >100:
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

