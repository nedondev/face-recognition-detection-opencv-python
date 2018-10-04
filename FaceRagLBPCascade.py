
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
from time import sleep

#SETTINGS
facesList, labelsList = [[],[],[],[]], [[],[],[],[]]
subjects = ["Box1", "Box2", "Box3", "Box4"]
CONFIDENT_RATE = 50
STREAM_NAME = "Face unlock Locker"
REGISTER_FACE_SIZE = 40000
CAMERA_NUM = 0
# In[2]:


face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')


# In[3]:


def detect_face(img):
    #image to gray image as opencv face detector expects gray images
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


# In[13]:


def drawRectangleOnDetectedFace(test_img, faceSize = 0):
    #make a copy of the image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    try:
        if rect == None:
            return False, img ,None
    except(ValueError):
        pass
    if len(face)*len(face[0]) < faceSize:
        return False, img ,None
    #draw a rectangle around face detected
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return True, img , face


# In[15]:


def registerFace(label, video_capture):
    faces = []
    labels = []
    picCount = 0
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            continue
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if ret:
            isRact, img, face = drawRectangleOnDetectedFace(frame, REGISTER_FACE_SIZE)
            # Display the resulting frame
            cv2.imshow(STREAM_NAME, img)

            inputKey = cv2.waitKey(1)
            if isRact :
                faces.append(face)
                labels.append(label)
                print('face'+str(picCount)+'have been added.')
                picCount += 1
            if inputKey == ord('q') or picCount >50:
                break

    facesList[label], labelsList[label] = faces, labels
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    for i, j in zip(facesList, labelsList):
        face_recognizer.update(i,np.array(j))
    print("Register to Box",int(label)+1)
    return face_recognizer


# In[6]:


#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# In[7]:


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img, face_recognizer):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    
    try:
        if rect == None:
            return img
    except(ValueError):
        pass
    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #print(confidence , face_recognizer.getThreshold())
        
    if confidence < CONFIDENT_RATE:
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img


# In[16]:


def main():
    video_capture = cv2.VideoCapture(CAMERA_NUM)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    trained = False
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            continue
            pass
        inputKey = cv2.waitKey(1)
        if inputKey == ord('q'):
            break
        elif inputKey == ord('r'):
            label = int(input('Input box number(1-4): '))-1
            face_recognizer = registerFace(label, video_capture)
            trained = True
        else :
            ret, frame = video_capture.read()

            if ret:
                isRact, img, face = drawRectangleOnDetectedFace(frame)
                if isRact and trained:
                    img = predict(img, face_recognizer)
                cv2.imshow(STREAM_NAME, img)
    video_capture.release()
    cv2.destroyAllWindows()
                    
if __name__ == "__main__":
    main()

