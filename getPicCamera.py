import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

video_capture = cv2.VideoCapture(0)
picCount = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    inputKey = cv2.waitKey(1)
    if inputKey == ord('c'):
        status = cv2.imwrite('/home/user/Project/faceRag/my-NewData/pic'+str(picCount)+'.jpg',frame)
        print('/home/user/Project/faceRag/my-NewData/pic'+str(picCount)+'.jpg have been saved.')
        picCount += 1
        inputKey = -2
    elif inputKey == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
