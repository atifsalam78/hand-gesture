import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import cv2 as cv
# import mediapipe as mp
import time
# import tensorflow as tf
import HandTrackingModule as htm


startTime = 0
currentTime = 0

cap = cv.VideoCapture(0)
detector = htm.HandDetector()
    

while cap.isOpened():
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        print(lmlist[4])
    currentTime = time.time()
    fps = 1/(currentTime - startTime)
    startTime = currentTime

    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)    

    cv.imshow('Image', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break