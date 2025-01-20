import cv2 as cv
import mediapipe as mp
import time

import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf 


# Create video object

cap = cv.VideoCapture(0)



mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, 
                         max_num_hands=2, 
                         min_detection_confidence=0.5, 
                         min_tracking_confidence=0.5) # if we go for deault parameters keep it blank

mpDraw = mp.solutions.drawing_utils

startTime = 0
currentTime = 0

while cap.isOpened():
    success, img = cap.read()
    ImgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(ImgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    cv.circle(img, (cx, cy), 25, (255, 0, 255), cv.FILLED)
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)
    currentTime = time.time()
    fps = 1/(currentTime - startTime)
    

    cv.imshow('Image', cv.flip(img, 1))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break