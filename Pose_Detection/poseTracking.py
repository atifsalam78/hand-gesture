import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# import mediapipe as mp
import cv2 as cv
import time
import PoseTrackingModule as ptm

cap = cv.VideoCapture(0)

# cap = cv.VideoCapture("E:/Codes/videos/05.mp4")

startTime = 0
currentTime = 0
detector = ptm.PoseDetector()

while cap.isOpened():
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPositon(img, draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), 5, cv.FILLED)
    img = cv.flip(img, 1)
    

    currentTime = time.time()
    fps = 1 / (currentTime - startTime)
    startTime = currentTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)

    cv.imshow("Pose", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break