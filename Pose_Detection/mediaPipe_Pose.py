import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations


import cv2 as cv
import mediapipe as mp
import time 


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

startTime = 0
currentTime = 0

while cap.isOpened():
    success, img = cap.read()
    img = cv.flip(img, 1)
    ImgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(ImgRgb)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv.circle(img, (cx, cy), 3, (255, 0, 0), 5, cv.FILLED) # This will overlay the previous point of detection with ours points
                        

    

    currentTime = time.time()
    fps = 1 / (currentTime - startTime)
    startTime = currentTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)

    cv.imshow("Pose", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break



