'''
Volume Control
Created By: ATif Salam
Created Date: 02 Feb 2025

'''

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import cv2 as cv
import mediapipe as mp
import time
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np



startTime = 0
currentTime = 0

WidthCam, HeightCam  = 740, 480

cap = cv.VideoCapture(0)
cap.set(4, HeightCam)
cap.set(3, WidthCam)

detector = htm.HandDetector(detectionCon=0.7)



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while cap.isOpened():    
    
    sucess, frame = cap.read()
    frame = detector.findHands(frame, draw=True)
    lmList = detector.findPosition(frame, draw=False)

    currentTime = time.time()
    fps = 1 / (currentTime- startTime)
    startTime = currentTime

    cv.putText(frame, f'FPS: {int(fps)}', (10,50), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2) // 2, (y1+y2) // 2
        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Hand Range 50 - 200
        # Volume Range -96 - 0

        vol = np.interp(length, [50, 200], [minVol, maxVol])
        volBar = np.interp(length, [50, 200], [400, 150])
        volPer = np.interp(length, [50, 200], [0, 100])

        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        cv.circle(frame, (x1, y1), 15, (225, 0, 225), cv.FILLED)
        cv.circle(frame, (x2, y2), 15, (225, 0, 225), cv.FILLED)
        cv.line(frame, (x1, y1), (x2, y2), (225, 0, 225), 3)
        cv.circle(frame, (cx, cy), 15, (225,0,225), cv.FILLED)

        if length <50:
            cv.circle(frame, (cx, cy), 15, (0,0,225), cv.FILLED)

    cv.rectangle(frame, (50, 150), (85, 400), (225, 0 , 0), 3)
    cv.rectangle(frame, (50, int(volBar)), (85,400), (225, 0, 0), cv.FILLED)
    cv.putText(frame, f'{int(volPer)}%', (50,450), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


    cv.imshow("Volume Control", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
