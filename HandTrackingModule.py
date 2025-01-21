import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations


import cv2 as cv
import mediapipe as mp
import time
import tensorflow as tf


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
            img = cv.flip(img, 1)
            ImgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = self.hands.process(ImgRGB)
            # print(results.multi_hand_landmarks)

            if results.multi_hand_landmarks:
                for handLMS in results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)

                    # for id, lm in enumerate(handLMS.landmark):
                    #     h, w, c = img.shape
                    #     cx, cy = int(lm.x*w), int(lm.y*h)
                    #     print(id, cx, cy)
                    #     if id == 4:
                    #         cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            return img
    



def main():
    startTime = 0
    currentTime = 0

    cap = cv.VideoCapture(0)
    detector = HandDetector()

    while cap.isOpened():
        success, img = cap.read()
        img = detector.findHands(img)

        currentTime = time.time()
        fps = 1/(currentTime - startTime)
        startTime = currentTime

        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)    

        cv.imshow('Image', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()