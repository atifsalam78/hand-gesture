import cv2 as cv
import mediapipe as mp
# import time
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
            self.results = self.hands.process(ImgRGB)
            # print(results.multi_hand_landmarks)

            if self.results.multi_hand_landmarks:
                for handLMS in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)

            return img
    

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255,0,255), cv.FILLED)
        return lmList

    



def main():
    pass
    


if __name__ == "__main__":
    main()