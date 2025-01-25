import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import cv2 as cv
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, model = 0, detectionCon = 0.5):
        self.model = model
        self.detectionCon = detectionCon

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(model_selection = self.model, 
                                         min_detection_confidence=self.detectionCon)
        self.mpDraw = mp.solutions.drawing_utils

        
    def findFace(self, img, draw=True):
        img.flags.writeable = False
        imgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.face.process(imgRgb)

        img.flags.writeable = True
        if self.results.detections:
            for detection in self.results.detections:
                self.mpDraw.draw_detection(img, detection)
        return img

    def findPosition(self, img, draw=True):
        dtList = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # dtList.append([id, detection])
                # print(id, detection)
                # print(detection.score)
                print(detection.location_data.relative_bounding_box)



def main():
    startTime = 0
    currentTime = 0
    detector = FaceDetector()

    cap = cv.VideoCapture(0)

    while cap.isOpened():
        sucess, img = cap.read()
        img = cv.flip(img, 1)
        img = detector.findFace(img)
        detector.findPosition(img)

        currentTime = time.time()
        fps = 1 / (currentTime - startTime)
        startTime = currentTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (0, 255, 0), 3)
        
        cv.imshow("Face", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()