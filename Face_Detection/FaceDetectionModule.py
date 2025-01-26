import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import cv2 as cv
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, model = 0, detectionCon = 0.75):
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
        # if self.results.detections:
        #     for detection in self.results.detections:
        #         self.mpDraw.draw_detection(img, detection)
        return img

    def findPosition(self, img, draw=True):
        dtList = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # dtList.append([id, detection])
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape

                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                cv.rectangle(img, bbox, (255, 0, 0), 2)
                cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                           cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0), 2)



def main():
    startTime = 0
    currentTime = 0
    detector = FaceDetector()

    cap = cv.VideoCapture(0)
    # cap = cv.VideoCapture("E:/Codes/videos/11.mp4")
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640 pixels
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480 pixels

    # # Create a named window and resize it
    # cv.namedWindow("Image", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Image", 640, 480)

    while cap.isOpened():
        sucess, img = cap.read()
        if not sucess:
            break

        img = cv.flip(img, 1)
        # img = cv.resize(img, (800,640))

        img = detector.findFace(img)
        detector.findPosition(img)

        currentTime = time.time()
        fps = 1 / (currentTime - startTime)
        startTime = currentTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 0), 3)
        
        cv.imshow("Face", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()