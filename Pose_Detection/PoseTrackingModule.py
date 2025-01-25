import cv2 as cv
import mediapipe as mp
import time


class PoseDetector():
    def __init__(self, mode = False, model_complexity = 1, smooth = True,
                 detectionCon = 0.5, trackingCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode = self.mode, model_complexity = self.model_complexity, 
                                     smooth_landmarks = self.smooth,min_detection_confidence = self.detectionCon,
                                     min_tracking_confidence = self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):

        ImgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(ImgRgb)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findPositon(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 3, (255, 0, 0), 5, cv.FILLED) # This will overlay the previous point of detection with ours points
        return lmList



def main():
    cap = cv.VideoCapture(0)

    startTime = 0
    currentTime = 0
    detector = PoseDetector()

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

if __name__ == "__main__":
    main()