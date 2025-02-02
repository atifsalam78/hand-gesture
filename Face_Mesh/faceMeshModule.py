import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, mode=True, maxFaces = 1, refLandmarks = True, detectionCon = 0.5 ):
        self.mode = mode
        self.maxFaces = maxFaces
        self.refLandmarks = refLandmarks
        self.detectionCon = detectionCon

        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.mpDrawStyle = mp.solutions.drawing_styles
        self.mpFaceMesh = mp.solutions.face_mesh

        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.mode, max_num_faces=self.maxFaces,
                                       refine_landmarks=self.refLandmarks, min_detection_confidence=self.detectionCon)

    def findFaceMesh(self, image, draw=True):
        imgRgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRgb)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLMS in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, faceLMS, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLMS.landmark):
                        ih, iw, ic = image.shape
                        x, y, z = int(lm.x*iw), int(lm.y*ih), int(lm.z*ic)
                        # cv.putText(image, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 0.50, (0,255,0), 1)
                        face.append([x, y])
                    faces.append(face)
        return image, faces    

def main():
    startTime = 0
    currentTime = 0

    cap = cv.VideoCapture(0)

    detector = FaceMeshDetector(maxFaces = 2)

    while cap.isOpened():
        sucess, image = cap.read()

        image, faces = detector.findFaceMesh(image)
        
        image = cv.flip(image, 1)
        if len(faces) != 0:
            print(faces[0])

        currentTime = time.time()
        fps = 1 / (currentTime-startTime)
        startTime = currentTime

        cv.putText(image, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3)  
        
        
        cv.imshow("Face Mesh", image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()