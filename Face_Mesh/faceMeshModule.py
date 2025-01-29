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


    def findFace(self, image, draw=True):
        imgRgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRgb)
        if results.multi_face_landmarks:
            for faceLMS in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, faceLMS, self.mpFaceMesh.FACEMESH_TESSELATION,
                                           self.drawSpec, self.drawSpec)
        return image

    def findPosition(self, image, draw=True):
        pass



def main():
    startTime = 0
    currentTime = 0

    cap = cv.VideoCapture(0)

    detector = FaceMeshDetector()
    


    while cap.isOpened():
        sucess, image = cap.read()

        image = detector.findFace(image)
        image = cv.flip(image, 1)

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