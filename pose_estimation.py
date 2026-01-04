import cv2
import mediapipe as mp # allow su to get our pose estimation 
import time

class poseDetector():
    def __init__(self,mode=False,smooth=True,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        print(self.results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList


def main():
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    pTime=0
    detector=poseDetector()
    
    
    while True:
        success, img = cap.read()
        detector.findPose(img)
        lmList=detector.getPosition(img)
        print(lmList)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    main()

