import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        self.bbox = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            xList = []
            yList = []

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            self.bbox = min(xList), min(yList), max(xList), max(yList)

        return self.lmList, self.bbox
    
    def fingersUp(self):
        fingers = []

        if not hasattr(self, "lmList") or len(self.lmList) == 0:
            return [0, 0, 0, 0, 0]

        # ---- THUMB (X direction) ----
        if self.lmList[4][1] > self.lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # ---- OTHER FINGERS (DISTANCE BASED) ----
        for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
            tip_y = self.lmList[tip][2]
            pip_y = self.lmList[pip][2]

            # finger is UP only if clearly extended
            if pip_y - tip_y > 20:   # 🔑 threshold
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

                

def main():
    cTime=0
    pTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmlist=detector.findPosition(img)
        # if len(lmlist)!=0:
        #     # print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (10, 70),
            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2
        )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
