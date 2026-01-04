import cv2
import numpy as np
import handtrack_basics as hb
import time
import autopy

wCam,hCam=640,480
frameR=100  #frame Reduction 
smoothening = 20
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)
pTime=0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
detector=hb.HandDetector(maxHands=1)
wScr,hScr=autopy.screen.size()  
while True:
    success, img = cap.read()
    img=detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    # fingers = detector.fingersUp()

#     lmList,bbox=detector.findPosition(img,draw=False)
# # check the tip of the index fingers and middle fingers
#     if len(lmList)!=0:
#         x1,y1=lmList[0][1:]
#         x2,y2=lmList[12][1:]
# # check which finger is up 
#         fingers=detector.fingersUp()
    # lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]     # Index finger tip
        x2, y2 = lmList[12][1:]    # Middle finger tip

        fingers = detector.fingersUp()

        print("Fingers:", fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
            (255, 0, 255), 2)
        # only index finger:Moving mode
        if fingers[1]==1 and fingers[2]==0:
            # convert coordinates
            
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # mirror X
            x3 = wScr - x3
            # SAFETY CLAMP
            x3 = np.clip(x3, 0, wScr - 1)
            y3 = np.clip(y3, 0, hScr - 1)

            # smoothen values
            clocX=plocX+(x3-plocX)/smoothening
            clocY=plocY+(y3-plocY)/smoothening
            # move mouse 
            autopy.mouse.move(x3,y3)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # both index and middle fingers are up:clicking mode
        if fingers[1]==1 and fingers[2]==1:
            # find the distance between fingers
            length,img,info=detector.findDistance(8,12,img)
            #  click mouse if distance is short
            if length < 40:
                cv2.circle(img, (info[4], info[5]),
                15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

# frame rate
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