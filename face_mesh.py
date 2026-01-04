import cv2
import mediapipe as mp
import time


cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
pTime=0

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)
while True:
    sucess,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
          mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)

          for id,lm in enumerate(faceLms.landmark):
            #print(lm)
            ih, iw, ic = img.shape
            x,y = int(lm.x*iw), int(lm.y*ih)
            print(id,x,y)


    cTime = time.time()
    if cTime != pTime:
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img, str(int(fps)), (10, 70),
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
        )
    cv2.imshow("IMG",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()