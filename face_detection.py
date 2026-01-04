import cv2
import time
import mediapipe as mp

mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
FaceDetection=mpFaceDetection.FaceDetection()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
pTime = 0

while True:
    success, img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  
    results=FaceDetection.process(imgRGB)

    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img,detection)
            # print(id,detection)
            # print(detection.location_data.relative_bounding_box)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
            int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
           (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
            2, (255, 0, 255), 2)


    if not success:
        print("Failed to read from webcam")
        break

    cTime = time.time()
    if cTime != pTime:
        fps = 1 / (cTime - pTime)
    pTime = cTime

    

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
