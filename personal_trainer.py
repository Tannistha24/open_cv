import cv2
import numpy as np
import pose_module as pm

cap = cv2.VideoCapture("Gym.mp4", cv2.CAP_FFMPEG)
detector = pm.poseDetector()

count = 0
stage = "down"   # "down" or "up"
prev_angle = 0


while True:
    success, img = cap.read()
    if not success:
        break

    # ✅ Proper resize (AFTER reading frame)
    img = cv2.resize(img, (400, 500))

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # ---------- RIGHT ARM ----------
        angle = detector.findAngle(img, 12, 14, 16)
        if angle > 180:
            angle = 360 - angle

        angle = np.clip(angle, 40, 160)
        per = int(np.interp(angle, (40, 160), (0, 100)))

        # thresholds (tune if needed)
        UP_THRESHOLD = 70
        DOWN_THRESHOLD = 55

        # ---------- REP LOGIC (FIXED & STABLE) ----------
        if per > UP_THRESHOLD and stage == "down":
            stage = "up"

        elif per < DOWN_THRESHOLD and stage == "up":
            stage = "down"
            count += 1   # ✅ ONLY ONE COUNT PER REP


       

        print("ANGLE:", angle, "PER:", per, "STAGE:", stage)
        # ---------- DISPLAY ----------
        cv2.putText(img, f"Angle: {int(angle)}", (30, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.putText(img, f"Reps: {count}", (30, 90),
                    cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
