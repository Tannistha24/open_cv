import cv2 as cv

img=cv.imread("C:\\Users\\tanni\\OneDrive\\Pictures\\43 - J5nzSCU.jpg")
cv.imshow('Wallpaper', img)

cv.waitKey(0)

# reading videos
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):  # Changed from waitkey to waitKey
        break
capture.release()
cv.destroyAllWindows()