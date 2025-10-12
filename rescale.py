import cv2 as cv # type: ignore

img=cv.imread("C:\\Users\\tanni\\OneDrive\\Pictures\\43 - J5nzSCU.jpg")
cv.imshow('Wallpaper', img)

def rescaleFrame(frame,scale=0.75):
    # live videos,images,videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img)
cv.imshow('Resized img',resized_img)

def changeRes(width,height):
    #live video
    capture.set(3,width)
    capture.set(4,height)
# reading videos
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()

    frame_resized=rescaleFrame(frame)
    cv.imshow('Video', frame)
    cv.imshow('Video Resized',frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):  # Changed from waitkey to waitKey
        break
capture.release()
cv.destroyAllWindows()