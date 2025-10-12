import cv2 as cv 

img=cv.imread("C:\\Users\\tanni\\OneDrive\\Pictures\\43 - J5nzSCU.jpg")
# cv.imshow('Wallpaper', img)
def rescaleFrame(frame,scale=0.5):
    # live videos,images,videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img)

# converting to grayscalegit
# gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray',gray)

# # Blur a image
# blur=cv.GaussianBlur(resized_img,(9,9),cv.BORDER_DEFAULT)
# cv.imshow("Blur",blur)

# Edge cascade
canny=cv.Canny(resized_img,125,175)
cv.imshow("canny",canny)

# ilating the image
dilated=cv.dilate(canny,(3,3),iterations=1)
cv.imshow("Dilated",dilated)

# eroded
eroded=cv.erode(dilated,(3,3),iterations=1)
cv.imshow("eroded",eroded)

cv.waitKey(0)


