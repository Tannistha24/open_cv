import cv2 as cv
import numpy as np

img=cv.imread("C:\\Users\\tanni\\OneDrive\\Pictures\\43 - J5nzSCU.jpg")
# cv.imshow('Wallpaper', img)

def rescaleFrame(frame,scale=0.5):
    # live videos,images,videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img)

gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)

# Laplaction
lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
# cv.imshow('Laplactian',lap)

#  Sobel
sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobely=cv.Sobel(gray,cv.CV_64F,0,1)
combined_sobel=cv.bitwise_or(sobelx,sobely)


cv.imshow('Sobel X',sobelx)
cv.imshow('Sobel Yl',sobely)
cv.imshow('Combined Sobel',combined_sobel)



cv.waitKey(0)