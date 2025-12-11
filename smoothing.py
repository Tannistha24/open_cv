import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np

img=cv.imread("IMG_6271.jpg")
# cv.imshow('Wallpaper', img)


def rescaleFrame(frame,scale=0.1):
    # live videos,images,videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img)

average=cv.blur(resized_img,(7,7))
cv.imshow('Average blur',average)

# Gaussian blur
gauss=cv.GaussianBlur(resized_img,(7,7),0)
cv.imshow('Gauss',gauss)

# Median Blur
median=cv.medianBlur(resized_img,7)
cv.imshow("Median Blur",median)

# Bilateral
bilateral=cv.bilateralFilter(resized_img,5,15,15)
cv.imshow("BIlateral",bilateral)

cv.waitKey(0)