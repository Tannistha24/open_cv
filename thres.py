import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt

img=cv.imread("C:\\Users\\tanni\\OneDrive\\Pictures\\43 - J5nzSCU.jpg")
cv.imshow('Wallpaper', img)

def rescaleFrame(frame,scale=0.5):
    # live videos,images,videos

    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img)

gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)

# simple thresholding
threshold,thresh=cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow('threshold',thresh)

threshold,thresh_inv=cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
cv.imshow('SIMPLE THRESHOLD INVERSE',thresh_inv)

#  Adoptive Thresholding

adaptive_thresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('Adaptive Threshold',adaptive_thresh)

cv.waitKey(0)


