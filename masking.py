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

blank=np.zeros(resized_img.shape[:2],dtype='uint8')

mask=cv.circle(blank,(resized_img.shape[1]//2,resized_img.shape[0]//2),100,255,-1)

circle = cv.circle(blank.copy(), (30, 30), 370, 255, -1)

cv.imshow('Mask',mask)
werid_shape=cv.bitwise_and(mask,circle)

masked=cv.bitwise_and(resized_img,resized_img,mask=mask)
cv.imshow('Masked Image',masked)


cv.waitKey(0)