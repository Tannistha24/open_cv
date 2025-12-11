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

blank=np.zeros(resized_img.shape[:2],dtype='uint8')

b,g,r=cv.split(resized_img)

blue=cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])

cv.imshow('lue',blue)
cv.imshow('Green',green)
cv.imshow("Red",red)

print(resized_img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merge=cv.merge([b,g,r])
cv.imshow("MErged Image",merge)

cv.waitKey(0)