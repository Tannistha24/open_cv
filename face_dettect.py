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

gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)
haar_cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_react=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

print(f'Number of faces found={len(faces_react)}')

for(x,y,w,h)in faces_react:
    cv.rectangle(resized_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('detected',resized_img)

cv.waitKey(0)
