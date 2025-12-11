import cv2 as cv 
import numpy as np

img=cv.imread("C:\\Users\\tanni\\OneDrive\\Pictures\\43 - J5nzSCU.jpg")
cv.imshow('Wallpaper', img) 

def rescaleFrame(frame,scale=0.5):
    # live videos,images,videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img)

def translate(img,x,y):
    transMat=np.float32([[1,0,x],[0,1,y]])
    dimanesions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimanesions)

translated=translate(img,100,180)
cv.imshow("translated",translated)


def rotate(resized_img,angle,rotPoint=None):
    (height,width)=resized_img.shape[:2]

    if rotPoint is None:
        rotPoint=(width//2,height//2)

    rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(width,height)

    return cv.warpAffine(resized_img,rotMat,dimensions)
rotated=rotate(resized_img,45)
cv.imshow("rotated",rotated)

# resize
resized=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow("Resized",resized)

# flippping
flip=cv.flip(resized_img,0) #0=vertical,1=hprizontal
cv.imshow("Flipped",flip)

# cropping
cropped=resized_img[200:400,300:400]
cv.imshow("cropped",cropped)

cv.waitKey(0)