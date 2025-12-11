import cv2 as cv 
import matplotlib.pyplot as plt

img=cv.imread("IMG_6271.jpg")
cv.imshow('Wallpaper', img)


def rescaleFrame(frame,scale=0.1):
    # live videos,images,videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img)

plt.imshow(resized_img)
plt.show()
cv.imshow('Resized img',resized_img)
#BGR to grayscale
gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)

# BGR to HSV
hsv=cv.cvtColor(resized_img,cv.COLOR_BGR2HSV)
cv.imshow('hsv',hsv)

# BGR to L*a*b
lab=cv.cvtColor(resized_img,cv.COLOR_BGR2LAB)
cv.imshow('LAB',lab)



cv.waitKey(0)