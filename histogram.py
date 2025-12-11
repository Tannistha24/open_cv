import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt

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

# Grayscale histogram
gray_hist=cv.calcHist([gray],[0],None,[256],[0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel('Bins')
plt.ylabel('pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

# Color Histogram
colors=('b','g','r')
for i,col in enumerate(colors):
    hist = cv.calcHist([resized_img], [i], None, [256], [0, 256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])

plt.show()

cv.waitKey(0)



