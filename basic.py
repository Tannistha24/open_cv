import cv2 as cv 

img=cv.imread("C:\\Users\\tanni\\OneDrive\\Pictures\\43 - J5nzSCU.jpg")
cv.imshow('Wallpaper', img)

# converting to grayscale
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# Blur a image
blur=cv.GaussianBlur(img,(9,9),cv.BORDER_DEFAULT)
cv.imshow("Blur",blur)

cv.waitKey(0)


