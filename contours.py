import cv2 as cv 

img=cv.imread("IMG_6271.jpg")
# cv.imshow('Wallpaper', img)

def rescaleFrame(frame,scale=0.2):
    # live videos,images,videos
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

resized_img=rescaleFrame(img)

gray=cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)
canny=cv.Canny(resized_img,175,175)
cv.imshow("canny img",canny)
contours,hierarchies=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contour(s) found !')

cv.waitKey(0)