import cv2
import numpy as np

imgPath = "Image/img4.png"
#imgPath = "Pic/test1.png"


src = cv2.imread(imgPath,cv2.IMREAD_COLOR)

hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)

dark_n = np.array([0,0,0])
light_n = np.array([80,80,80])

mask = cv2.inRange(hsv,dark_n,light_n)

res = cv2.bitwise_and(src,src,mask=mask)

b_rec =cv2.bitwise_not(res,res)

cv2.imshow('original',src)
cv2.imshow('mask',mask)
cv2.imshow('result',res)
 
cv2.waitKey(0)
cv2.destroyAllWindows()