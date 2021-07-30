import cv2
import numpy as np

#imgPath = "Image/img4.png"
imgPath = "Pic/test3.png"

src =cv2.imread(imgPath)

gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150)

image = src.copy()
threshold = 110

kernel2 = cv2.getStructuringElement(cv2. MORPH_CROSS, (5, 5))
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_3 = np.ones((3,3),np.uint8)
kernel_5 = np.ones((5,5),np.uint8)
kernel_7 = np.ones((7,7),np.uint8)
kernel_9 = np.ones((9,9),np.uint8)
kernel_11 = np.ones((11,11),np.uint8)

#opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_3)
#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_11)

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel2)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel_3)
#closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_7)

#closing = cv2.erode(closing,kernel_5)
#closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_3)

#closing = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT,kernel_3)
#closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_5)

_, dst = cv2.threshold(closing, 110, 255, cv2.THRESH_BINARY)

edges = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(dst,30,90)

cv2.imshow("opening",closing)
cv2.imshow("Canny",edges)
#cv2.imshow("opening",opening)
#cv2.imshow("closing",dst)
#cv2.imshow("src",src)
#cv2.waitKey(0)


lines = cv2.HoughLines(edges,1,np.pi/180,threshold)
print("霍夫直线获取数量为：",len(lines))

pLines = cv2.HoughLinesP(edges,1,np.pi/180,0)
print("霍夫线段获取数量为：",len(pLines))

contours, _ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print("矩形获取数量为:",len(contours))

cv2.imshow("Canny",edges)

for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    #cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 1)

for line in pLines:
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[0][2]
    y2 = line[0][3]
    #cv2.line(dst, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow("after", dst)



max_Area = 0
max_contour = None
for contour in contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(dst, [box], 0, (0, 255, 0), 3)
    #cv2.imshow("contours", dst)
    #cv2.waitKey(0)

#rect = cv2.minAreaRect(contour)
#box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
#box = np.int0(box)
# 画出来
#cv2.drawContours(dst, [box], 0, (0, 255, 0), 3)


#x,y,w,h = cv2.minAreaRect(contour)
#cv2.rectangle(dst,(x,y+h),(x+w,y),(0,255,0))


cv2.imshow("contours", dst)
cv2.waitKey(0)

"""
maxArea = 0
maxArea_i = 0
for i in range(len(contours)):
    rect = cv2.minAreaRect(contours[i])

    area = rect[1][0] * rect[1][1]
        
    if maxArea < area:
        maxArea = area
        maxArea_i = i


rect = cv2.minAreaRect(contours[maxArea_i])
print(rect)

    
x1 = (int)(rect[0][0])
x2 = (int)(rect[0][0]+rect[1][0])
y1 = (int)(rect[0][1])
y2 = (int)(rect[0][1]+rect[1][1])
print(rect)
print(x1,y1,x2,y2)
#cv2.rectangle(image, (240, 0), (480, 375), (0, 255, 0), -1)
cv2.rectangle(image, (x1,y1),(x2,y2), (0,0,255), 1)
    
cv2.imshow("IMAGE",image)
cv2.waitKey(0)
"""