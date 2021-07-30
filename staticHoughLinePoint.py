import cv2
import numpy as np

imagePath = "Image/img3.jpg"    # 原图
tempPath = "Temp/binary3.png"   # 缓存图片
edgesPath = "Temp/edges.jpg"    # 边缘检测缓存图，可删
zoom = 0.2

"""
原图太大了，先给他做一个缩放，这一步在实际情况中可以去掉或者做成自己的
"""
source = cv2.imread(imagePath)
print("原图shape:",source.shape)
src = cv2.resize(source,(int(source.shape[1] * zoom), int(source.shape[0] * zoom)))
print("缩放后shape:",src.shape)

# 高斯滤波, 中值滤波，卷积核用(5,5)
gaussBlur = cv2.GaussianBlur(src, (5, 5), 0)
medianBlur = cv2.medianBlur(src, 5)

# 灰度化
gray = cv2.cvtColor(medianBlur,cv2.COLOR_BGR2GRAY)

# 二值化
retval, dst = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# 腐蚀操作 -  不使用，会把寻找范围变大
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))
eroded = cv2.erode(kernel,kernel)

# canny 边缘算子算法
edges = cv2.Canny(dst, 50, 150, apertureSize=3)

#cv2.imshow("src",src)
#cv2.imshow("gaussBlur",gaussBlur)
#cv2.imshow("medianBlur",medianBlur)
#cv2.imshow("binary",dst)
cv2.imshow("edges",edges)
#cv2.imshow("eroded",eroded)
#cv2.waitKey(0)


"""
HoughLinesP 参数：
src：输入图像，必须8-bit的灰度图像
rho：生成极坐标时候的像素扫描步长
theta：生成极坐标时候的角度步长
threshold：阈值，只有获得足够交点的极坐标点才被看成是直线
lines：输出的极坐标来表示直线
minLineLength：最小直线长度，比这个短的线都会被忽略。
maxLineGap：最大间隔，如果小于此值，这两条直线 就被看成是一条直线。
"""

img = src.copy()
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 0, 0, 0)

for line in lines:
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[0][2]
    y2 = line[0][3]
    #print(line)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)


print("发现线段数量:",len(lines))

cv2.imshow("HoughLines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()