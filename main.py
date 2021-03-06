import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.utils import source
 
 
def nothing(x):  # 滑动条的回调函数
    pass
 
 
imagePath = "Image/img4.png"
tempPath = "Temp/binary.png"
source = cv2.imread(imagePath)
print(source.shape)
src = cv2.resize(source,(int(source.shape[0]), int(source.shape[1])))
srcBlur = cv2.GaussianBlur(src, (3, 3), 0)

gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
retval, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite(tempPath, dst)

gray = cv2.imread(tempPath,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
WindowName = 'Approx'  # 窗口名
cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)  # 建立空窗口
 
cv2.createTrackbar('threshold', WindowName, 0, 60, nothing,)  # 创建滑动条
 
while(1):
    img = src.copy()
    threshold = 100 + 2 * cv2.getTrackbarPos('threshold', WindowName)  # 获取滑动条值
 
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
 
    if lines is None:
        cv2.imshow(WindowName, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        continue
 
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
 
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
 
    cv2.imshow(WindowName, img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()