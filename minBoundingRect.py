from types import new_class
import cv2
import numpy as np
from numpy.lib.function_base import _select_dispatcher
from numpy.lib.type_check import imag

imagePath = "Image/img4.png"
image = cv2.imread(imagePath)
# 图像转灰度图
#img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 图像转二值图
#ret, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)


# 高斯滤波，灰度，二值化，Canny算子
srcBlur = cv2.GaussianBlur(image, (5, 5), 0)
gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
retval, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(dst, 50, 150, apertureSize=3)

cv2.imwrite("edges.png",edges)

# 功能：cv2.findContours()函数来查找检测物体的轮廓。
#参数:
# 参数1：寻找轮廓的图像，接收的参数为二值图，即黑白的（不是灰度图），所以读取的图像要先转成灰度的，再转成二值图
# 参数2: 轮廓的检索模式，有四种。
#       cv2.RETR_EXTERNAL 表示只检测外轮廓;
#       cv2.RETR_LIST 检测的轮廓不建立等级关系;
#       cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
#       cv2.RETR_TREE 建立一个等级树结构的轮廓。
#
# 参数3: 轮廓的近似办法.
#       cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
#       cv2.CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
#       cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法
# 注：opencv2返回两个值：contours：hierarchy。opencv3会返回三个值,分别是img, countours, hierarchy
#
#返回值
#cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

maxArea = 0
maxArea_i = 0
# 找到最大的矩形位置
for i in range(len(contours)):
    rect = cv2.minAreaRect(contours[i])
    
    w,h = rect[1][0],rect[1][1]
    area = w * h
    
    if maxArea < area:
        maxArea = area
        maxArea_i = i

for c in contours:
    # 找到边界坐标
    area = cv2.contourArea(c)
    if area <= 10:
        continue

    # 找面积最小的矩形
    rect = cv2.minAreaRect(c)

    w, h = rect[1][0],rect[1][1]
    arr = w * h
    print(rect,w,h,arr)
    
    # 得到最小矩形的坐标
    box = cv2.boxPoints(rect)   # 转换C#代码时候需要注意点
    
    # 标准化坐标到整数
    box = np.int0(box)
    
    # 画出边界
    cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    
    """     
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])
    
    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    """




cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
cv2.imshow("img", image)
cv2.imwrite("img_1.jpg", image)
cv2.waitKey(0)