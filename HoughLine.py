from __future__ import print_function
import cv2
import numpy as np
import math

"""
整体说明
算法过程：
1. 读取，预处理（高斯滤波，灰度化，二值化，Canny边缘算子），将图片的杂质内容去除
2. 将Canny边缘算子作为HoughLine的传入图片，rho取1，theta取np.pi/180，threshold以100为基数网上增加（本例子取110），minLineLength及maxLineGap默认即可
3. HoughLine返回笛卡尔坐标系下的直线坐标（rho为距离原点坐标, theta为与y轴负方向的角度），命名为Lines
4. 对Lines进行FindSide操作
5. 操作完成后形成的newLines进行坐标系变换，通过公式求出sin，cos，从而复原直线

Note: 经过测试，使用笛卡尔坐标系作为数据区分比较合理，直角坐标系下可能出现，两个直线的0点过近从而误判
Note: 最终结果数据可能为小数，注意转换int()避坑

FindSide操作说明：
判断是否属于一个独立的线段，是则append进入返回列表(resultLine)，反之表示可以和现存的直线进行合并操作
合并操作，先找到返回列表中的线段，当前线段属于哪个返回列表线段的，然后和已有的返回列表线段进行相加/2的操作，意在取平均值

rho （0,0）到直线的距离，rho的正负判定：Y轴截距大于0的均为正，Y轴截距小于0则rho为负。
theta为直线与Y轴负方向的夹角，以Y轴负轴为起始轴，逆时针旋转到直线的角度。
"""

def isInRange(data,posData,range):
    """
    判断数据是否处于范围内
    """
    if data >= posData - range and data <= posData + range:
        return True
    else:
        return False

def isCompareAllLine(rho,theta,inputLines,rangeOf_rho,rangeOf_theta):
    """
    将单个数据和已放入result中的数据进行对比，同时满足rho和theta的算作返回false
    表示当前线段存在相同线段，因此需要放弃
    反之则表示不存在相同的线段，可以把这个线段新添加进入result数组中
    """
    # rho & theta -:
    for i in range(len(inputLines)):
        rho_flag = False
        theta_flag = False
        
        if isInRange(rho,inputLines[i][0],rangeOf_rho):
            rho_flag=True
        
        if isInRange(theta,inputLines[i][1],rangeOf_theta):
            theta_flag=True
        
        if rho_flag and theta_flag:
            return False
    
    return True


def mergeLinesByAverage(rho,theta,inputLines,rangeOf_rho,rangeOf_theta):
    """
    利用平均值法，合并线段数据
    """
    for i in range(len(inputLines)):
        if isInRange(rho,inputLines[i][0],rangeOf_rho) and isInRange(theta,inputLines[i][1],rangeOf_theta):
            rho_now = (rho + inputLines[i][0]) / 2
            inputLines[i][0] = rho_now
            
            theta_now = (theta + inputLines[i][1]) / 2
            inputLines[i][1] = theta_now
            
            return inputLines

def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    """
    点到直线的距离
    """
    a=lineY2-lineY1
    b=lineX1-lineX2
    c=lineX2*lineY1-lineX1*lineY2
    dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
    return dis

def mergeLinesByRencent(rho,theta,inputLines,rangeOf_rho,rangeOf_theta,centPoint_x,centPoint_y):
    """
    利用中心坐标，仅留下最接近的线段
    """
    for i in range(len(inputLines)):
        preDis = 0
        if isInRange(rho,inputLines[i][0],rangeOf_rho) and isInRange(theta,inputLines[i][1],rangeOf_theta):
            rho = inputLines[i][0]
            theta = inputLines[i][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            #lines_ori.append([x1,y1,x2,y2])
            dis = getDis(centPoint_x,centPoint_y,x1,y1,x2,y2)
            if i == 0:
                preDis = dis
            else:
                if(dis < preDis):
                    preDis = dis
                    inputLines[i][0] = rho
                    inputLines[i][1] = theta
                    
    return inputLines
        

def findAverageSide(lines,rangeOf_rho=30,rangeOf_theta=0.1):
    """
    @param: lines: 输入的线段
    @param: rangeOf_rho: 输入的rho范围，
    @param: rangeOf_theta: 输入的theta范围
    """
    resultLine = []
    #rangeOf_rho = 30    # rho 判断的幅度（笛卡尔坐标系距离值）
    #rangeOf_theta = 0.1 # theta 判断的幅度（笛卡尔坐标系角度值）
    for line in lines:
        #print("now:",resultLine,"line is",line)
        rho = line[0][0]
        theta = line[0][1]
        if isCompareAllLine(rho,theta,resultLine,rangeOf_rho,rangeOf_theta):
            resultLine.append([rho,theta])
        else:
            resultLine = mergeLinesByAverage(rho,theta,resultLine,rangeOf_rho,rangeOf_theta)
    
    return resultLine



def findRecentSide(lines,cent_x,cent_y,rangeOf_rho=30,rangeOf_theta=0.1):
    """
    @param: lines: 输入的线段
    @param: cent_x: 输入的中心点x坐标
    @param: cent_y: 输入的中心点y坐标
    """
    resultLine = []
    for line in lines:
        #print("now:",resultLine,"line is",line)
        rho = line[0][0]
        theta = line[0][1]
        
        if isCompareAllLine(rho,theta,resultLine,rangeOf_rho,rangeOf_theta):
            resultLine.append([rho,theta])
        else:
            resultLine = mergeLinesByRencent(rho,theta,resultLine,rangeOf_rho,rangeOf_theta,cent_x,cent_y)
    
    return resultLine

def getCentPoint(contours):
    """
    获取中心点
    """
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
    
    # 获得最小矩形的坐标
    rect = cv2.minAreaRect(contours[maxArea_i])
    print(rect)
    box = cv2.boxPoints(rect)
    print(box)
    
    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])
    
    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    
    centPoint_x = int((left_point_x+right_point_x+top_point_x+bottom_point_x) / 4)
    centPoint_y = int((left_point_y+right_point_y+top_point_y+bottom_point_y) / 4)
    #print(box)
    #print(centPoint_x,centPoint_y)
    print("找到了中心点",centPoint_x,centPoint_y)
    return centPoint_x,centPoint_y
    
      
# 图片地址
imagePath = "Image/img7.png"

# 读取图片
src = cv2.imread(imagePath)

# 高斯滤波，灰度，二值化，Canny算子
srcBlur = cv2.GaussianBlur(src, (5, 5), 0)
gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
retval, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(dst, 50, 150, apertureSize=3)

# 深拷贝image，并且声明 threshold 参数
image = src.copy()
threshold = 55
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)

# 获取边缘，求最小内接矩形
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cent_x, cent_y = getCentPoint(contours)

cv2.circle(image,(int(cent_x),int(cent_y)),3,(255,255,0),-1)    # 标注中心点


print("发现：",len(lines),"条线段")


"""
过滤，最贴近中心点找到的线段
"""
rangeOf_rho = 150
rangeOf_theta = 0.1
lines_rencent = findRecentSide(lines,cent_x,cent_y)
for line in lines_rencent:
    rho = line[0]
    theta = line[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    #cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)     # 画线段
    #cv2.circle(image,(int(x0),int(y0)),3,(255,255,0),-1)    # 标注中心点
    
#rencentLines = findRecentSide(lines_ori,cent_x,cent_y)

"""
过滤经过平均算法找到的新线段
"""
rangeOf_rho = 30
rangeOf_theta = 0.1
newLines = findAverageSide(lines,rangeOf_rho,rangeOf_theta)
print("处理后线段剩余：",newLines," 共",len(newLines),"条")

for newLine in newLines:
    rho = newLine[0]
    theta = newLine[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)     # 画线段
    cv2.circle(image,(int(x0),int(y0)),3,(255,255,0),-1)    # 标注中心点

cv2.imshow("Windows",image)
cv2.imwrite("img3-save.png",edges)
cv2.waitKey(0)


def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [px, py]

def generateLineSegment():
    pass