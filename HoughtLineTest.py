from sys import flags
import cv2
import numpy as np

def isInRange(data,posData,range):
    if data >= posData - range and data <= posData + range:
        return True
    else:
        return False

def isCompareAllLine(rho,theta,inputLines,rangeOf_rho,rangeOf_theta):
    # rho & theta -:
    for i in range(len(inputLines)):
        rho_flag = False
        theta_flag = False
        
        if isInRange(rho,inputLines[i][0],rangeOf_rho):
            rho_flag=True
        
        if isInRange(theta,inputLines[i][1],rangeOf_theta):
            theta_flag=True
        
        if rho_flag and theta_flag:
            print("is same line, so give up")
            return False
    
    print("Not same line, so append")
    return True


def mergeLines(rho,theta,inputLines,rangeOf_rho,rangeOf_theta):
    # rho:
    for i in range(len(inputLines)):
        if isInRange(rho,inputLines[i][0],rangeOf_rho) and isInRange(theta,inputLines[i][1],rangeOf_theta):
            rho_now = (rho + inputLines[i][0]) / 2
            inputLines[i][0] = rho_now
            
            theta_now = (theta + inputLines[i][1]) / 2
            inputLines[i][1] = theta_now
            
            return inputLines
        

def findSide(lines):
    """
    @param: lines: 输入的线段
    @param: extent: 控制判断的幅度
    """
    resultLine = []
    rangeOf_rho = 50
    rangeOf_theta = 0.1
    for line in lines:
        print("now:",resultLine,"line is",line)
        rho = line[0][0]
        theta = line[0][1]
        if not resultLine:
            resultLine.append([rho,theta])
        elif isCompareAllLine(rho,theta,resultLine,rangeOf_rho,rangeOf_theta):
            print("++++++++++++")
            resultLine.append([rho,theta])
        else:
            print("--------------------")
            resultLine = mergeLines(rho,theta,resultLine,rangeOf_rho,rangeOf_theta)
    
    return resultLine
      
    

imagePath = "Image/img7.png"

src = cv2.imread(imagePath)
srcBlur = cv2.GaussianBlur(src, (5, 5), 0)

gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
retval, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#cv2.imshow("gray",gray)
#cv2.imshow("dst",dst)
#cv2.imshow("Canny",edges)
#cv2.waitKey(0)

image = src.copy()
threshold = 100 + 2 * 5
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
print(len(lines))

"""
rho （0,0）到直线的距离，rho的正负判定：Y轴截距大于0的均为正，Y轴截距小于0则rho为负。
theta为直线与Y轴负方向的夹角，以Y轴负轴为起始轴，逆时针旋转到直线的角度。
"""
for line in lines:
    print("line:", line)
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
    
    #cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #cv2.circle(image,(x0,y0),3,(255,0,0),-1)

newLines = findSide(lines)
print(newLines)
for newLine in newLines:
    rho = newLine[0]
    theta = newLine[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(image,(int(x0),int(y0)),3,(255,0,0),-1)

cv2.imshow("Windows",image)
cv2.waitKey(0)

    